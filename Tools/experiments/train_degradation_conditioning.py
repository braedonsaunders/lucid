#!/usr/bin/env python3
"""Matched constant/estimated/oracle degradation-condition experiment."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from clean_lr_targets import batch_clean_lr, load_clean_targets
from train_causal_detail import digest, load_bank
from train_presented_detail import state_digest, reconstruction_objective
from native_stages import preprocess_sequence
from eval_checkpoint import load
from train_span import Unshuffled
from architectures.degradation_conditioning import DegradationConditionedSPAN, oracle_degradation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def condition_for(model, source, oracle):
    if model.mode == 'oracle':
        return oracle
    if model.mode == 'constant':
        return torch.full_like(oracle, .25)
    return model.estimator(source)


@torch.inference_mode()
def validate(model, data, targets, native_inputs, source_ids):
    scorer = Scorer(torch.device('cuda'))
    rows, calibration = [], []
    model.eval()
    image = lambda x: Image.fromarray(x.clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        x = torch.from_numpy(lr[:3].copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        truth = torch.from_numpy(targets[identity][2].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        if native_inputs:
            x = preprocess_sequence(x)
        source = x[:, -1]
        oracle = oracle_degradation(source, truth)
        condition = condition_for(model, source, oracle)
        prediction = model.conditioned(source, condition)
        reference = Image.fromarray(hr[2, 8:-8, 8:-8])
        for variant, output in [('base', model.sr(source)), (model.mode, prediction)]:
            rows.append({'source_id': source_ids[identity], 'sequence_id': identity,
                         'variant': variant, 'frame': 2,
                         'metrics': scorer.spatial(image(output[..., 8:-8, 8:-8]), reference)})
        calibration.append({'id': identity, 'condition_mae': float((condition[..., 4:-4, 4:-4] - oracle[..., 4:-4, 4:-4]).abs().mean())})
    return {'complete': True, 'scope': 'source-disjoint third-frame patch diagnostic; no native output stages or release claim',
            'oracle_status': 'reference-only mechanism diagnostic; not a deployable model or mathematical upper bound',
            'rows': rows, 'summary': summarize(rows), 'calibration': calibration}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'clean-cache', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--mode', choices=('constant', 'estimated', 'oracle'), required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--native-input-stages', action='store_true')
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch and even crop >=32 required')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] < 3 or any(min(lr.shape[1:3]) < args.crop for rows in data.values() for lr, _, _ in rows):
        ap.error('three aligned frames and adequate crop geometry required')
    bank_hash = digest(args.bank / 'manifest.json')
    targets, _ = load_clean_targets(args.clean_cache, data, bank_hash)
    base, _, frames = load(args.init, 'cuda')
    if type(base) is not Unshuffled or frames != 1:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = DegradationConditionedSPAN(base, args.mode).cuda().train()
    frozen_hash = state_digest(model.sr.state_dict())
    parameters = list(model.modulation.parameters())
    if args.mode == 'estimated':
        parameters += list(model.estimator.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.0002, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
                  'clean_target_manifest_sha256': digest(args.clean_cache / 'manifest.json'),
                  'frozen_backbone_sha256': frozen_hash,
                  'estimator_initial_sha256': state_digest(model.estimator.state_dict()),
                  'modulation_initial_sha256': state_digest(model.modulation.state_dict()),
                  'source_hashes': {str(p.relative_to(source_root)): digest(p) for p in [Path(__file__),
                      source_root / 'architectures/degradation_conditioning.py', source_root / 'architectures/subspace_adapter.py',
                      source_root / 'experiments/native_stages.py', source_root / 'experiments/clean_lr_targets.py',
                      source_root / 'experiments/train_presented_detail.py']},
                  'recipe': '3 aligned frames; final-frame reconstruction + 0.5 estimator L1; detached SR condition; frozen backbone; no GAN',
                  'limitation': 'Curvature also responds to real texture. Oracle clean-LR error is supervised damage, not an input-only blockiness measure.',
                  'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                  'purpose': 'mechanism ablation; no release or native quality claim'}
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        x, clean, hr = batch_clean_lr(data['train'], targets, rng, args.batch, 3, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'decoded': x, 'clean': clean, 'reference': hr})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        x, clean, hr = x.cuda(), clean[:, -1].cuda(), hr[:, -1].cuda()
        if args.native_input_stages:
            with torch.no_grad():
                x = preprocess_sequence(x)
        source = x[:, -1]
        oracle = oracle_degradation(source, clean)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            condition = condition_for(model, source, oracle)
            output = model.conditioned(source, condition.detach())
        a, b = output.float()[..., 8:-8, 8:-8], hr[..., 8:-8, 8:-8]
        reconstruction = reconstruction_objective(a, b, b, 'reference')
        calibration = F.l1_loss(condition.float()[..., 4:-4, 4:-4], oracle[..., 4:-4, 4:-4])
        loss = reconstruction + (.5 * calibration if args.mode == 'estimated' else 0)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite degradation-condition objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = .0002 * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'reconstruction': float(reconstruction.detach()),
                              'calibration': float(calibration.detach()), 'minutes': (time.monotonic() - started) / 60}), flush=True)
    if state_digest(model.sr.state_dict()) != frozen_hash:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'degradation_conditioned_span2x',
                'conditioning_mode': args.mode, 'estimator_channels': 16, 'channels': model.core.conv_1.out_channels,
                'scale': 2, 'frames': 1, 'version': model.version, 'step': args.steps, 'experiment': experiment},
               args.out / f'step{args.steps:06d}.pth')
    result = validate(model, data['validation'], targets, args.native_input_stages,
                      {row['id']: row['source_id'] for row in manifest['sequences']})
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'steps': args.steps,
        'backbone_unchanged': True, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
