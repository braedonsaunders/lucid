#!/usr/bin/env python3
"""Calibrate a frozen SR model's variance head and test causal stage policies."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from train_causal_detail import batch, digest, load_bank
from train_presented_detail import state_digest
from native_stages import preprocess_sequence, quantize8
from confidence_stages import confidence_sequence
from architectures.confidence_span import ConfidenceSPAN, gaussian_error_nll
from eval_checkpoint import load
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def calibration_bins(prediction, reference, log_variance):
    variance = log_variance.exp()
    error = (prediction - reference).square().mean(-3, keepdim=True)
    std = variance.sqrt()
    rows = []
    for low, high in zip((0, .01, .02, .05, .1, .2), (.01, .02, .05, .1, .2, 1.01)):
        mask = (std >= low) & (std < high)
        rows.append({'std_range': [low, high], 'pixels': int(mask.sum()),
                     'variance_sum': float(variance[mask].sum()), 'squared_error_sum': float(error[mask].sum())})
    return rows


@torch.inference_mode()
def validate(model, data, source_ids):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal, calibration = [], [], []
    image = lambda a: Image.fromarray(a.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        target = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        for policy in ('fixed', 'constant', 'post', 'causal'):
            output, variance, controls, raw = confidence_sequence(model, source, policy)
            for t in sorted({0, source.shape[1] // 2, source.shape[1] - 1}):
                rows.append({'source_id': source_ids[identity], 'sequence_id': identity, 'variant': policy, 'frame': t,
                             'metrics': scorer.spatial(image(output[0, t, :, 8:-8, 8:-8]), image(target[0, t, :, 8:-8, 8:-8]))})
            error = output[..., 8:-8, 8:-8] - target[..., 8:-8, 8:-8]
            temporal.append({'id': identity, 'variant': policy, 'residual_change_l1': float(torch.diff(error, dim=1).abs().mean()),
                             'rgb_mse': float(error.square().mean()), 'deband_strength_mean': float(controls.mean())})
            calibration.append({'id': identity, 'variant': policy, 'bins': calibration_bins(raw[..., 8:-8, 8:-8],
                target[..., 8:-8, 8:-8], variance[..., 8:-8, 8:-8])})
    return {'complete': True, 'scope': 'source-disjoint full-bank-sequence patch validation through stage proxies; not native playback',
            'variance_target': 'RGB8 reconstruction error before output stages',
            'policies': {'fixed': 'shipping stage strengths', 'constant': 'fixed half-strength deband/sharpen control',
                         'post': 'current confidence controls sharpening only',
                         'causal': 'previous aligned confidence controls debanding; current confidence controls sharpening'},
            'rows': rows, 'summary': summarize(rows), 'temporal': temporal, 'calibration': calibration}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--seed', type=int, default=20260914)
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
    base, _, frames = load(args.init, 'cuda')
    if type(base) is not Unshuffled or frames != 1:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = ConfidenceSPAN(base).cuda().train()
    frozen_hash = state_digest(model.sr.state_dict())
    optimizer = torch.optim.AdamW(model.variance_head.parameters(), lr=.001, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    files = ['architectures/confidence_span.py', 'architectures/subspace_adapter.py', 'experiments/train_confidence_head.py',
             'experiments/confidence_stages.py', 'experiments/native_stages.py', 'experiments/native_output_stages.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'head_parameters': sum(p.numel() for p in model.variance_head.parameters()),
        'source_hashes': {p: digest(root / p) for p in files}, 'confidence_reference_variance': .0025,
        'training': 'three-frame fixed native input proxy; final-frame RGB8 SR Gaussian error NLL; only variance head trained',
        'log_variance_bounds': [-12, 0], 'native_control_status': 'proxy only; no extra current-frame inference for debanding',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, target = batch(data['train'], rng, args.batch, 3, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': target})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        with torch.no_grad():
            inputs = preprocess_sequence(source.cuda())[:, -1]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, log_variance = model.predict_with_uncertainty(inputs)
        output = quantize8(output.float())[..., 8:-8, 8:-8]
        reference = target[:, -1, :, 8:-8, 8:-8].cuda()
        loss = gaussian_error_nll(output, reference, log_variance[..., 8:-8, 8:-8])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite confidence objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.variance_head.parameters(), 1)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = .001 * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'nll': float(loss.detach()), 'minutes': (time.monotonic() - started) / 60}), flush=True)
    if state_digest(model.sr.state_dict()) != frozen_hash:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'confidence_span2x', 'scale': 2, 'frames': 1,
                'channels': model.core.conv_1.out_channels, 'version': model.version, 'step': args.steps,
                'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, data['validation'], {r['id']: r['source_id'] for r in manifest['sequences']})
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': True,
        'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
