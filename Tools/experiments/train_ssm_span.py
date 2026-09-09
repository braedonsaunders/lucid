#!/usr/bin/env python3
"""Train the selective-scan history branch with a matched SR backbone.

This reconstruction-only recipe requires fresh matched controls. It does not
match r54's paired-critic recipe or establish comparability with r55 native
outputs. Two experimental arms:

  ssm_frozen  - scan module only, backbone frozen (history 2e-4)
  ssm_joint   - scan module plus fused SR backbone (backbone 2e-5)

plus an explicitly trained no-scan control. First-frame output must equal the
backbone's current-frame output exactly. Feature state and RGB history share
the same input-derived motion field.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from train_causal_detail import batch, digest, load_bank
from train_presented_detail import reconstruction_objective, state_digest
from architectures.ssm_recurrent_span import SSMRecurrentSPAN
from native_stages import preprocess_sequence, quantize8
from eval_checkpoint import load
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def run_ssm_sequence(model, decoded, *, use_scan=True, source_motion_policy='raw',
                     return_inputs=False):
    if source_motion_policy not in ('raw', 'taa', 'search'):
        raise ValueError('source motion policy must be raw, taa or search')
    inputs = (decoded if source_motion_policy == 'raw' else
              preprocess_sequence(decoded, motion_policy=source_motion_policy))
    state, outputs = None, []
    for t in range(decoded.shape[1]):
        current = inputs[:, t]
        if t == 0 or not use_scan:
            aligned = current.new_zeros(current.shape[0], 3,
                                        current.shape[-2] * 2, current.shape[-1] * 2)
            confidence = aligned[:, :1]
        else:
            from recurrent_frame_feeding import warp_previous
            with torch.no_grad():
                aligned, confidence, cut, state_grid = warp_previous(
                    decoded[:, t], decoded[:, t - 1], previous_output.detach(),
                    motion_seed='search', return_state_grid=True)
            # The grid is detached correspondence; gradients through the carried
            # features remain intact across time. RGB alignment alone is insufficient.
            state = F.grid_sample(state, state_grid, padding_mode='zeros', align_corners=False)
            # A cut in one batch sample must not reset another sample's history.
            state = torch.where(cut[:, None, None, None], 0, state)
        output, state = model(current, aligned, confidence,
                              state if use_scan else None)
        output = output.clamp(0, 1)
        outputs.append(output)
        previous_output = quantize8(output.detach())
    result = torch.stack(outputs, 1)
    return (result, inputs) if return_inputs else result


def select_validation(bank, reds_bank, manifest, data):
    if reds_bank is None:
        return data['validation'], {r['id']: r['source_id'] for r in manifest['sequences']}
    from replay_recurrent_motion import validation_pairs
    pairs, sources = validation_pairs(bank, manifest)
    reds_pairs, reds_sources = validation_pairs(reds_bank, manifest, reds=True)
    if sources.keys() & reds_sources.keys():
        raise ValueError('duplicate validation identities')
    return pairs + reds_pairs, {**sources, **reds_sources}


def clip_optimizer_groups(optimizer):
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], 1)


@torch.inference_mode()
def validate(model, data, source_ids, *, use_scan=True, baseline=None,
             source_motion_policy='raw'):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        output, inputs = run_ssm_sequence(model, source, use_scan=use_scan,
            source_motion_policy=source_motion_policy, return_inputs=True)
        base = torch.stack([model.sr(inputs[:, t]).clamp(0, 1) for t in range(source.shape[1])], 1)
        variants = [('base', base), ('ssm', output)]
        if baseline is not None:
            variants.append(('initial', torch.stack([baseline(inputs[:, t]).clamp(0, 1)
                             for t in range(source.shape[1])], 1)))
        for variant, values in variants:
            for t in sorted({0, source.shape[1] // 2, source.shape[1] - 1}):
                rows.append({'source_id': source_ids[identity], 'sequence_id': identity,
                    'variant': variant, 'frame': t,
                    'metrics': scorer.spatial(image(values[0, t, :, 8:-8, 8:-8]),
                                              image(reference[0, t, :, 8:-8, 8:-8]))})
            error = values[..., 8:-8, 8:-8] - reference[..., 8:-8, 8:-8]
            temporal.append({'id': identity, 'variant': variant,
                'residual_change_l1': float(torch.diff(error, dim=1).abs().mean()),
                'mean_rgb_mse': float(error.square().mean()),
                'last_rgb_mse': float(error[:, -1].square().mean())})
    return {'complete': True, 'scope': 'source-disjoint full-bank-sequence patch diagnostic; not native playback',
            'use_scan': use_scan, 'source_motion_policy': source_motion_policy,
            'rows': rows, 'summary': summarize(rows), 'temporal': temporal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--source-motion-policy', choices=('raw', 'taa', 'search'), default='raw')
    ap.add_argument('--reds-bank', type=Path, help='Include fixed 16-frame held-out REDS validation')
    ap.add_argument('--joint', action='store_true', help='Optimize the fused SR backbone along with the scan module')
    ap.add_argument('--no-scan', action='store_true', help='Matched joint-training control; scan stays zero and disabled')
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.frames < 3 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch, frames >=3 and even crop >=32 required')
    if args.no_scan and not args.joint:
        ap.error('no-scan control requires a trainable backbone')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] < args.frames or any(min(lr.shape[1:3]) < args.crop for rows in data.values() for lr, _, _ in rows):
        ap.error('adequate bank sequence/crop geometry required')
    validation_data, validation_sources = select_validation(args.bank, args.reds_bank, manifest, data)
    base, _, frames = load(args.init, 'cuda')
    if type(base) is not Unshuffled or frames != 1:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = SSMRecurrentSPAN(base, train_backbone=args.joint).cuda().train()
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False)
    if args.no_scan:
        for p in model.scan.parameters():
            p.requires_grad_(False)
    frozen_hash = state_digest(model.sr.state_dict())
    groups = []
    if not args.no_scan:
        groups.append({'params': model.scan.parameters(), 'lr': .0002, 'base_lr': .0002})
    if args.joint:
        groups.append({'params': model.sr.parameters(), 'lr': .00002, 'base_lr': .00002})
    optimizer = torch.optim.AdamW(groups, lr=.0002, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    files = ['Tools/architectures/ssm_recurrent_span.py',
             'Tools/architectures/subspace_adapter.py',
             'Tools/experiments/recurrent_frame_feeding.py',
             'Tools/experiments/native_stages.py',
             'Tools/experiments/train_causal_detail.py',
             'Tools/experiments/replay_recurrent_motion.py',
             'Tools/experiments/mmap_training_bank.py',
             'Tools/experiments/build_causal_bank.py',
             'Tools/experiments/train_presented_detail.py',
             'Tools/experiments/train_ssm_span.py',
             'Tools/train_span.py',
             'Tools/frontier_eval/evaluate_sequences.py', 'Tools/eval_checkpoint.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'source_hashes': {p: digest(source_root / p) for p in files},
        'initial_scan_sha256': state_digest(model.scan.state_dict()),
        'scan_parameters': model.history_parameters,
        'recipe': 'final-frame HR reconstruction; FP32 fused SR; backpropagation through scan state',
        'backbone_trainable': args.joint, 'use_scan': not args.no_scan,
        'optimizer': 'AdamW/cosine; scan 2e-4, joint backbone 2e-5; no weight decay',
        'gradient_clipping': 'per optimizer group norm1',
        'source_motion_policy': args.source_motion_policy,
        'state_representation': 'aligned_scan_features_v1',
        'motion': 'decoded-LR native integer search plus half-pixel refinement; detached correspondence',
        'limitations': 'crop-local scan state; no native warp/cost parity; no browser timing or release admission',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    if args.reds_bank:
        experiment['reds_bank_sha256'] = digest(args.reds_bank / 'manifest.json')
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, reference = batch(data['train'], rng, args.batch, args.frames, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': reference})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        source, reference = source.cuda(), reference[:, -1].cuda()
        optimizer.zero_grad(set_to_none=True)
        output, inputs = run_ssm_sequence(model, source, use_scan=not args.no_scan,
            source_motion_policy=args.source_motion_policy, return_inputs=True)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            experiment['first_input_sha256'] = state_digest({'inputs': inputs})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        a, b = output[:, -1, :, 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
        loss = reconstruction_objective(a, b, b, 'reference')
        if not torch.isfinite(loss):
            raise ValueError('nonfinite ssm objective')
        loss.backward()
        clip_optimizer_groups(optimizer)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    backbone_unchanged = state_digest(model.sr.state_dict()) == frozen_hash
    if not args.joint and not backbone_unchanged:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'ssm_recurrent_span2x', 'scale': 2,
                'state_representation': 'aligned_scan_features_v1',
                'source_motion_policy': args.source_motion_policy,
                'channels': model.sr.core.conv_1.out_channels, 'version': model.sr.version,
                'step': args.steps, 'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, validation_data, validation_sources,
                      use_scan=not args.no_scan, baseline=initial,
                      source_motion_policy=args.source_motion_policy)
    result['checkpoint_sha256'] = digest(args.out / f'step{args.steps:06d}.pth')
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': backbone_unchanged,
        'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
