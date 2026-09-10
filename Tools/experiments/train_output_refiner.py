#!/usr/bin/env python3
"""Train the output-resolution refiner head with a matched no-head control.

Phase 1 (this script's default): freeze the SPAN trunk, train the head
only. If a trained head on a frozen trunk gives no gain over the frozen
baseline, output-resolution signal carries nothing the trunk could not
already emit, and the idea is dead for a fraction of a full arm's cost.
Phase 2 (only if phase 1 is positive): --joint fine-tunes trunk + head.

Arms: --head-channels 0 is the frozen/no-head baseline (control); 8 and
16 are the A/B refiner heads. Same seed, pinned recipe (--no-history
--joint semantics inherited from the full-LR experiment: trainable
backbone only with --joint; control keeps a matched trainable-nothing
run for step-count parity is the caller's --steps choice, not this
script's). All arms start at exactly the original SR function: the
final head convolution is zero-initialized, so first-step output equals
the backbone output.

Source-proxy diagnostics only; native export, warp parity and timing
unverified.
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

from train_causal_detail import batch, digest, load_bank
from train_presented_detail import state_digest
from architectures.output_refiner import OutputRefinerSPAN, load_output_refiner_checkpoint
from train_ssm_span import select_validation, clip_optimizer_groups, training_objective
from eval_checkpoint import load
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def run_head_sequence(model, decoded, *, use_head=True, source_motion_policy='raw',
                      return_inputs=False):
    if source_motion_policy not in ('raw', 'taa', 'search'):
        raise ValueError('source motion policy must be raw, taa or search')
    if source_motion_policy != 'raw':
        from native_stages import preprocess_sequence
        feeding = preprocess_sequence(decoded, motion_policy=source_motion_policy)
    else:
        feeding = decoded
    outputs = []
    for t in range(decoded.shape[1]):
        current = feeding[:, t]
        if not use_head and model.head is not None:
            with torch.no_grad():
                base = model.sr(current).clamp(0, 1)
            outputs.append(base)
        else:
            outputs.append(model(current).clamp(0, 1))
    result = torch.stack(outputs, 1)
    return (result, feeding) if return_inputs else result


@torch.inference_mode()
def validate(model, data, source_ids, *, use_head=True, baseline=None,
             source_motion_policy='raw'):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        output, inputs = run_head_sequence(model, source, use_head=use_head,
            source_motion_policy=source_motion_policy, return_inputs=True)
        base = torch.stack([model.sr(inputs[:, t]).clamp(0, 1) for t in range(source.shape[1])], 1)
        current_only = run_head_sequence(model, source, use_head=False,
            source_motion_policy=source_motion_policy)
        variants = [('base', base), ('current_only', current_only), ('refined', output)]
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
            'use_head': use_head, 'source_motion_policy': source_motion_policy,
            'rows': rows, 'summary': summarize(rows), 'temporal': temporal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--source-motion-policy', choices=('raw', 'taa', 'search'), default='raw')
    ap.add_argument('--state-channels', type=int, default=8,
        help='Pinned recipe parity with the full-LR experiment; the refiner carries no state and ignores this')
    ap.add_argument('--head-channels', type=int, default=8,
        help='Output-resolution refiner width; 0 is the matched no-head control')
    ap.add_argument('--reds-bank', type=Path, help='Include fixed 16-frame held-out REDS validation')
    ap.add_argument('--joint', action='store_true', help='Optimize the fused SR backbone along with the refiner head')
    ap.add_argument('--no-history', action='store_true', help='Pinned recipe parity; the refiner is single-frame either way')
    ap.add_argument('--dino-gan-weight', type=float, default=0.0075)
    ap.add_argument('--reconstruction-edge', type=float, default=0.2,
        help='Sobel edge-term weight in the reconstruction objective (default preserves the control)')
    ap.add_argument('--reconstruction-fft', type=float, default=0.05,
        help='FFT spectral-term weight in the reconstruction objective (default preserves the control)')
    ap.add_argument('--pixrestore-repository', type=Path)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.frames < 3 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch, frames >=3 and even crop >=32 required')
    if args.head_channels < 0:
        ap.error('nonnegative refiner head channels required')
    if not args.no_history:
        ap.error('pinned recipe requires --no-history')
    if not math.isfinite(args.dino_gan_weight) or args.dino_gan_weight < 0:
        ap.error('nonnegative finite critic weight required')
    for name, value in (('reconstruction edge', args.reconstruction_edge),
                        ('reconstruction fft', args.reconstruction_fft)):
        if not math.isfinite(value) or value < 0:
            ap.error(f'nonnegative finite {name} weight required')
    if args.dino_gan_weight and not all((args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)):
        ap.error('paired critic requires pinned source repositories and weights')
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
    model = OutputRefinerSPAN(base, train_backbone=args.joint,
                              head_channels=args.head_channels).cuda().train()
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False)
    frozen_hash = state_digest(model.sr.state_dict())
    groups = []
    if args.head_channels:
        groups.append({'params': list(model.head_parameters()), 'lr': .0002, 'base_lr': .0002})
    if args.joint:
        groups.append({'params': model.sr.parameters(), 'lr': .00002, 'base_lr': .00002})
    if not groups:
        ap.error('no-head control requires --joint to train anything')
    optimizer = torch.optim.AdamW(groups, lr=.0002, weight_decay=0)
    adversary = None
    if args.dino_gan_weight:
        from paired_dino_adversary import PairedDinoAdversary
        adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    files = ['Tools/architectures/output_refiner.py',
             'Tools/experiments/train_output_refiner.py',
             'Tools/architectures/subspace_adapter.py',
             'Tools/experiments/recurrent_frame_feeding.py',
             'Tools/experiments/native_stages.py',
             'Tools/experiments/train_causal_detail.py',
             'Tools/experiments/mmap_training_bank.py',
             'Tools/experiments/build_causal_bank.py',
             'Tools/experiments/train_presented_detail.py',
             'Tools/experiments/train_ssm_span.py',
             'Tools/train_span.py',
             'Tools/frontier_eval/evaluate_sequences.py', 'Tools/eval_checkpoint.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'source_hashes': {p: digest(source_root / p) for p in files},
        'initial_head_sha256': state_digest(dict(model.head.state_dict())) if args.head_channels else None,
        'head_parameters': sum(p.numel() for p in model.head_parameters()),
        'recipe': 'final-frame HR reconstruction; FP32 fused SR; zero-init output-resolution residual',
        'backbone_trainable': args.joint, 'use_head': bool(args.head_channels),
        'optimizer': 'AdamW/cosine; head 2e-4, joint backbone 2e-5; no weight decay',
        'gradient_clipping': 'per optimizer group norm1',
        'source_motion_policy': args.source_motion_policy,
        'motion': 'decoded-LR native integer search plus half-pixel refinement; detached correspondence',
        'limitations': 'crop-local head diagnostic; no native export/cost parity; no browser timing or release admission',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    if args.reds_bank:
        experiment['reds_bank_sha256'] = digest(args.reds_bank / 'manifest.json')
    if adversary is not None:
        experiment['adversary'] = adversary.metadata
        experiment['discriminator_initial_sha256'] = state_digest(adversary.discriminator.state_dict())
        experiment['recipe'] += '; paired-DINO adversarial term, BF16 critic, raw decoded condition'
        for name in ('paired_dino_adversary.py', 'dino_adversary.py', 'dino_supervision.py'):
            path = 'Tools/experiments/' + name
            experiment['source_hashes'][path] = digest(source_root / path)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, reference = batch(data['train'], rng, args.batch, args.frames, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': reference})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        source, reference = source.cuda(), reference[:, -1].cuda()
        optimizer.zero_grad(set_to_none=True)
        output, inputs = run_head_sequence(model, source, use_head=True,
            source_motion_policy=args.source_motion_policy, return_inputs=True)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            experiment['first_input_sha256'] = state_digest({'inputs': inputs})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        a, b = output[:, -1, :, 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
        loss, fake_features, loss_metrics = training_objective(a, b, source[:, -1],
            adversary=adversary, gan_weight=args.dino_gan_weight,
            edge_weight=args.reconstruction_edge, fft_weight=args.reconstruction_fft)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite refiner objective')
        loss.backward()
        clip_optimizer_groups(optimizer)
        optimizer.step()
        if adversary is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss_metrics['discriminator'] = adversary.update(b, fake_features)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()), **loss_metrics,
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    backbone_unchanged = state_digest(model.sr.state_dict()) == frozen_hash
    if not args.joint and not backbone_unchanged:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'output_refiner2x', 'scale': 2,
                'head_channels': args.head_channels,
                'source_motion_policy': args.source_motion_policy,
                'channels': model.sr.core.conv_1.out_channels, 'version': model.sr.version,
                'step': args.steps, 'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, validation_data, validation_sources,
                      use_head=True, baseline=initial,
                      source_motion_policy=args.source_motion_policy)
    result['checkpoint_sha256'] = digest(args.out / f'step{args.steps:06d}.pth')
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': backbone_unchanged,
        'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
