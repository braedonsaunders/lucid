#!/usr/bin/env python3
"""Train previous SR or decoded-observation inputs with a matched SR backbone."""
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
from native_stages import preprocess_sequence
from recurrent_frame_feeding import recurrent_step
from architectures.recurrent_span import RecurrentSPAN
from eval_checkpoint import load
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def run_sequence(model, decoded, native_inputs=False, *, use_history=True, source_motion_policy='taa'):
    if source_motion_policy not in ('taa', 'search'):
        raise ValueError('source motion policy must be taa or search')
    if native_inputs:
        with torch.no_grad():
            inputs = preprocess_sequence(decoded, motion_policy=source_motion_policy)
    else:
        inputs = decoded
    state, outputs = None, []
    for t in range(decoded.shape[1]):
        output, state, _ = recurrent_step(model, inputs[:, t], decoded[:, t], state,
                                          stream='sequence', index=t, use_history=use_history)
        outputs.append(output)
    return torch.stack(outputs, 1), inputs


@torch.inference_mode()
def validate(model, data, source_ids, native_inputs, *, use_history=True, baseline=None,
             source_motion_policy='taa'):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        output, inputs = run_sequence(model, source, native_inputs, use_history=use_history,
                                      source_motion_policy=source_motion_policy)
        base = torch.stack([model.sr(inputs[:, t]).clamp(0, 1) for t in range(inputs.shape[1])], 1)
        variants = [('base', base), ('recurrent', output)]
        if baseline is not None:
            variants.append(('initial', torch.stack([baseline(inputs[:, t]).clamp(0, 1)
                             for t in range(inputs.shape[1])], 1)))
        for variant, values in variants:
            # Longer validation history than the training unroll exposes drift.
            for t in sorted({0, inputs.shape[1] // 2, inputs.shape[1] - 1}):
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
            'use_history': use_history, 'base_definition': 'current checkpoint without history; initial is the unchanged starting model when present',
            'history_source': model.history_source,
            'native_input_stages': native_inputs,
            'source_motion_policy': source_motion_policy,
            'motion_seed': model.motion_seed,
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
    ap.add_argument('--native-input-stages', action='store_true')
    ap.add_argument('--source-motion-policy', choices=('taa', 'search'), default='taa',
                    help='Source TAA policy; legacy default preserves archived recipes, current app uses search')
    ap.add_argument('--reds-bank', type=Path, help='Include the fixed held-out REDS selection in final validation')
    ap.add_argument('--separate-gradient-clipping', action='store_true',
                    help='Clip each optimizer group independently so history gradients cannot rescale SR gradients')
    ap.add_argument('--joint', action='store_true', help='Optimize the fused SR backbone along with the history convolution')
    ap.add_argument('--no-history', action='store_true', help='Matched joint-training control; history remains zero and disabled')
    ap.add_argument('--history-source', choices=('sr', 'decoded'), default='sr',
                    help='Previous generated SR or bicubic-lifted decoded observation; decoded branch retains one observed frame')
    ap.add_argument('--motion-seed', choices=('taa', 'search'), default='search',
                    help='Preserve integer search for SR refinement; taa reproduces historical stationary overrides')
    ap.add_argument('--dino-gan-weight', type=float, default=0)
    ap.add_argument('--pixrestore-repository', type=Path)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.frames < 3 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch, frames >=3 and even crop >=32 required')
    if args.no_history and not args.joint:
        ap.error('no-history control requires a trainable backbone')
    if not math.isfinite(args.dino_gan_weight) or args.dino_gan_weight < 0:
        ap.error('nonnegative finite critic weight required')
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
    validation_data = data['validation']
    validation_sources = {r['id']: r['source_id'] for r in manifest['sequences']}
    if args.reds_bank:
        from replay_recurrent_motion import validation_pairs
        validation_data, validation_sources = validation_pairs(args.bank, manifest)
        reds_data, reds_sources = validation_pairs(args.reds_bank, manifest, reds=True)
        if validation_sources.keys() & reds_sources.keys():
            raise ValueError('duplicate validation identities')
        validation_data.extend(reds_data)
        validation_sources.update(reds_sources)
    base, _, frames = load(args.init, 'cuda')
    if type(base) is not Unshuffled or frames != 1:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = RecurrentSPAN(base, train_backbone=args.joint,
                          history_source=args.history_source, motion_seed=args.motion_seed).cuda().train()
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False) if args.joint else None
    if args.no_history:
        model.history.requires_grad_(False)
    frozen_hash = state_digest(model.sr.state_dict())
    groups = []
    if not args.no_history:
        groups.append({'params': model.history.parameters(), 'lr': .0002, 'base_lr': .0002})
    if args.joint:
        groups.append({'params': model.sr.parameters(), 'lr': .00002, 'base_lr': .00002})
    optimizer = torch.optim.AdamW(groups, lr=.0002, weight_decay=0)
    adversary = None
    if args.dino_gan_weight:
        from paired_dino_adversary import PairedDinoAdversary
        adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    files = ['architectures/recurrent_span.py', 'architectures/subspace_adapter.py',
             'experiments/recurrent_frame_feeding.py', 'experiments/native_stages.py',
             'experiments/train_presented_detail.py', 'experiments/train_recurrent_span.py',
             'experiments/train_causal_detail.py', 'experiments/replay_recurrent_motion.py',
             'frontier_eval/evaluate_sequences.py', 'eval_checkpoint.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'source_hashes': {p: digest(source_root / p) for p in files},
        'history_parameters': sum(p.numel() for p in model.history.parameters()),
        'recipe': 'final-frame HR reconstruction; FP32 fused SR; ' + (
            'backpropagation through quantized recurrent outputs' if args.history_source == 'sr'
            else 'previous decoded observation only; no generated-history backpropagation'),
        'backbone_trainable': args.joint, 'use_history': not args.no_history,
        'optimizer': 'AdamW/cosine; history 2e-4, joint backbone 2e-5; no weight decay',
        'gradient_clipping': 'per optimizer group norm1' if args.separate_gradient_clipping else 'joint norm1',
        'source_motion_policy': args.source_motion_policy,
        'motion': 'decoded-LR native integer search plus half-pixel refinement around integer and zero seeds; detached correspondence',
        'motion_seed': args.motion_seed,
        'history_storage': 'RGB8 straight-through quantization before output detail stages',
        'history_input': 'previous RGB8 SR output' if args.history_source == 'sr' else
                         'previous decoded RGB8 lifted to 2x by FP32 bicubic, align_corners=False, clamped; same motion and history convolution',
        'limitations': 'crop-local history; no native SR warp/cost parity; no browser timing or release admission',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    if args.reds_bank:
        experiment['reds_bank_sha256'] = digest(args.reds_bank / 'manifest.json')
    if adversary:
        experiment['adversary'] = adversary.metadata
        experiment['discriminator_initial_sha256'] = state_digest(adversary.discriminator.state_dict())
        for name in ('dino_adversary.py', 'dino_supervision.py', 'paired_dino_adversary.py'):
            experiment['source_hashes'][name] = digest(Path(__file__).with_name(name))
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, reference = batch(data['train'], rng, args.batch, args.frames, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': reference})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        source, reference = source.cuda(), reference[:, -1].cuda()
        optimizer.zero_grad(set_to_none=True)
        # Keep warping/quantization FP32. Only the neural graph needs autocast;
        # this initial causal probe deliberately uses FP32 throughout.
        output, inputs = run_sequence(model, source, args.native_input_stages, use_history=not args.no_history,
                                      source_motion_policy=args.source_motion_policy)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            experiment['first_input_sha256'] = state_digest({'inputs': inputs})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        a, b = output[:, -1, :, 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
        loss = reconstruction_objective(a, b, b, 'reference')
        if adversary:
            condition = F.interpolate(source[:, -1], scale_factor=2, mode='bicubic',
                                      align_corners=False, antialias=True).clamp(0, 1)[..., 8:-8, 8:-8]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gan_loss, fake_features = adversary.generator_loss(a, condition)
            loss = loss + args.dino_gan_weight * gan_loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite recurrent objective')
        loss.backward()
        if args.separate_gradient_clipping:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], 1)
        else:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1)
        optimizer.step()
        if adversary:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                adversary.update(b, fake_features)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    backbone_unchanged = state_digest(model.sr.state_dict()) == frozen_hash
    if not args.joint and not backbone_unchanged:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'recurrent_span2x', 'scale': 2,
                'history_source': args.history_source,
                'motion_seed': args.motion_seed,
                'channels': model.sr.core.conv_1.out_channels, 'version': model.sr.version,
                'step': args.steps, 'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, validation_data, validation_sources, args.native_input_stages,
                      use_history=not args.no_history, baseline=initial,
                      source_motion_policy=args.source_motion_policy)
    result['checkpoint_sha256'] = digest(args.out / f'step{args.steps:06d}.pth')
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': backbone_unchanged,
        'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
