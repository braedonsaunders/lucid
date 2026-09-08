#!/usr/bin/env python3
"""Joint degradation conditioning with matched SR-only and constant controls."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.degradation_conditioning import DegradationConditionedSPAN, oracle_degradation
from clean_lr_targets import differentiable_clean_lr
from eval_checkpoint import load
from native_stages import preprocess_sequence
from replay_recurrent_motion import validation_pairs
from train_causal_detail import batch, digest, load_bank
from train_presented_detail import reconstruction_objective, state_digest
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def condition_for(model, source, oracle):
    if not model.use_conditioning or model.mode == 'constant':
        return torch.full_like(source[:, :2], .25)
    return oracle if model.mode == 'oracle' else model.estimator(source)


def joint_objective(output, reference, condition, oracle, calibration_weight):
    if not math.isfinite(calibration_weight) or calibration_weight < 0:
        raise ValueError('finite nonnegative calibration weight required')
    a, b = output[..., 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
    calibration = F.l1_loss(condition[..., 4:-4, 4:-4], oracle[..., 4:-4, 4:-4])
    return reconstruction_objective(a, b, b, 'reference') + calibration_weight * calibration


@torch.inference_mode()
def validate(model, initial, pairs, sources, motion_policy):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, calibration, temporal = [], [], []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in pairs:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        inputs = preprocess_sequence(source, motion_policy=motion_policy)
        clean = differentiable_clean_lr(reference[0])[None]
        candidate, base, start, errors = [], [], [], []
        for t in range(inputs.shape[1]):
            x = inputs[:, t]
            oracle = oracle_degradation(x, clean[:, t])
            condition = condition_for(model, x, oracle)
            candidate.append(model.conditioned(x, condition).clamp(0, 1))
            base.append(model.sr(x).clamp(0, 1))
            start.append(initial(x).clamp(0, 1))
            errors.append(float((condition[..., 4:-4, 4:-4] - oracle[..., 4:-4, 4:-4]).abs().mean()))
        for variant, values in [('candidate', candidate), ('base', base), ('initial', start)]:
            values = torch.stack(values, 1)
            for t in (0, 8, 15):
                rows.append({'source_id': sources[identity], 'sequence_id': identity,
                             'variant': variant, 'frame': t,
                             'metrics': scorer.spatial(image(values[0, t, :, 8:-8, 8:-8]),
                                                       image(reference[0, t, :, 8:-8, 8:-8]))})
            error = values[..., 8:-8, 8:-8] - reference[..., 8:-8, 8:-8]
            temporal.append({'id': identity, 'variant': variant,
                             'residual_change_l1': float(torch.diff(error, dim=1).abs().mean()),
                             'mean_rgb_mse': float(error.square().mean()),
                             'last_rgb_mse': float(error[:, -1].square().mean())})
        calibration.append({'id': identity, 'source_id': sources[identity], 'condition_mae': float(np.mean(errors))})
    return {'complete': True, 'scope': '72 source-disjoint16-frame validation patches;216 scored frames/variant; source-stage proxy,not native playback',
            'source_motion_policy': motion_policy, 'conditioning_mode': model.mode,
            'use_conditioning': model.use_conditioning, 'rows': rows, 'summary': summarize(rows),
            'calibration': calibration, 'temporal': temporal,
            'oracle_status': 'Reference-only mechanism diagnostic; never a deployable inference mode or mathematical upper bound'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'reds-bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--mode', choices=('none', 'constant', 'estimated', 'oracle'), required=True)
    ap.add_argument('--source-motion-policy', choices=('taa', 'search'), default='search')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--estimator-channels', type=int, default=16)
    ap.add_argument('--condition-lr', type=float, default=.0002)
    ap.add_argument('--sr-lr', type=float, default=.00002)
    ap.add_argument('--calibration-weight', type=float, default=.5)
    ap.add_argument('--dino-gan-weight', type=float, default=.0075)
    for key in ('pixrestore-repository', 'dino-repository', 'dino-checkpoint'):
        ap.add_argument('--' + key, type=Path)
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch and even crop >=32 required')
    if (args.estimator_channels < 4 or any(not math.isfinite(v) or v <= 0 for v in (args.condition_lr, args.sr_lr))
            or any(not math.isfinite(v) or v < 0 for v in (args.calibration_weight, args.dino_gan_weight))):
        ap.error('valid channel count, learning rates and nonnegative loss weights required')
    if args.dino_gan_weight and any(getattr(args, k) is None for k in ('pixrestore_repository', 'dino_repository', 'dino_checkpoint')):
        ap.error('paired critic repositories and checkpoint required')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] != 16 or any(min(lr.shape[1:3]) < args.crop for lr, _, _ in data['train']):
        ap.error('16-frame bank and adequate crop geometry required')
    pairs, sources = validation_pairs(args.bank, manifest)
    reds, reds_sources = validation_pairs(args.reds_bank, manifest, reds=True)
    if sources.keys() & reds_sources.keys():
        raise ValueError('duplicate validation identities')
    pairs.extend(reds)
    sources.update(reds_sources)
    sr, _, frames = load(args.init, 'cuda')
    if type(sr) is not Unshuffled or frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
        ap.error('ordinary single-frame2x SPAN initialization required')
    mode = 'estimated' if args.mode == 'none' else args.mode
    model = DegradationConditionedSPAN(sr, mode, args.estimator_channels, train_backbone=True,
                                      use_conditioning=args.mode != 'none').cuda().train()
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False)
    backbone_hash = state_digest(model.sr.state_dict())
    groups = [{'params': model.sr.parameters(), 'lr': args.sr_lr, 'base_lr': args.sr_lr}]
    if model.use_conditioning:
        groups.append({'params': model.modulation.parameters(), 'lr': args.condition_lr, 'base_lr': args.condition_lr})
        if mode == 'estimated':
            groups.append({'params': model.estimator.parameters(), 'lr': args.condition_lr, 'base_lr': args.condition_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    adversary = None
    if args.dino_gan_weight:
        from paired_dino_adversary import PairedDinoAdversary
        adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    root = Path(__file__).resolve().parents[1]
    files = ['architectures/degradation_conditioning.py', 'architectures/subspace_adapter.py', 'eval_checkpoint.py',
             'experiments/train_joint_conditioning.py', 'experiments/train_causal_detail.py',
             'experiments/train_presented_detail.py', 'experiments/clean_lr_targets.py',
             'experiments/native_stages.py', 'experiments/paired_dino_adversary.py',
             'experiments/dino_adversary.py', 'experiments/dino_supervision.py',
             'experiments/replay_recurrent_motion.py', 'frontier_eval/evaluate_sequences.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'bank_sha256': digest(args.bank / 'manifest.json'), 'reds_bank_sha256': digest(args.reds_bank / 'manifest.json'),
                  'checkpoint_sha256': digest(args.init), 'initial_backbone_sha256': backbone_hash,
                  'initial_estimator_sha256': state_digest(model.estimator.state_dict()),
                  'initial_modulation_sha256': state_digest(model.modulation.state_dict()),
                  'conditioning_mode': model.mode, 'use_conditioning': model.use_conditioning,
                  'source_motion_policy': args.source_motion_policy,
                  'source_sha256': {f: digest(root / f) for f in files},
                  'condition_parameters': sum(p.numel() for module in (model.modulation, model.estimator) for p in module.parameters()),
                  'clean_target': 'FP32 RGB8 Lanczos3 half-pixel2x downsample preserving full chroma; derived from HR reference,not older subsampled clean-cache',
                  'recipe': 'Three-frame native input proxy, final-frame HR reconstruction plus estimated-condition calibration and paired critic; condition detached from SR loss; fusedSR/conditioning FP32,critic BF16; AdamW cosine no weight decay; each optimizer group clipped independently at norm1 so calibration cannot rescale SR gradients',
                  'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    if adversary:
        experiment['discriminator_initial_sha256'] = state_digest(adversary.discriminator.state_dict())
        experiment['adversary'] = adversary.metadata
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, reference = batch(data['train'], rng, args.batch, 3, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': reference})
        source, reference = source.cuda(), reference[:, -1].cuda()
        with torch.no_grad():
            inputs = preprocess_sequence(source, motion_policy=args.source_motion_policy)[:, -1]
            clean_target = differentiable_clean_lr(reference)
            oracle = oracle_degradation(inputs, clean_target)
        optimizer.zero_grad(set_to_none=True)
        condition = condition_for(model, inputs, oracle)
        output = model.conditioned(inputs, condition.detach()).clamp(0, 1)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            experiment['first_input_sha256'] = state_digest({'inputs': inputs})
            experiment['first_oracle_target_sha256'] = state_digest({'oracle': oracle})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        loss = joint_objective(output, reference, condition, oracle,
                               args.calibration_weight if args.mode == 'estimated' else 0.)
        if adversary:
            a, b = output[..., 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
            critic_condition = F.interpolate(source[:, -1], scale_factor=2, mode='bicubic',
                                            align_corners=False, antialias=True).clamp(0, 1)[..., 8:-8, 8:-8]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gan, fake_features = adversary.generator_loss(a, critic_condition)
            loss = loss + args.dino_gan_weight * gan
        if not torch.isfinite(loss):
            raise ValueError('nonfinite joint conditioning loss')
        loss.backward()
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1)
        optimizer.step()
        if adversary:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                adversary.update(b, fake_features)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    checkpoint = args.out / f'step{args.steps:06d}.pth'
    torch.save({'model': model.state_dict(), 'architecture': 'degradation_conditioned_span2x',
                'scale': 2, 'frames': 1, 'channels': model.core.conv_1.out_channels, 'version': model.version,
                'conditioning_mode': mode, 'use_conditioning': model.use_conditioning,
                'estimator_channels': args.estimator_channels, 'source_motion_policy': args.source_motion_policy,
                'step': args.steps, 'experiment': experiment}, checkpoint)
    report = validate(model, initial, pairs, sources, args.source_motion_policy)
    report['checkpoint_sha256'] = digest(checkpoint)
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'steps': args.steps,
        'backbone_changed': state_digest(model.sr.state_dict()) != backbone_hash,
        'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
