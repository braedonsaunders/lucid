#!/usr/bin/env python3
"""Three-arm chroma-first reallocation test: 32ch RGB vs 40ch RGB vs inject.

Arm A (control_32): ordinary single-frame Unshuffled(32) backbone, wrapped in
  the same FullLR wrapper/training code as the rest. Bit-for-bit reproduces
  the pinned recipe control (same seed, same recipe as train_luma_arms.py).
Arm B (width_40): Unshuffled widened 32->40 function-preserving from the
  shared init. Isolates WIDTH at matched total MACs: any B-vs-A gap is the
  extra channels alone (plain extra capacity has failed here before:
  widen_span 32->48 gave no win).
Arm C (chroma40): ChromaInject(40) warm-started from the widened init: wide
  luma trunk + small learned chroma branch with cross-component luma-feature
  injection (CCALF-style). The DECISIVE comparison is C vs B: same width,
  same start, same MACs, and the only difference is the reallocation
  (capacity moved from the 0.10-error luma plane to the 0.30-0.38-error
  chroma planes).

All arms run the pinned recipe identically: --no-history current-only FullLR
wrapper, --joint, --state-channels 8, --source-motion-policy raw, --frames 3,
--batch 4, --crop 96, --dino-gan-weight 0.0075, seed 20260914, steps 2000.

Validation reports LPIPS/DISTS/PSNR_Y/detail_energy/fine_correlation per
source and source-balanced for all arms, PLUS per-plane error (Y/Cb/Cr, each
normalized by its own reference energy, as in the root measurement) for each
arm -- the metric this idea is about. Arm C should close the chroma gap.

TORCH-ONLY. Torch-level gains here have twice failed to survive into the
deployed native pipeline (r23b, r20). No result from this script is a
deployable-gain claim; transfer is unconfirmed by construction.
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
from architectures.full_lr_feature_span import FullLRFeatureSPAN
from full_lr_frame_feeding import run_full_lr_sequence
from train_ssm_span import select_validation, clip_optimizer_groups, training_objective
from eval_checkpoint import load
from train_span import Unshuffled
from widen_span import widen
from chroma_inject import (ChromaInject, FullLRChromaInjectSPAN,
                           warm_start_chroma_inject, report_macs, RGB2YCC)
from chroma_revive import ChromaReviveSPAN, FullLRChromaReviveSPAN
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def build_backbone(arm, init_path, luma_channels, rgb_channels, seed,
                   branch_channels=32, branch_depth=3):
    base, _, frames = load(init_path, 'cpu')
    if type(base) is not Unshuffled or frames != 1:
        raise ValueError('ordinary single-frame 2x SPAN initialization required')
    if base.core.conv_1.eval_conv.out_channels != 32:
        raise ValueError('32-channel init checkpoint required')
    if arm == 'control_32':
        return base, {'backbone': 'rgb32', 'channels': 32}
    if arm == 'chroma_revive':
        # Wraps the UNMODIFIED 32ch checkpoint directly (r97, see
        # chroma_revive.py) -- no widening, no warm start, bit-exact at
        # construction. This is a cleaner ablation against control_32 than
        # chroma40's own comparison against width_40: chroma_revive and
        # control_32 share literally the same pretrained weights, so any
        # arm gap is exactly the new branch's effect, with zero other
        # confounds (not even a different init channel count).
        revive = ChromaReviveSPAN(base, branch_channels=branch_channels, branch_depth=branch_depth)
        return revive, {'backbone': 'chroma_revive', 'channels': 32,
                        'branch_channels': branch_channels, 'branch_depth': branch_depth}
    wide = widen(base, 40, seed=seed)
    if arm == 'width_40':
        return wide, {'backbone': 'rgb40', 'channels': 40}
    if arm == 'chroma40':
        inject = ChromaInject(luma_channels, scale=2, version=wide.version)
        warm_start_chroma_inject(inject, wide)
        return inject, {'backbone': 'chroma40', 'luma_channels': luma_channels}
    raise ValueError('unknown arm')


def per_plane_error(output, reference):
    """YCbCr per-plane normalized reconstruction error (root measurement).

    Each plane's MSE divided by that plane's own reference energy, on 0-1
    tensors: err_p = mean((out_p - ref_p)^2) / mean(ref_p^2). Matches the
    root-measured diagnostic (Y 0.1006 / Cb 0.2968 / Cr 0.3763). Accepts a
    (B,T,3,H,W) sequence pair (validated over all frames) or a (B,3,H,W)
    single pair.
    """
    if output.shape != reference.shape or output.shape[1:3] != reference.shape[1:3]:
        raise ValueError('matched RGB pair required')
    if output.ndim == 5:
        if output.shape[2] != 3:
            raise ValueError('matched BTCHW RGB sequence pair required')
        output = output.reshape(-1, *output.shape[2:])
        reference = reference.reshape(-1, *reference.shape[2:])
    if output.ndim != 4 or output.shape[1] != 3:
        raise ValueError('matched BCHW RGB pair required')
    from luma_asym import YCC_BIAS
    mat = RGB2YCC.to(output.device).float()
    bias = YCC_BIAS.to(output.device).float().view(1, 3, 1, 1)
    # Identical to the backbone front-end: matmul then +[0, 0.5, 0.5].
    ycc_out = torch.einsum('ij,bjhw->bihw', mat, output) + bias
    ycc_ref = torch.einsum('ij,bjhw->bihw', mat, reference) + bias
    errors = {}
    for name, plane_out, plane_ref in zip(
            ('Y', 'Cb', 'Cr'), ycc_out.unbind(1), ycc_ref.unbind(1)):
        # AC energy: a near-black frame has ~zero reference energy, so
        # normalizing by raw mean-square explodes. Centering each plane on
        # its own mean keeps the ratio meaningful on any frame without
        # changing textured values.
        num = plane_out.sub(plane_ref).square().mean()
        den = plane_ref.sub(plane_ref.mean()).square().mean().clamp_min(1e-6)
        errors[name] = float(num / den)
    return errors


@torch.inference_mode()
def validate(model, data, source_ids, *, source_motion_policy='raw', state_warp='full_lr', baseline=None):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    planes = []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        output, inputs = run_full_lr_sequence(model, source, use_history=False,
            source_motion_policy=source_motion_policy, state_warp=state_warp, return_inputs=True)
        base_out = torch.stack([model.sr(inputs[:, t]).clamp(0, 1) for t in range(source.shape[1])], 1)
        variants = [('sr_backbone', base_out), ('arm_output', output)]
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
            plane = per_plane_error(values[..., 8:-8, 8:-8],
                                    reference.expand_as(values)[..., 8:-8, 8:-8])
            plane['id'] = identity
            plane['variant'] = variant
            plane['chroma_luma_ratio'] = ((plane['Cb'] + plane['Cr']) / 2
                                          / max(plane['Y'], 1e-12))
            planes.append(plane)
    return {'complete': True, 'scope': 'source-disjoint full-bank-sequence patch diagnostic; not native playback',
            'use_history': False, 'source_motion_policy': source_motion_policy, 'state_warp': state_warp,
            'rows': rows, 'summary': summarize(rows), 'temporal': temporal,
            'per_plane': planes,
            'per_plane_balanced': {k: float(np.mean([p[k] for p in planes
                                                     if p['variant'] == 'arm_output']))
                                   for k in ('Y', 'Cb', 'Cr', 'chroma_luma_ratio')}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--arm', choices=('control_32', 'width_40', 'chroma40', 'chroma_revive'), required=True)
    ap.add_argument('--branch-channels', type=int, default=32)
    ap.add_argument('--branch-depth', type=int, default=3)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--source-motion-policy', choices=('raw', 'taa', 'search'), default='raw')
    ap.add_argument('--state-warp', choices=('full_lr', 'packed_half_lr'), default='full_lr')
    ap.add_argument('--state-channels', type=int, default=8)
    ap.add_argument('--backbone-channels', type=int, default=40)
    ap.add_argument('--luma-channels', type=int, default=40)
    ap.add_argument('--initial-decay-bias', type=float, default=0.0)
    ap.add_argument('--reds-bank', type=Path)
    ap.add_argument('--joint', action='store_true')
    ap.add_argument('--dino-gan-weight', type=float, default=0)
    ap.add_argument('--reconstruction-edge', type=float, default=0.2)
    ap.add_argument('--reconstruction-fft', type=float, default=0.05)
    ap.add_argument('--pixrestore-repository', type=Path)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.frames < 3 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch, frames >=3 and even crop >=32 required')
    if args.state_channels < 3:
        ap.error('at least three full-LR state channels required')
    if not math.isfinite(args.initial_decay_bias):
        ap.error('finite initial decay bias required')
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
    backbone, tag = build_backbone(args.arm, args.init, args.luma_channels, args.backbone_channels, args.seed,
                                   branch_channels=args.branch_channels, branch_depth=args.branch_depth)
    if args.arm == 'chroma_revive':
        wrapper = FullLRChromaReviveSPAN
    elif args.arm == 'chroma40':
        wrapper = FullLRChromaInjectSPAN
    else:
        wrapper = FullLRFeatureSPAN
    model = wrapper(backbone, train_backbone=args.joint, state_channels=args.state_channels,
                    initial_decay_bias=args.initial_decay_bias).cuda().train()
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False)
    frozen_hash = state_digest(model.sr.state_dict())
    groups = [{'params': list(model.branch_parameters()), 'lr': .0002, 'base_lr': .0002}]
    if args.joint:
        # chroma_revive's branch_parameters() already includes the new
        # chroma-revival branch (see FullLRChromaReviveSPAN's docstring for
        # why it gets the fast branch rate rather than the slow joint rate);
        # the joint group here must therefore be scoped to the PRETRAINED
        # trunk only (backbone_parameters()), or its params would appear in
        # both optimizer groups at once (AdamW rejects that outright).
        joint_params = (list(model.backbone_parameters()) if hasattr(model, 'backbone_parameters')
                        else list(model.sr.parameters()))
        groups.append({'params': joint_params, 'lr': .00002, 'base_lr': .00002})
    optimizer = torch.optim.AdamW(groups, lr=.0002, weight_decay=0)
    adversary = None
    if args.dino_gan_weight:
        from paired_dino_adversary import PairedDinoAdversary
        adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    files = ['Tools/architectures/full_lr_feature_span.py',
             'Tools/experiments/chroma_inject.py',
             'Tools/experiments/chroma_revive.py',
             'Tools/experiments/full_lr_frame_feeding.py',
             'Tools/experiments/train_chroma_arms.py',
             'Tools/experiments/widen_span.py',
             'Tools/experiments/recurrent_frame_feeding.py',
             'Tools/experiments/native_stages.py',
             'Tools/experiments/train_causal_detail.py',
             'Tools/experiments/train_presented_detail.py',
             'Tools/experiments/train_ssm_span.py',
             'Tools/train_span.py',
             'Tools/frontier_eval/evaluate_sequences.py', 'Tools/eval_checkpoint.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'arm': args.arm, 'backbone_tag': tag,
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'source_hashes': {p: digest(source_root / p) for p in files},
        'initial_branch_sha256': state_digest({k: v for k, v in model.state_dict().items() if not k.startswith('sr.')}),
        'initial_shared_branch_sha256': state_digest({k: v for k, v in model.state_dict().items()
            if not k.startswith('sr.') and k != 'decay.bias'}),
        'branch_parameters': sum(p.numel() for p in model.branch_parameters()),
        'backbone_parameters': sum(p.numel() for p in (model.backbone_parameters()
                                                        if hasattr(model, 'backbone_parameters')
                                                        else model.sr.parameters())),
        'backbone_trainable': args.joint, 'use_history': False,
        'mac_report': (report_macs() if args.arm != 'chroma_revive' else
                       {'note': ('chroma_inject.report_macs() is a formula for the chroma40 '
                                'reallocation design and does not apply to chroma_revive; see '
                                '.build/quality-breakthrough-r97-chromarevival/report.json for the '
                                'real forward-hook-measured added params/MACs on the shipping checkpoint')}),
        'recipe': 'final-frame HR reconstruction; FP32 fused SR; backpropagation through decoded feature state',
        'widen_seed_note': 'width_40 and chroma40 both derive from the shared 32->40 function-preserving widening at the run seed; chroma_revive wraps the unmodified 32ch init directly (no widening)',
        'warm_start_note': ('chroma40 luma trunk verbatim + luma-projected first/upsampler; '
                            'chroma branch zero-init (starts as bicubic-chroma); '
                            'NOT function-preserving on color input; achromatic axis exact. '
                            'chroma_revive (r97) instead wraps the trained model directly and is '
                            'bit-exact with the control on color input too (verified in torch and '
                            'after Core ML conversion on real frames; see r97 README)'),
        'optimizer': 'AdamW/cosine; raw branch 2e-4, joint backbone 2e-5; no weight decay',
        'gradient_clipping': 'per optimizer group norm1',
        'source_motion_policy': args.source_motion_policy,
        'state_representation': 'full_lr_observation_features_v1',
        'feature_source': 'decoded_rgb8',
        'state_warp': args.state_warp, 'state_channels': args.state_channels,
        'motion': 'decoded-LR native integer search plus half-pixel refinement; detached correspondence',
        'limitations': ('TORCH-ONLY patch diagnostic; no native warp/cost parity; '
                        'no browser timing or release admission; torch gains here have twice '
                        'failed to transfer into the native pipeline (r23b, r20)'),
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
        output, inputs = run_full_lr_sequence(model, source, use_history=False,
            source_motion_policy=args.source_motion_policy, state_warp=args.state_warp, return_inputs=True)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            experiment['first_input_sha256'] = state_digest({'inputs': inputs})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        a, b = output[:, -1, :, 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
        loss, fake_features, loss_metrics = training_objective(a, b, source[:, -1],
            adversary=adversary, gan_weight=args.dino_gan_weight,
            edge_weight=args.reconstruction_edge, fft_weight=args.reconstruction_fft)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite chroma-arm objective')
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
    torch.save({'model': model.state_dict(), 'architecture': 'full_lr_feature_span2x', 'scale': 2,
                'state_representation': 'full_lr_observation_features_v1',
                'feature_source': 'decoded_rgb8',
                'state_warp': args.state_warp, 'state_channels': args.state_channels,
                'source_motion_policy': args.source_motion_policy,
                'version': model.sr.version,
                **tag,
                'backbone_tag': tag['backbone'],
                'channels': model.sr.core.conv_1.out_channels,
                'luma_note': 'chroma40 trunk input is single-channel luma; conv_1 out_channels is trunk width',
                'step': args.steps, 'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, validation_data, validation_sources, baseline=initial,
                      source_motion_policy=args.source_motion_policy, state_warp=args.state_warp)
    result['checkpoint_sha256'] = digest(args.out / f'step{args.steps:06d}.pth')
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': backbone_unchanged,
        'arm': args.arm, 'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
