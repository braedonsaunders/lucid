#!/usr/bin/env python3
"""Distill a fixed shipping/PixRestore mixture into a shipping-initialized 2x model.

Full-context shipping targets are cached before random training crops. Both arms
share data/RNG/objectives; only the fixed teacher mixture differs. No flicker loss.
"""
import argparse
import hashlib
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
from teacher_distillation import load_teacher_cache
from fold_shipping_head import fold_head
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from reconstruction_loss import sobel_loss
from train_span import fft_loss


def reconstruction_objective(output, reference, intended, detail_target='mixture', fidelity=None):
    if detail_target not in ('mixture', 'reference'):
        raise ValueError('unknown detail supervision target')
    detail = reference if detail_target == 'reference' else intended
    if fidelity is not None:
        if detail_target != 'reference' or not torch.equal(reference, intended):
            raise ValueError('AESOP replacement requires reference-only supervision')
        pixel = 1.1 * fidelity(output, reference)
    else:
        # Preserve the control's original floating-point operation order.
        return (F.l1_loss(output, intended) + .2 * sobel_loss(output, detail)
                + .05 * fft_loss(output, detail) + .1 * F.l1_loss(output, reference))
    return pixel + .2 * sobel_loss(output, detail) + .05 * fft_loss(output, detail)


def temporal_consistency(previous_output, output, previous_reference, reference, threshold=2 / 255):
    """Penalize output change where the reference barely moved (masked L1).

    Two consecutive frames of the same scene under two draws of codec noise are the
    same signal; forbidding the output to follow that difference removes invented
    grain that flickers and, measured 2026-09-04, improves per-frame perception too.
    No inference cost: the shipping model stays single-frame.
    """
    motion = F.avg_pool2d((reference - previous_reference).abs().mean(1, keepdim=True), 3, 1, 1)
    still = (motion < threshold).float()
    coverage = still.mean().clamp_min(1e-3)
    return ((output - previous_output).abs() * still).mean() / coverage, float(still.mean())


def local_variance(x, size=7):
    """Per-pixel variance of a residual over a size x size window, per channel."""
    mean = F.avg_pool2d(x, size, 1, size // 2, count_include_pad=False)
    square = F.avg_pool2d(x * x, size, 1, size // 2, count_include_pad=False)
    return (square - mean * mean).clamp_min(0)


def ldl_artifact_map(output, reference, ema_output, size=7, exponent=.2, formulation='reference'):
    """LDL residual-variance heuristic, not a guarantee of artifact detection.

    The author's formulation uses reflected, unbiased window variance and
    differentiates through the current residual map. ``legacy`` reproduces r27
    (detached map, population variance, truncated border windows).
    """
    if size < 3 or size % 2 == 0 or min(output.shape[-2:]) <= size // 2:
        raise ValueError('odd window >=3 and images larger than its padding required')
    if formulation not in ('reference', 'legacy'):
        raise ValueError('unknown LDL formulation')
    residual = (reference - output).abs().sum(1, keepdim=True)
    residual_ema = (reference - ema_output).abs().sum(1, keepdim=True).detach()
    if formulation == 'legacy':
        residual = residual.detach()
    patch = residual.flatten(1).var(1).clamp_min(1e-12).pow(exponent).view(-1, 1, 1, 1)
    if formulation == 'reference':
        padded = F.pad(residual, [size // 2] * 4, mode='reflect')
        windows = padded.unfold(2, size, 1).unfold(3, size, 1)
        variance = windows.var(dim=(-1, -2), unbiased=True)
    else:
        variance = local_variance(residual, size)
    weight = patch * variance
    return torch.where(residual < residual_ema, torch.zeros_like(weight), weight)


def ldl_loss(output, reference, ema_output, formulation='reference'):
    weight = ldl_artifact_map(output, reference, ema_output, formulation=formulation)
    return (weight * (output - reference).abs()).mean()


@torch.no_grad()
def update_ema(ema_model, model, decay=.999):
    """Update by name, including buffers, and fail on a mismatched twin."""
    source, target = model.state_dict(), ema_model.state_dict()
    if source.keys() != target.keys():
        raise ValueError('EMA and student states differ')
    for name, value in target.items():
        current = source[name].to(value)
        if value.is_floating_point():
            value.lerp_(current, 1 - decay)
        else:
            value.copy_(current)


def bounded_adversarial_scale(base_norm, weighted_adversarial_norm, ratio_cap):
    """Only attenuate the GAN term; the detached scale adds no second derivatives."""
    if not math.isfinite(ratio_cap) or ratio_cap <= 0:
        raise ValueError('positive finite head-gradient ratio cap required')
    return (ratio_cap * base_norm.detach() / weighted_adversarial_norm.detach().clamp_min(1e-12)).clamp(0, 1)


def augment_input_noise(x, sigma, generator=None):
    """Add per-sample Gaussian noise of random strength in [0, sigma] to the LR input only.

    The reference target is unchanged, so the student learns not to amplify sensor
    grain and codec noise into speckle; grainy holdout scenes are where every
    candidate so far lost LPIPS.
    """
    if sigma <= 0:
        return x
    strength = torch.rand(x.shape[0], 1, 1, 1, device=x.device, generator=generator) * sigma
    return (x + torch.randn(x.shape, device=x.device, generator=generator) * strength).clamp(0, 1)


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@torch.inference_mode()
def shipping_targets(model, data, out, bank_hash, checkpoint_hash):
    manifest_path = out / 'manifest.json'
    if out.exists():
        targets, receipt = load_teacher_cache(out, bank_hash, data, digest)
        if receipt.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('cached shipping weights differ')
        return targets, receipt
    out.mkdir(parents=True)
    targets, rows = {}, []
    for lr, hr, identity in data['train']:
        outputs = []
        for pixels in lr:
            x = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].cuda().float() / 255
            y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(np.rint(y * 255).astype(np.uint8))
            image = image.resize((hr.shape[2], hr.shape[1]), Image.Resampling.BICUBIC)
            outputs.append(np.asarray(image).copy())
        target = np.stack(outputs)
        path = out / f'{identity}.npz'
        np.savez_compressed(path, teacher=target)
        targets[identity] = target
        rows.append({'id': identity, 'file': path.name, 'sha256': digest(path)})
        print('cache', len(rows), len(data['train']), flush=True)
    receipt = {'bank_sha256': bank_hash, 'checkpoint_sha256': checkpoint_hash,
        'code_sha256': digest(__file__), 'sequences': rows, 'split': 'train only',
        'inference': 'FP32 shipping 4x; clamp/round RGB8; PIL bicubic to 2x before training crop',
        'reference_used_for_prediction': False}
    manifest_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return targets, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--pixrestore-cache', type=Path, required=True)
    ap.add_argument('--shipping-cache', type=Path, required=True)
    ap.add_argument('--init', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--teacher-mix', type=float, required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--lr', type=float, default=0.00002)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--architecture', choices=['coupled', 'anchored_detail', 'anchored_lowpass',
                    'subspace_full', 'subspace_protected'], default='coupled')
    ap.add_argument('--detail-channels', type=int, default=32)
    ap.add_argument('--detail-blocks', type=int, default=4)
    ap.add_argument('--detail-target', choices=['mixture', 'reference'], default='mixture',
                    help='Controlled supervision ablation; inference architecture and weights format are unchanged')
    ap.add_argument('--dino-gan-weight', type=float, default=0)
    ap.add_argument('--ldl-weight', type=float, default=0,
        help='weight of the Details-or-Artifacts residual-variance penalty against an EMA twin (alpha 0.999)')
    ap.add_argument('--ldl-formulation', choices=['reference', 'legacy'], default='reference',
        help='reference = differentiable reflected sample variance; legacy reproduces ladder r27')
    ap.add_argument('--native-input-stages', action='store_true',
        help='Train on a three-frame proxy of native deband + TAA BEFORE the model')
    ap.add_argument('--native-output-stages', action='store_true',
        help='Apply the fixed native 2x sharpen/tone/grain proxy to predictions before losses')
    ap.add_argument('--aesop-checkpoint', type=Path,
        help='Pinned published AESOP autoencoder; replaces both pixel L1 terms')
    ap.add_argument('--clean-lr-weight', type=float, default=0,
        help='Clean full-chroma RGB Lanczos LR consistency against downsampled HR')
    ap.add_argument('--training-frames', type=int, choices=[1, 2, 3], default=1,
        help='Minimum sequence length; use 3 for the matched native-input control')
    ap.add_argument('--temporal', type=float, default=0,
                    help='Weight of the reference-static temporal consistency term (two consecutive frames per crop; 0 = off)')
    ap.add_argument('--input-noise', type=float, default=0,
                    help='Max Gaussian noise sigma (0-1 scale) added to LR inputs during training; targets unchanged')
    ap.add_argument('--intended', choices=['mixture', 'reference', 'teacher'], default='mixture',
                    help='Regression target: the fixed shipping/teacher mixture (control), the HR reference itself, or the teacher cache alone')
    ap.add_argument('--paired-negatives', choices=['shift', 'shift+smooth'], default='shift',
                    help='Paired-critic negative recipe; shift+smooth adds noise texture on smooth reference regions as a fake')
    ap.add_argument('--smooth-negative-share', type=float, default=.125,
                    help='Share of the 0.25 negative mass given to the smooth-texture fake when shift+smooth is used')
    ap.add_argument('--paired-dino', action='store_true',
                    help='Training-only input-conditioned critic with wrong-detail negatives')
    ap.add_argument('--gan-head-ratio-cap', type=float, default=0,
                    help='Optional per-step GAN/reconstruction gradient-norm cap at the output head; zero preserves control')
    ap.add_argument('--pixrestore-repository', type=Path)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    args = ap.parse_args()
    if args.out.exists() or args.teacher_mix not in (0, .5) or args.steps < 1 or args.batch < 1 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, fixed teacher mix 0/.5, positive steps/batch and even crop >=32 required')
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error('positive finite learning rate required')
    if not math.isfinite(args.input_noise) or args.input_noise < 0 or args.input_noise > .1:
        ap.error('input noise sigma must be within [0, 0.1]')
    if args.detail_channels < 4 or args.detail_blocks < 1:
        ap.error('detail channels >=4 and positive block count required')
    if not math.isfinite(args.dino_gan_weight) or args.dino_gan_weight < 0:
        ap.error('nonnegative finite adversarial weight required')
    if not math.isfinite(args.ldl_weight) or args.ldl_weight < 0:
        ap.error('nonnegative finite LDL weight required')
    if not math.isfinite(args.clean_lr_weight) or args.clean_lr_weight < 0:
        ap.error('nonnegative finite clean LR weight required')
    if args.aesop_checkpoint and (args.intended != 'reference' or args.detail_target != 'reference' or args.teacher_mix):
        ap.error('AESOP replacement requires reference intended/detail targets and no teacher mixture')
    if not math.isfinite(args.gan_head_ratio_cap) or args.gan_head_ratio_cap < 0 or (args.gan_head_ratio_cap and not args.dino_gan_weight):
        ap.error('nonnegative finite ratio cap requires an adversary when enabled')
    if args.dino_gan_weight and not all((args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)):
        ap.error('adversarial supervision requires pinned PixRestore and DINO sources/weights')
    if args.paired_dino and not args.dino_gan_weight:
        ap.error('paired critic requires a positive adversarial weight')
    if not torch.cuda.is_available():
        raise ValueError('this experiment requires the authorized CUDA worker')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    mapped = manifest.get('storage') == 'mmap-pairs-v1'
    if mapped and (args.intended != 'reference' or str(args.pixrestore_cache) != 'none' or str(args.shipping_cache) != 'none'):
        ap.error('mapped banks currently require reference supervision and no teacher caches')
    bank_hash = digest(args.bank / 'manifest.json')
    if args.intended == 'teacher' and str(args.pixrestore_cache) == 'none':
        ap.error('--intended teacher requires a teacher cache')
    if str(args.pixrestore_cache) == 'none':
        if args.intended != 'reference' or args.teacher_mix:
            ap.error('a bank without a teacher cache requires --intended reference and --teacher-mix 0')
        teacher, teacher_receipt = {}, {'note': 'no teacher cache; reference-target training'}
    else:
        teacher, teacher_receipt = load_teacher_cache(args.pixrestore_cache, bank_hash, data, digest)
    shipping, _, frames = load(args.init, 'cuda')
    if any(hasattr(shipping, name) for name in ('cleaner', 'estimator', 'variance_head')):
        ap.error('cleaner/conditioning/confidence checkpoints require their dedicated trainer')
    if frames != 1:
        raise ValueError('single-frame shipping initialization required')
    already_2x = shipping.core.upsampler[1].upscale_factor == 4
    if already_2x and args.intended != 'reference':
        ap.error('a 2x starting checkpoint has no 4x shipping targets; use --intended reference')
    if already_2x or (args.intended in ('reference', 'teacher') and str(args.shipping_cache) == 'none'):
        # Starting from an earlier 2x stage, or training to the reference on a bank
        # without cached shipping targets: the cache is not needed.
        target, shipping_receipt = {}, {'checkpoint_sha256': digest(args.init), 'split': 'unused'}
    else:
        target, shipping_receipt = shipping_targets(shipping, data, args.shipping_cache, bank_hash, digest(args.init))
    # Form the full-frame RGB8 mixture before augmentation, including both arms.
    mixed = {identity: np.rint(pixels.astype(np.float32) * (1-args.teacher_mix)
             + teacher[identity].astype(np.float32) * args.teacher_mix).astype(np.uint8)
             for identity, pixels in target.items()}
    if args.intended == 'teacher':
        # Distillation: regress to the teacher's full-frame outputs; the reference
        # terms (0.1 L1 and, with --detail-target reference, Sobel/FFT) stay.
        mixed = {identity: teacher[identity] for _, _, identity in data['train']}
    if args.intended == 'reference':
        # The fixed 50% shipping/PixRestore mixture scores only +2.9% LPIPS / +6.6% DISTS
        # over shipping on the development set, below what the paired-critic student
        # already reaches, so that target holds the student back; regress to truth.
        mixed = None if mapped else {identity: hr for _, hr, identity in data['train']}
    del teacher, target
    model = (shipping if already_2x else fold_head(shipping)).cuda().train()
    del shipping
    subspace = args.architecture in ('subspace_full', 'subspace_protected')
    model_channels = model.core.conv_1.eval_conv.out_channels
    if subspace:
        from architectures.subspace_adapter import attach_adapters, merge_adapters, adapter_report
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model = attach_adapters(model, constrained=args.architecture == 'subspace_protected', energy=.95).cuda().train()
    if args.architecture in ('anchored_detail', 'anchored_lowpass'):
        from architectures.anchored_detail import AnchoredDetail
        # Added branch initialization must not change the matched discriminator
        # RNG stream relative to the completed coupled reconstruction control.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model = AnchoredDetail(model, args.detail_channels, args.detail_blocks,
                                   residual_lowpass=args.architecture == 'anchored_lowpass').cuda().train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0)
    fidelity = None
    if args.aesop_checkpoint:
        from aesop_fidelity import load_aesop_loss
        fidelity = load_aesop_loss(args.aesop_checkpoint, 'cuda')
    if args.clean_lr_weight:
        from clean_lr_targets import differentiable_clean_lr, clean_lr_consistency
    rng = np.random.default_rng(args.seed)
    adversary = None
    if args.dino_gan_weight:
        from dino_adversary import DinoAdversary
        if args.paired_dino:
            from paired_dino_adversary import PairedDinoAdversary
            adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint,
                                            negatives=args.paired_negatives, smooth_share=args.smooth_negative_share)
        else:
            adversary = DinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    args.out.mkdir(parents=True)
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
        'teacher_manifest_sha256': digest(args.pixrestore_cache / 'manifest.json') if str(args.pixrestore_cache) != 'none' else None,
        'shipping_manifest_sha256': digest(args.shipping_cache / 'manifest.json') if str(args.shipping_cache) != 'none' and (args.shipping_cache / 'manifest.json').is_file() else None,
        'initialization': '2x checkpoint used directly' if already_2x else 'folded 4x shipping head',
        'source_hashes': {str(p.name): digest(p) for p in (Path(__file__),
            Path(__file__).with_name('fold_shipping_head.py'), Path(__file__).with_name('train_causal_detail.py'),
            Path(__file__).resolve().parents[1] / 'train_span.py',
            Path(__file__).resolve().parents[1] / 'architectures/span_arch.py')},
        'loss': 'L1 to ' + {'reference': 'HR reference', 'teacher': 'teacher cache', 'mixture': 'fixed mixture'}[args.intended]
                + f' + 0.2 signed Sobel to {args.detail_target} + 0.05 FFT to {args.detail_target} + 0.1 L1 to reference; 8 output-pixel border excluded',
        'precision': 'CUDA BF16 autocast; AdamW FP32; no compilation',
        'input_noise': args.input_noise,
        'temporal': args.temporal,
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
        'purpose': 'controlled quality/performance experiment; not shipping promotion'}
    if mapped:
        experiment['source_hashes']['mmap_training_bank.py'] = digest(Path(__file__).with_name('mmap_training_bank.py'))
        experiment['bank_storage'] = 'checked read-only memory maps; only sampled crops copied; HR target aliases sampled reference'
    if fidelity is not None:
        experiment['aesop'] = fidelity.metadata
        experiment['source_hashes']['aesop_fidelity.py'] = digest(Path(__file__).with_name('aesop_fidelity.py'))
        experiment['loss'] = '1.1 AESOP decoded-image L1 + 0.2 signed Sobel + 0.05 FFT to HR; 8 output-pixel border excluded'
    if args.clean_lr_weight:
        experiment['source_hashes']['clean_lr_targets.py'] = digest(Path(__file__).with_name('clean_lr_targets.py'))
        experiment['clean_lr'] = {
            'weight': args.clean_lr_weight, 'target': 'HR downsampled with the same full-chroma RGB Lanczos-3 operator as prediction',
            'order': 'before 8-pixel HR loss crop; after output-stage proxy when enabled',
            'boundary_exclusion_lr': 4, 'quantization': 'RGB8 straight-through on prediction; target detached',
            'scope': 'clean RGB supervision; not bit-exact bank pre-encode YUV; no decoded-input self-consistency',
            'cache': 'none; older cleaner FFmpeg cache is incompatible with full-chroma operator'}
    if args.architecture in ('anchored_detail', 'anchored_lowpass'):
        experiment['source_hashes']['anchored_detail.py'] = digest(
            Path(__file__).resolve().parents[1] / 'architectures/anchored_detail.py')
        experiment['residual_filter'] = 'sigma-1 radius-3 Gaussian, replicated boundaries' if args.architecture == 'anchored_lowpass' else 'none'
        experiment['fidelity_anchor'] = 'frozen folded shipping weights; only conditional detail branch is optimized'
    if subspace:
        experiment['source_hashes']['subspace_adapter.py'] = digest(
            Path(__file__).resolve().parents[1] / 'architectures/subspace_adapter.py')
        experiment['subspace'] = {'energy': .95, 'constrained': args.architecture == 'subspace_protected',
            'basis': 'SVD of frozen fused convolution weights plus bias; no reference images',
            'initial_layers': adapter_report(model),
            'limitation': 'orthogonal local weight changes do not guarantee nonlinear image fidelity',
            'inference': 'merge adapters into ordinary convolutions; no added layers'}
        def subspace_anchor_digest():
            return state_digest({k: v for k, v in model.state_dict().items()
                                 if k.endswith(('.anchor', '.basis', '.protected_basis'))})
        experiment['subspace_anchor_sha256'] = subspace_anchor_digest()
    if adversary:
        experiment['adversary'] = adversary.metadata
        experiment['discriminator_initial_sha256'] = state_digest(adversary.discriminator.state_dict())
        experiment['source_hashes']['dino_adversary.py'] = digest(Path(__file__).with_name('dino_adversary.py'))
        experiment['source_hashes']['dino_supervision.py'] = digest(Path(__file__).with_name('dino_supervision.py'))
        if args.paired_dino:
            experiment['source_hashes']['paired_dino_adversary.py'] = digest(Path(__file__).with_name('paired_dino_adversary.py'))
    (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    if args.native_input_stages:
        from native_stages import preprocess_sequence
        experiment['source_hashes']['native_stages.py'] = digest(Path(__file__).with_name('native_stages.py'))
        experiment['native_input_stages'] = {
            'order': 'decoded RGB -> quantized 420 deband -> source-resolution TAA -> RGB -> model',
            'frames': 3, 'feedback': .5, 'guard': .005,
            'limitations': 'crop boundaries; short reset history; approximate RGB/420 conversion; output sharpen/grade not emulated'}
        (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    if args.native_output_stages:
        from native_output_stages import postprocess_rgb
        for name in ('native_stages.py', 'native_output_stages.py'):
            experiment['source_hashes'][name] = digest(Path(__file__).with_name(name))
        experiment['native_output_stages'] = {
            'order': 'model -> RGB8/420 proxy -> sharpen .2 -> tone/contrast .2/adaptive grain .01 -> RGB8 proxy -> losses',
            'radius': 2, 'reference_radius': 2,
            'limitations': 'crop-local grain phase/statistics; approximate RGB/420 conversion and chroma siting; straight-through quantization'}
        (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    started = time.monotonic()
    anchor_hash = None
    max_applied_head_ratio = torch.zeros((), device='cuda')
    min_adversarial_scale = torch.ones((), device='cuda')
    temporal_coverage = []
    ema_model = None
    for step in range(1, args.steps + 1):
        frames = max(args.training_frames, 3 if args.native_input_stages else 2 if args.temporal else 1)
        sampled = batch(data['train'], rng, args.batch, frames, args.crop, mixed)
        x, reference, intended = (*sampled, sampled[1]) if mapped else sampled
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': x, 'reference': reference})
        if args.native_output_stages:
            decoded_current = x[:, -1].cuda()
            if args.temporal:
                decoded_previous = x[:, -2].cuda()
        if args.native_input_stages:
            # Native source stages do not depend on model weights. Cacheable in
            # principle; no gradient graph is needed for this input-only arm.
            with torch.no_grad():
                x = preprocess_sequence(x.cuda())
        if args.temporal:
            previous = tuple(v[:, -2].cuda() for v in (x, reference))
        x, reference, intended = (v[:, -1].cuda() for v in (x, reference, intended))
        if step == 1:
            experiment['first_batch_sha256'] = state_digest({'source': x, 'reference': reference, 'intended': intended})
        x = augment_input_noise(x, args.input_noise)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x)
        if step == 1:
            experiment['first_output_sha256'] = state_digest({'output': output})
            if args.architecture in ('anchored_detail', 'anchored_lowpass'):
                # The first forward materializes the frozen inference convolutions.
                anchor_hash = state_digest(model.anchor.state_dict())
                experiment['anchor_initial_sha256'] = anchor_hash
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        if args.native_output_stages:
            output = postprocess_rgb(output.float(), decoded_current)
        if args.clean_lr_weight:
            with torch.no_grad():
                clean_target = differentiable_clean_lr(reference)
            clean_loss = clean_lr_consistency(output, clean_target)
        output, reference, intended = (v.float()[:, :, 8:-8, 8:-8] for v in (output, reference, intended))
        loss = reconstruction_objective(output, reference, intended, args.detail_target, fidelity)
        if args.clean_lr_weight:
            loss = loss + args.clean_lr_weight * clean_loss
        if args.ldl_weight:
            if ema_model is None:
                import copy
                ema_model = copy.deepcopy(model).eval()
                for parameter in ema_model.parameters(): parameter.requires_grad_(False)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                ema_output = ema_model(x).float()
            if args.native_output_stages:
                with torch.no_grad():
                    ema_output = postprocess_rgb(ema_output, decoded_current)
            ema_output = ema_output[:, :, 8:-8, 8:-8]
            artifact = ldl_loss(output, reference, ema_output, args.ldl_formulation)
            if step in (1, 200, 1000):
                print(json.dumps({'step': step, 'reconstruction': float(loss), 'ldl_raw': float(artifact),
                    'ldl_weighted': float(args.ldl_weight * artifact)}), flush=True)
            loss = loss + args.ldl_weight * artifact
        if args.temporal:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                previous_output = model(augment_input_noise(previous[0], args.input_noise))
            if args.native_output_stages:
                previous_output = postprocess_rgb(previous_output.float(), decoded_previous)
            previous_output = previous_output.float()[:, :, 8:-8, 8:-8]
            consistency, covered = temporal_consistency(previous_output, output, previous[1].float()[:, :, 8:-8, 8:-8], reference)
            temporal_coverage.append(covered)
            loss = loss + args.temporal * consistency
        gan_loss, discriminator_loss = None, None
        if adversary:
            adversarial_scale = 1
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.paired_dino:
                    condition = F.interpolate(x.float(), scale_factor=2, mode='bicubic',
                        align_corners=False, antialias=True).clamp(0, 1)[:, :, 8:-8, 8:-8]
                    gan_loss, fake_features = adversary.generator_loss(output, condition)
                else:
                    gan_loss, fake_features = adversary.generator_loss(output)
            if args.gan_head_ratio_cap or step in (1, 200, 1000):
                head = (model.core.upsampler[0].coefficients if subspace else model.head.weight
                        if args.architecture in ('anchored_detail', 'anchored_lowpass') else model.core.upsampler[0].weight)
                base_gradient = torch.autograd.grad(loss, head, retain_graph=True)[0].float().norm()
                adversarial_gradient = torch.autograd.grad(args.dino_gan_weight * gan_loss, head, retain_graph=True)[0].float().norm()
                if args.gan_head_ratio_cap:
                    adversarial_scale = bounded_adversarial_scale(base_gradient, adversarial_gradient, args.gan_head_ratio_cap)
                    applied_ratio = adversarial_scale * adversarial_gradient / base_gradient.clamp_min(1e-12)
                    max_applied_head_ratio = torch.maximum(max_applied_head_ratio, applied_ratio.detach())
                    min_adversarial_scale = torch.minimum(min_adversarial_scale, adversarial_scale)
            if step in (1, 200, 1000) or (args.gan_head_ratio_cap and step % 200 == 0):
                print(json.dumps({'step': step, 'head_base_gradient_norm': float(base_gradient),
                    'head_adversarial_gradient_norm': float(adversarial_gradient),
                    'head_gradient_ratio': float(adversarial_gradient / base_gradient.clamp_min(1e-12)),
                    'adversarial_scale': float(adversarial_scale),
                    'applied_head_gradient_ratio': float(adversarial_scale * adversarial_gradient / base_gradient.clamp_min(1e-12))}), flush=True)
            loss = loss + (args.dino_gan_weight * gan_loss) * adversarial_scale
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if args.ldl_weight and ema_model is not None:
            update_ema(ema_model, model)
        if adversary:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                discriminator_loss = adversary.update(reference, fake_features)
        rate = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        if step % 200 == 0:
            extra = f' generator={float(gan_loss.detach()):.5f} discriminator={discriminator_loss:.5f}' if adversary else ''
            print(f'step {step}/{args.steps} loss={float(loss):.6f} minutes={(time.monotonic()-started)/60:.2f}{extra}', flush=True)
        if step % 2000 == 0 or step == args.steps:
            if args.architecture in ('anchored_detail', 'anchored_lowpass') and state_digest(model.anchor.state_dict()) != anchor_hash:
                raise ValueError('frozen reconstruction weights changed during detail training')
            anchor = model.anchor if args.architecture in ('anchored_detail', 'anchored_lowpass') else model
            state = model.state_dict()
            if subspace:
                if subspace_anchor_digest() != experiment['subspace_anchor_sha256']:
                    raise ValueError('frozen subspace anchor or coordinates changed')
                layers = adapter_report(model)
                if args.architecture == 'subspace_protected' and any(r['protected_update_norm'] > 1e-5 for r in layers):
                    raise ValueError('merged FP32 update escaped the protected subspace tolerance')
                (args.out / f'subspace{step:06d}.json').write_text(json.dumps(layers, indent=2) + '\n')
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    merged = merge_adapters(model)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if not torch.equal(model(x), merged(x)):
                        raise ValueError('merged training graph changes BF16 output')
                state = merged.state_dict()
            checkpoint = {'model': state, 'channels': model_channels,
                'scale': 2, 'frames': 1, 'version': anchor.version, 'step': step,
                'architecture': 'shipping_direct2x_area', 'experiment': experiment}
            if subspace:
                checkpoint['architecture'] = 'fused_span2x'
            if args.architecture in ('anchored_detail', 'anchored_lowpass'):
                checkpoint.update(architecture='anchored_lowpass2x' if args.architecture == 'anchored_lowpass' else 'anchored_detail2x', detail_channels=args.detail_channels,
                                  detail_blocks=args.detail_blocks)
            torch.save(checkpoint, args.out / f'step{step:06d}.pth')
            if ema_model is not None:
                ema_state = merge_adapters(ema_model).state_dict() if subspace else ema_model.state_dict()
                torch.save({**checkpoint, 'model': ema_state, 'weights': 'EMA decay 0.999'},
                           args.out / f'ema{step:06d}.pth')
            if adversary:
                torch.save({'discriminator': adversary.discriminator.state_dict(),
                    'optimizer': adversary.optimizer.state_dict(), 'step': step}, args.out / f'discriminator{step:06d}.pth')
    (args.out / 'complete.json').write_text(json.dumps({'steps': args.steps,
        'minutes': (time.monotonic()-started)/60,
        'temporal_static_coverage_mean': float(np.mean(temporal_coverage)) if temporal_coverage else None,
        'max_applied_head_gradient_ratio': float(max_applied_head_ratio) if args.gan_head_ratio_cap else None,
        'min_adversarial_scale': float(min_adversarial_scale) if args.gan_head_ratio_cap else None}) + '\n')


if __name__ == '__main__':
    main()
