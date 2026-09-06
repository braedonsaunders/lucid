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


def reconstruction_objective(output, reference, intended, detail_target='mixture'):
    if detail_target not in ('mixture', 'reference'):
        raise ValueError('unknown detail supervision target')
    detail = reference if detail_target == 'reference' else intended
    return (F.l1_loss(output, intended) + .2 * sobel_loss(output, detail)
            + .05 * fft_loss(output, detail) + .1 * F.l1_loss(output, reference))


def bounded_adversarial_scale(base_norm, weighted_adversarial_norm, ratio_cap):
    """Only attenuate the GAN term; the detached scale adds no second derivatives."""
    if not math.isfinite(ratio_cap) or ratio_cap <= 0:
        raise ValueError('positive finite head-gradient ratio cap required')
    return (ratio_cap * base_norm.detach() / weighted_adversarial_norm.detach().clamp_min(1e-12)).clamp(0, 1)


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
    ap.add_argument('--intended', choices=['mixture', 'reference'], default='mixture',
                    help='Regression target: the fixed shipping/teacher mixture (control) or the HR reference itself')
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
    if args.detail_channels < 4 or args.detail_blocks < 1:
        ap.error('detail channels >=4 and positive block count required')
    if not math.isfinite(args.dino_gan_weight) or args.dino_gan_weight < 0:
        ap.error('nonnegative finite adversarial weight required')
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
    bank_hash = digest(args.bank / 'manifest.json')
    teacher, teacher_receipt = load_teacher_cache(args.pixrestore_cache, bank_hash, data, digest)
    shipping, _, frames = load(args.init, 'cuda')
    if frames != 1:
        raise ValueError('single-frame shipping initialization required')
    target, shipping_receipt = shipping_targets(shipping, data, args.shipping_cache, bank_hash, digest(args.init))
    # Form the full-frame RGB8 mixture before augmentation, including both arms.
    mixed = {identity: np.rint(pixels.astype(np.float32) * (1-args.teacher_mix)
             + teacher[identity].astype(np.float32) * args.teacher_mix).astype(np.uint8)
             for identity, pixels in target.items()}
    if args.intended == 'reference':
        # The fixed 50% shipping/PixRestore mixture scores only +2.9% LPIPS / +6.6% DISTS
        # over shipping on the development set, below what the paired-critic student
        # already reaches, so that target holds the student back; regress to truth.
        mixed = {identity: hr for _, hr, identity in data['train']}
    del teacher, target
    model = fold_head(shipping).cuda().train()
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
    rng = np.random.default_rng(args.seed)
    adversary = None
    if args.dino_gan_weight:
        from dino_adversary import DinoAdversary
        if args.paired_dino:
            from paired_dino_adversary import PairedDinoAdversary
            adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
        else:
            adversary = DinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)
    args.out.mkdir(parents=True)
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
        'teacher_manifest_sha256': digest(args.pixrestore_cache / 'manifest.json'),
        'shipping_manifest_sha256': digest(args.shipping_cache / 'manifest.json'),
        'source_hashes': {str(p.name): digest(p) for p in (Path(__file__),
            Path(__file__).with_name('fold_shipping_head.py'), Path(__file__).with_name('train_causal_detail.py'),
            Path(__file__).resolve().parents[1] / 'train_span.py',
            Path(__file__).resolve().parents[1] / 'architectures/span_arch.py')},
        'loss': f'L1 to {"HR reference" if args.intended == "reference" else "fixed mixture"} + 0.2 signed Sobel to {args.detail_target} + 0.05 FFT to {args.detail_target} + 0.1 L1 to reference; 8 output-pixel border excluded',
        'precision': 'CUDA BF16 autocast; AdamW FP32; no compilation',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
        'purpose': 'controlled quality/performance experiment; not shipping promotion'}
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
    started = time.monotonic()
    anchor_hash = None
    max_applied_head_ratio = torch.zeros((), device='cuda')
    min_adversarial_scale = torch.ones((), device='cuda')
    for step in range(1, args.steps + 1):
        x, reference, intended = batch(data['train'], rng, args.batch, 1, args.crop, mixed)
        x, reference, intended = (v[:, 0].cuda() for v in (x, reference, intended))
        if step == 1:
            experiment['first_batch_sha256'] = state_digest({'source': x, 'reference': reference, 'intended': intended})
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
        output, reference, intended = (v.float()[:, :, 8:-8, 8:-8] for v in (output, reference, intended))
        loss = reconstruction_objective(output, reference, intended, args.detail_target)
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
            if adversary:
                torch.save({'discriminator': adversary.discriminator.state_dict(),
                    'optimizer': adversary.optimizer.state_dict(), 'step': step}, args.out / f'discriminator{step:06d}.pth')
    (args.out / 'complete.json').write_text(json.dumps({'steps': args.steps,
        'minutes': (time.monotonic()-started)/60,
        'max_applied_head_gradient_ratio': float(max_applied_head_ratio) if args.gan_head_ratio_cap else None,
        'min_adversarial_scale': float(min_adversarial_scale) if args.gan_head_ratio_cap else None}) + '\n')


if __name__ == '__main__':
    main()
