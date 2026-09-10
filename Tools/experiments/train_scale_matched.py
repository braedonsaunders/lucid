#!/usr/bin/env python3
"""r91 section-4's matched single-stage 4x arm, and its scale=2 control.

r87/r91 measured that lucidbig2k's shipping 2x network, self-composed on its
own output, beats a high-quality stretch at 2560x1440 - but the one genuine
single-stage 4x-trained checkpoint in this repo (SPAN_x4_ch32u, an old
from-scratch L1-only run on the old corpus) loses to plain bilinear. That
result is confounded by recipe and corpus, not just decomposition. This
trainer removes the confound: it is a small, scoped fork of
train_ssm_span.py's loss orchestration - reconstruction_objective (imported,
not reimplemented) plus PairedDinoAdversary wiring, the same
--reconstruction-edge 0.2 --reconstruction-fft 0.05 defaults - with the
recurrence/scan machinery (SSMRecurrentSPAN, warp_previous, motion search)
removed entirely. What is left is a plain Unshuffled(scale, frames=1) trained
on a matched objective, so a fresh --scale 4 head (Arm B) and a fresh
--scale 2 control (Arm C, optional per r91 - Arm A is the already-shipped
checkpoint and needs no training) differ only in the one thing being tested.

Two data formats exist in this repo and neither can serve both scales:

  --bank   the real stream-bank sequence format lucidbig2k actually trained
           on (train_causal_detail.load_bank / mmap_training_bank). Both its
           loaders hard-assert scale == 2; there is no scale=4 stream bank
           anywhere in this repo (verified against every stream-bank-* on
           lucid-gpu-vpn: v2, v3, v4, v4c, v5, v6, big - all scale=2,
           lr 128x128 -> hr 256x256). Valid for --scale 2 only.

  --corpus the hr/lr patch-pair layout train_span.py's build_bank reads (the
           corpus that trained SPAN_x4_ch32u - 1,800 real x264/VP9 pairs,
           genuinely 4x). train_span.py's patch bank hardcodes its module
           SCALE=4 for the geometry check, so this path is valid for
           --scale 4 only. See .build/quality-breakthrough-r98-scalematched/
           README.md for whether the actual pixels exist yet - as of this
           writing they do not (pruned after the original training run);
           only the deterministic generator and the manifest survive.

Passing --bank with --scale 4, or --corpus with --scale 2, is refused before
any data is touched - that combination cannot be trained on data that means
what the recipe assumes it means.

  .venv-convert/bin/python Tools/experiments/train_scale_matched.py \\
      --scale 4 --corpus .build/corpus4 --bank-dir .build/bank-scale4 \\
      --init Model/weights/span_ch32u_final60k.pth \\
      --out .build/r98-arm-b --steps 2000 --seed 20260914 \\
      --dino-gan-weight 0.1 --pixrestore-repository ... \\
      --dino-repository ... --dino-checkpoint ...
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_span import Unshuffled
import train_span as _train_span
from train_causal_detail import batch as stream_batch, digest, load_bank as load_stream_bank
from train_presented_detail import reconstruction_objective


CROP_MARGIN = 8  # matches train_ssm_span.py's conv-border crop; not scale-dependent


def build_model(channels, scale, frames=1, version=1):
    """The architecture r91 says needs no new code: Unshuffled is already
    parametrised by scale (lucidbig2k passes 2, SPAN_x4_ch32u's trunk is the
    same class with scale=4). This just names that call so it can be tested
    once instead of re-derived at every call site."""
    if scale not in (2, 4):
        raise ValueError('net scale must be 2 or 4')
    return Unshuffled(channels, scale=scale, frames=frames, version=version)


def matched_objective(output, reference, decoded_current, scale, *, adversary=None, gan_weight=0,
                       edge_weight=0.2, fft_weight=0.05):
    """train_ssm_span.py's training_objective, generalised to any net scale.

    At scale=2 with gan_weight=0 (the default) this is byte-for-byte the same
    computation as that function - see test_train_scale_matched.py - because
    a matched control has to reuse the objective, not approximate it. The one
    real generalisation is the critic's bicubic condition, which must be
    upsampled by the net's own scale to land on the cropped output's size;
    train_ssm_span.py hardcodes scale_factor=2 because it only ever trains a
    2x backbone.
    """
    if scale not in (2, 4):
        raise ValueError('net scale must be 2 or 4')
    if not math.isfinite(gan_weight) or gan_weight < 0:
        raise ValueError('nonnegative finite critic weight required')
    if gan_weight and adversary is None:
        raise ValueError('positive critic weight requires an adversary')
    for name, value in (('edge', edge_weight), ('fft', fft_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'nonnegative finite {name} weight required')
    reconstruction = reconstruction_objective(output, reference, reference, 'reference',
                                              edge_weight=edge_weight, fft_weight=fft_weight)
    loss, fake = reconstruction, None
    metrics = {'reconstruction': float(reconstruction.detach())}
    if gan_weight:
        condition = F.interpolate(decoded_current.detach(), scale_factor=scale,
            mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)[
                ..., CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]
        if condition.shape != output.shape:
            raise ValueError('critic condition must match cropped reconstruction')
        with torch.autocast(output.device.type, dtype=torch.bfloat16, enabled=output.is_cuda):
            gan, fake = adversary.generator_loss(output, condition)
        loss = loss + gan_weight * gan
        metrics['gan_unweighted'] = float(gan.detach())
    return loss, fake, metrics


# ---- data sources -----------------------------------------------------------
# Two incompatible formats, one per scale. Each source is a (manifest, sample)
# pair; sample(rng, batch_size) returns (current, reference) float tensors in
# [0, 1], NCHW, on the CPU - device transfer happens in the training loop.

def stream_bank_source(bank_dir, crop=96):
    manifest, data = load_stream_bank(Path(bank_dir))

    def sample(rng, batch_size):
        lr, hr = stream_batch(data['train'], rng, batch_size, 1, crop)
        return lr[:, 0], hr[:, 0]

    return manifest, sample


def patch_corpus_source(corpus, bank_dir, per_pair=8, min_correlation=0.85, seed=0):
    if _train_span.SCALE != 4:
        raise ValueError(
            "train_span.py's patch-bank builder only ever produces a scale=4 "
            "bank (its SCALE module constant); this source cannot serve any "
            "other scale")
    bank_dir = Path(bank_dir)
    if not (bank_dir / 'bank.json').exists():
        _train_span.build_bank(str(corpus), str(bank_dir), per_pair=per_pair,
                               seed=seed, min_correlation=min_correlation)
    lr_bank, hr_bank, frames = _train_span.load_bank(str(bank_dir))
    if frames != 1:
        raise ValueError('single-frame patch bank required')
    count = len(lr_bank)
    if count == 0:
        raise ValueError('empty patch bank')

    def sample(rng, batch_size):
        indices = np.sort(rng.integers(0, count, size=batch_size))
        lr = torch.from_numpy(np.ascontiguousarray(lr_bank[indices])).permute(0, 3, 1, 2).float() / 255
        hr = torch.from_numpy(np.ascontiguousarray(hr_bank[indices])).permute(0, 3, 1, 2).float() / 255
        return lr, hr

    return {'count': count, 'lr_patch': _train_span.LR_PATCH, 'scale': _train_span.SCALE}, sample


def make_source(args):
    if bool(args.bank) == bool(args.corpus):
        raise ValueError('exactly one of --bank or --corpus is required')
    if args.bank:
        if args.scale != 2:
            raise ValueError(
                "--bank is the stream-bank sequence format lucidbig2k trained "
                "on; both its loaders hard-assert scale == 2 and there is no "
                "scale=4 stream bank in this repo (see the module docstring) "
                "- pass --corpus for a --scale 4 run")
        return stream_bank_source(args.bank, crop=args.crop)
    if args.scale != 4:
        raise ValueError(
            "--corpus is train_span.py's patch-pair format, which hardcodes "
            "a scale=4 geometry check; there is no scale=2 analogue of this "
            "corpus format in the repo - pass --bank for a --scale 2 run")
    return patch_corpus_source(args.corpus, args.bank_dir, per_pair=args.per_pair,
                               min_correlation=args.min_correlation, seed=args.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scale', type=int, required=True, choices=(2, 4))
    ap.add_argument('--bank', type=Path, help='stream-bank sequence directory (scale=2 only)')
    ap.add_argument('--corpus', type=Path, help='hr/lr patch corpus, train_span.py layout (scale=4 only)')
    ap.add_argument('--bank-dir', type=Path, default=Path('.build/bank-scale-matched'),
                    help='memmap patch-bank cache for --corpus; built once, reused after')
    ap.add_argument('--per-pair', type=int, default=8)
    ap.add_argument('--min-correlation', type=float, default=0.85)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--init', type=Path, help='warm-start weights, fresh optimiser at step 0')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96, help='--bank crop size; --corpus patches are pre-cropped and ignore this')
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5, help='matches train_ssm_span.py --joint backbone lr')
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--dino-gan-weight', type=float, default=0)
    ap.add_argument('--reconstruction-edge', type=float, default=0.2)
    ap.add_argument('--reconstruction-fft', type=float, default=0.05)
    ap.add_argument('--pixrestore-repository', type=Path)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    args = ap.parse_args()

    if args.out.exists():
        ap.error('fresh output path required')
    if min(args.steps, args.batch) < 1:
        ap.error('positive steps and batch required')
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    manifest, sample = make_source(args)  # raises before any CUDA/model work if the scale/format pair is wrong

    device = torch.device('cuda')
    model = build_model(args.channels, args.scale).to(device).train()
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'fine-tuning from {args.init} (step {state.get("step", "?")}) at a fresh step 0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    adversary = None
    if args.dino_gan_weight:
        from paired_dino_adversary import PairedDinoAdversary
        adversary = PairedDinoAdversary(args.pixrestore_repository, args.dino_repository, args.dino_checkpoint)

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    files = ['Tools/train_span.py', 'Tools/experiments/train_causal_detail.py',
             'Tools/experiments/train_presented_detail.py', 'Tools/experiments/train_ssm_span.py',
             'Tools/experiments/train_scale_matched.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'manifest': manifest if isinstance(manifest, dict) and 'sources' not in manifest else
            {k: v for k, v in manifest.items() if k not in ('sources', 'sequences')},
        'checkpoint_sha256': digest(args.init) if args.init else None,
        'source_hashes': {p: digest(source_root / p) for p in files},
        'recipe': ('final-frame HR reconstruction; scoped fork of train_ssm_span.py '
                  'with the recurrence/scan machinery removed; plain single-frame '
                  f'Unshuffled(scale={args.scale})'),
        'net_scale': args.scale, 'data_format': 'bank' if args.bank else 'corpus',
        'optimizer': f'AdamW, lr {args.lr}, no weight decay, cosine schedule (matches train_ssm_span backbone group)',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    if adversary is not None:
        experiment['adversary'] = adversary.metadata
        experiment['recipe'] += '; paired-DINO adversarial term, BF16 critic'
        for name in ('paired_dino_adversary.py', 'dino_adversary.py', 'dino_supervision.py'):
            path = 'Tools/experiments/' + name
            experiment['source_hashes'][path] = digest(source_root / path)
    (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')

    started = time.monotonic()
    for step in range(1, args.steps + 1):
        current, reference = sample(rng, args.batch)
        current, reference = current.to(device), reference.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(current)
        a = output[..., CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]
        b = reference[..., CROP_MARGIN:-CROP_MARGIN, CROP_MARGIN:-CROP_MARGIN]
        loss, fake_features, loss_metrics = matched_objective(
            a, b, current, args.scale, adversary=adversary, gan_weight=args.dino_gan_weight,
            edge_weight=args.reconstruction_edge, fft_weight=args.reconstruction_fft)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if adversary is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss_metrics['discriminator'] = adversary.update(b, fake_features)
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()), **loss_metrics,
                              'minutes': (time.monotonic() - started) / 60}), flush=True)

    checkpoint = {'model': model.state_dict(), 'channels': args.channels, 'frames': 1,
        'scale': args.scale, 'version': 1, 'step': args.steps, 'architecture': 'scale_matched_span',
        'experiment': experiment}
    destination = args.out / f'step{args.steps:06d}.pth'
    torch.save(checkpoint, destination)
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'steps': args.steps,
        'checkpoint_sha256': digest(destination), 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
