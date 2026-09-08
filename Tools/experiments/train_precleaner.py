#!/usr/bin/env python3
"""Train only a tiny pre-cleaner, supervised by clean RGB LR; freeze the SR net."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from clean_lr_targets import batch_clean_lr, load_clean_targets
from train_causal_detail import digest, load_bank
from train_presented_detail import state_digest
from native_stages import preprocess_sequence
from eval_checkpoint import load
from architectures.precleaner import PrecleanedSPAN
from train_span import Unshuffled
from reconstruction_loss import sobel_loss


def cleaner_loss(prediction, target):
    # Covers both the 7px cleaner receptive field and cropped Lanczos boundary.
    prediction, target = (x[..., 4:-4, 4:-4] for x in (prediction, target))
    return F.l1_loss(prediction, target) + .2 * sobel_loss(prediction, target)


@torch.inference_mode()
def validate_cleaner(model, data, targets, native_inputs):
    rows = []
    for lr, _, identity in data:
        x = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        truth = torch.from_numpy(targets[identity].copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        # Match the training history contract for this diagnostic, not a full
        # native video claim. Score the third frame of each validation sequence.
        x, truth = x[:, :3], truth[:, 2]
        if native_inputs:
            x = preprocess_sequence(x)
        source = x[:, -1]
        prediction = model.cleaner(source)
        target = truth[..., 4:-4, 4:-4]
        raw_mse = float((source[..., 4:-4, 4:-4] - target).square().mean())
        clean_mse = float((prediction[..., 4:-4, 4:-4] - target).square().mean())
        rows.append({'id': identity, 'input_mse': raw_mse, 'cleaned_mse': clean_mse,
                     'input_psnr': -10 * math.log10(max(raw_mse, 1e-12)),
                     'cleaned_psnr': -10 * math.log10(max(clean_mse, 1e-12))})
    return {'complete': True, 'scope': 'source-disjoint clean-LR patch diagnostic, not SR quality',
            'mean_input_mse': float(np.mean([r['input_mse'] for r in rows])),
            'mean_cleaned_mse': float(np.mean([r['cleaned_mse'] for r in rows])), 'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'clean-cache', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--native-input-stages', action='store_true')
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch and even crop >=32 required')
    if not math.isfinite(args.lr) or args.lr <= 0 or args.channels < 4:
        ap.error('positive finite learning rate and channels >=4 required')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] < 3 or any(min(lr.shape[1:3]) < args.crop for rows in data.values() for lr, _, _ in rows):
        ap.error('three-frame sequences and patches covering the requested crop required')
    bank_hash = digest(args.bank / 'manifest.json')
    targets, target_manifest = load_clean_targets(args.clean_cache, data, bank_hash)
    sr, _, frames = load(args.init, 'cuda')
    if type(sr) is not Unshuffled or frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = PrecleanedSPAN(sr, args.channels).cuda().train()
    base_hash = state_digest(model.sr.state_dict())
    channels = sr.core.conv_1.eval_conv.out_channels
    optimizer = torch.optim.AdamW(model.cleaner.parameters(), lr=args.lr, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
                  'clean_target_manifest_sha256': digest(args.clean_cache / 'manifest.json'),
                  'target': target_manifest['target'], 'base_initial_state_sha256': base_hash,
                  'cleaner_parameters': sum(p.numel() for p in model.cleaner.parameters()),
                  'training': 'three aligned frames; only third-frame clean LR supervised; SR backbone frozen; no GAN',
                  'source_hashes': {p.name: digest(p) for p in (Path(__file__),
                      Path(__file__).with_name('clean_lr_targets.py'), Path(__file__).with_name('native_stages.py'),
                      Path(__file__).resolve().parents[1] / 'architectures/precleaner.py')},
                  'purpose': 'cleaner mechanism experiment; requires native SR quality and latency admission'}
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        x, clean, _ = batch_clean_lr(data['train'], targets, rng, args.batch, 3, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'decoded': x, 'clean': clean})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        if args.native_input_stages:
            with torch.no_grad():
                x = preprocess_sequence(x.cuda())
        source, target = x[:, -1].cuda(), clean[:, -1].cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.cleaner(source)
        loss = cleaner_loss(prediction.float(), target)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite cleaner loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cleaner.parameters(), 1)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    if state_digest(model.sr.state_dict()) != base_hash:
        raise ValueError('frozen SR backbone changed')
    checkpoint = {'model': model.state_dict(), 'architecture': 'precleaned_span2x',
                  'channels': channels, 'scale': 2, 'frames': 1, 'version': sr.version,
                  'step': args.steps, 'cleaner_channels': args.channels, 'experiment': experiment}
    torch.save(checkpoint, args.out / f'step{args.steps:06d}.pth')
    validation = validate_cleaner(model.eval(), data['validation'], targets, args.native_input_stages)
    (args.out / 'clean-lr-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'steps': args.steps,
        'minutes': (time.monotonic() - started) / 60, 'backbone_unchanged': True}) + '\n')


if __name__ == '__main__':
    main()
