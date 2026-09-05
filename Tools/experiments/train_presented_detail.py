#!/usr/bin/env python3
"""Distill a fixed shipping/PixRestore mixture into a shipping-initialized 2x model.

Full-context shipping targets are cached before random training crops. Both arms
share data/RNG/objectives; only the fixed teacher mixture differs. No flicker loss.
"""
import argparse
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
    args = ap.parse_args()
    if args.out.exists() or args.teacher_mix not in (0, .5) or args.steps < 1 or args.batch < 1 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, fixed teacher mix 0/.5, positive steps/batch and even crop >=32 required')
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error('positive finite learning rate required')
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
    del teacher, target
    model = fold_head(shipping).cuda().train()
    del shipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': bank_hash, 'checkpoint_sha256': digest(args.init),
        'teacher_manifest_sha256': digest(args.pixrestore_cache / 'manifest.json'),
        'shipping_manifest_sha256': digest(args.shipping_cache / 'manifest.json'),
        'source_hashes': {str(p.name): digest(p) for p in (Path(__file__),
            Path(__file__).with_name('fold_shipping_head.py'), Path(__file__).with_name('train_causal_detail.py'),
            Path(__file__).resolve().parents[1] / 'train_span.py',
            Path(__file__).resolve().parents[1] / 'architectures/span_arch.py')},
        'loss': 'L1 to fixed mixture + 0.2 signed Sobel to mixture + 0.05 FFT to mixture + 0.1 L1 to reference; 8 output-pixel border excluded',
        'precision': 'CUDA BF16 autocast; AdamW FP32; no compilation',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
        'purpose': 'controlled quality/performance experiment; not shipping promotion'}
    (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        x, reference, intended = batch(data['train'], rng, args.batch, 1, args.crop, mixed)
        x, reference, intended = (v[:, 0].cuda() for v in (x, reference, intended))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x)
        output, reference, intended = (v.float()[:, :, 8:-8, 8:-8] for v in (output, reference, intended))
        loss = F.l1_loss(output, intended) + .2 * sobel_loss(output, intended) \
             + .05 * fft_loss(output, intended) + .1 * F.l1_loss(output, reference)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        rate = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        if step % 200 == 0:
            print(f'step {step}/{args.steps} loss={float(loss):.6f} minutes={(time.monotonic()-started)/60:.2f}', flush=True)
        if step % 2000 == 0 or step == args.steps:
            torch.save({'model': model.state_dict(), 'channels': model.core.conv_1.eval_conv.out_channels,
                'scale': 2, 'frames': 1, 'version': model.version, 'step': step,
                'architecture': 'shipping_direct2x_area', 'experiment': experiment}, args.out / f'step{step:06d}.pth')
    (args.out / 'complete.json').write_text(json.dumps({'steps': args.steps,
        'minutes': (time.monotonic()-started)/60}) + '\n')


if __name__ == '__main__':
    main()
