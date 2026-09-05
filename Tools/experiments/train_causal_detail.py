#!/usr/bin/env python3
"""Train the causal reconstruction prototype and its matched history-free control."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail import CausalDetail
from reconstruction_loss import sobel_loss
from build_causal_bank import validate_sources


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_bank(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    validate_sources(manifest['sources'])
    if manifest['scale'] != 2:
        raise ValueError('training requires a genuine 2x bank')
    result = {'train': [], 'validation': []}
    source_index = {s['id']: s for s in manifest['sources']}
    seen = set()
    for seq in manifest['sequences']:
        source = source_index[seq['source_id']]
        if seq['id'] in seen or any(seq[k] != source[k] for k in ('split', 'family')) or seq['source_sha256'] != source['sha256']:
            raise ValueError('sequence provenance disagrees with source split')
        seen.add(seq['id'])
        path = directory / seq['file']
        if digest(path) != seq['sha256']:
            raise ValueError(f'changed sequence: {seq["id"]}')
        with np.load(path, allow_pickle=False) as pair:
            lr, hr = pair['lr'].copy(), pair['hr'].copy()
        if lr.dtype != np.uint8 or hr.dtype != np.uint8 or lr.shape[0] != manifest['frames'] or lr.shape[0] != hr.shape[0] or hr.shape[1:3] != (lr.shape[1]*2, lr.shape[2]*2):
            raise ValueError('sequence alignment or scale mismatch')
        result[seq['split']].append((lr, hr, seq['id']))
    if not all(result.values()):
        raise ValueError('both source-disjoint splits required')
    return manifest, result


def batch(data, rng, count, frames, crop):
    sources, references = [], []
    for _ in range(count):
        lr, hr, _ = data[int(rng.integers(len(data)))]
        t = int(rng.integers(lr.shape[0]-frames+1))
        y, x = (int(rng.integers(lr.shape[d]-crop+1)) for d in (1, 2))
        a = lr[t:t+frames, y:y+crop, x:x+crop]
        b = hr[t:t+frames, y*2:(y+crop)*2, x*2:(x+crop)*2]
        for axis in (1, 2):
            if rng.random() < 0.5:
                a, b = np.flip(a, axis), np.flip(b, axis)
        if rng.random() < 0.5:
            a, b = np.swapaxes(a, 1, 2), np.swapaxes(b, 1, 2)
        sources.append(a.copy()); references.append(b.copy())
    def tensor(images):
        return torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3).float().div_(255)
    return tensor(sources), tensor(references)


def sequence(model, frames, no_history):
    state = model.initial_state(frames[:, 0])
    results = []
    for t in range(frames.shape[1]):
        valid = frames.new_full((frames.shape[0], 1, 1, 1), float(t > 0 and not no_history))
        output, state = model(frames[:, t], state, valid)
        results.append(output)
    return torch.stack(results, dim=1)


@torch.inference_mode()
def validate(model, data, device, no_history):
    model.eval()
    rows = []
    for lr, hr, identity in data:
        x = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().to(device)/255
        target = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().to(device)/255
        output = sequence(model, x, no_history).clamp(0, 1)
        mse = float((output-target).square().mean())
        baseline = F.interpolate(x.flatten(0, 1), scale_factor=2, mode='bilinear', align_corners=False).reshape_as(target)
        residual_change = float(torch.diff(output-target, dim=1).abs().mean())
        rows.append({'id': identity, 'psnr_rgb': -10*math.log10(max(mse, 1e-12)),
            'mse_rgb': mse, 'bilinear_mse_rgb': float((baseline-target).square().mean()),
            'bilinear_psnr_rgb': -10*math.log10(max(float((baseline-target).square().mean()), 1e-12)),
            'temporal_residual_l1': residual_change})
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--lr', type=float, default=0.0002)
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-history', action='store_true')
    ap.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    args = ap.parse_args()
    if args.steps < 1 or args.batch < 1 or args.crop < 16 or args.crop % 4:
        ap.error('positive steps/batch and crop >=16 divisible by 4 required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest, data = load_bank(args.bank)
    if args.crop > manifest['lr_patch']:
        ap.error('crop exceeds bank dimensions')
    rng = np.random.default_rng(args.seed)
    model = CausalDetail(args.channels, args.blocks, scale=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*0.01)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {'architecture': 'causal_detail_v1', 'scale': 2, 'channels': args.channels,
        'blocks': args.blocks, 'no_history': args.no_history, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank/'manifest.json'), 'torch': str(torch.__version__),
        'device': str(device), 'training_code_sha256': digest(__file__),
        'architecture_sha256': digest(Path(__file__).resolve().parents[1]/'architectures/causal_detail.py')}
    (args.out/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    started = time.monotonic()
    print(json.dumps(metadata), flush=True)
    print(f'{len(data["train"])} train / {len(data["validation"])} source-family-held-out sequences', flush=True)
    for step in range(args.steps):
        # Identical curriculum and frames for the recurrent and control runs.
        frames = min(manifest['frames'], 3 if step < args.steps//5 else (7 if step < args.steps*3//5 else 12))
        lr, hr = batch(data['train'], rng, args.batch, frames, args.crop)
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and args.precision == 'bf16'):
            output = sequence(model, lr, args.no_history)
        output = output.float()
        difference = output-hr
        pixel = torch.sqrt(difference.square()+1e-6).mean()
        edge = sobel_loss(output.flatten(0, 1), hr.flatten(0, 1))
        spectral = (torch.fft.rfft2(output, norm='ortho')-torch.fft.rfft2(hr, norm='ortho')).abs().mean()
        temporal = torch.diff(difference, dim=1).abs().mean()
        loss = pixel + 0.05*edge + 0.01*spectral + 0.02*temporal
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite training loss at step {step+1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step(); scheduler.step()
        if (step+1) % 100 == 0:
            print(f'step {step+1}/{args.steps} frames={frames} loss={float(loss.detach()):.6f} pixel={float(pixel.detach()):.6f} rate={(step+1)/(time.monotonic()-started):.2f} steps/s', flush=True)
        if (step+1) % 2000 == 0 or step+1 == args.steps:
            rows = validate(model, data['validation'], device, args.no_history)
            state = {**metadata, 'step': step+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'validation': rows,
                'rng': {'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
            temporary = args.out/'latest.tmp.pth'; torch.save(state, temporary); temporary.replace(args.out/'latest.pth')
            torch.save(state, args.out/f'step{step+1:06d}.pth')
            print(f'validation step={step+1} psnr={-10*math.log10(max(np.mean([r["mse_rgb"] for r in rows]), 1e-12)):.3f} bilinear={-10*math.log10(max(np.mean([r["bilinear_mse_rgb"] for r in rows]), 1e-12)):.3f}', flush=True)
    print(f'finished {args.steps} steps in {(time.monotonic()-started)/60:.2f} minutes', flush=True)


if __name__ == '__main__':
    main()
