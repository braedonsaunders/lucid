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
from architectures.causal_detail_v2 import make_model
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
    if manifest.get('storage') == 'mmap-pairs-v1':
        from mmap_training_bank import load_mapped_bank
        return load_mapped_bank(directory, manifest)
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


def batch(data, rng, count, frames, crop, teachers=None):
    sources, references, targets = [], [], []
    for _ in range(count):
        lr, hr, identity = data[int(rng.integers(len(data)))]
        t = int(rng.integers(lr.shape[0]-frames+1))
        y, x = (int(rng.integers(lr.shape[d]-crop+1)) for d in (1, 2))
        a = lr[t:t+frames, y:y+crop, x:x+crop]
        b = hr[t:t+frames, y*2:(y+crop)*2, x*2:(x+crop)*2]
        c = teachers[identity][t:t+frames, y*2:(y+crop)*2, x*2:(x+crop)*2] if teachers is not None else None
        for axis in (1, 2):
            if rng.random() < 0.5:
                a, b = np.flip(a, axis), np.flip(b, axis)
                if c is not None:
                    c = np.flip(c, axis)
        if rng.random() < 0.5:
            a, b = np.swapaxes(a, 1, 2), np.swapaxes(b, 1, 2)
            if c is not None:
                c = np.swapaxes(c, 1, 2)
        sources.append(a.copy()); references.append(b.copy())
        if c is not None:
            targets.append(c.copy())
    def tensor(images):
        return torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3).float().div_(255)
    pair = tensor(sources), tensor(references)
    return (*pair, tensor(targets)) if teachers is not None else pair


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
        floor = model.floor(x.flatten(0, 1)).reshape_as(target) if hasattr(model, 'floor') else baseline
        residual_change = float(torch.diff(output-target, dim=1).abs().mean())
        rows.append({'id': identity, 'psnr_rgb': -10*math.log10(max(mse, 1e-12)),
            'mse_rgb': mse, 'bilinear_mse_rgb': float((baseline-target).square().mean()),
            'floor_mse_rgb': float((floor.clamp(0, 1)-target).square().mean()),
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
    ap.add_argument('--architecture', choices=('causal_detail_v1', 'causal_detail_v2'), default='causal_detail_v1')
    ap.add_argument('--init', type=Path, help='Trusted local checkpoint; starts a new optimizer/schedule for fine-tuning')
    ap.add_argument('--dino-weight', type=float, default=0)
    ap.add_argument('--dino-repository', type=Path)
    ap.add_argument('--dino-checkpoint', type=Path)
    ap.add_argument('--dino-size', type=int, default=224)
    ap.add_argument('--teacher-cache', type=Path)
    ap.add_argument('--teacher-weight', type=float, default=0)
    ap.add_argument('--teacher-policy', choices=('reference_checked', 'unfiltered'), default='reference_checked',
                    help='Controlled ablation: unfiltered retains reference losses but omits the local teacher gate')
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
    if not math.isfinite(args.dino_weight) or args.dino_weight < 0:
        ap.error('DINO weight must be finite and nonnegative')
    if args.dino_weight and (not args.dino_repository or not args.dino_checkpoint):
        ap.error('DINO supervision requires a pinned local repository and checkpoint')
    if not math.isfinite(args.teacher_weight) or args.teacher_weight < 0 or (args.teacher_weight and not args.teacher_cache):
        ap.error('teacher weight must be finite, nonnegative, and accompanied by a cache')
    if args.teacher_weight and args.architecture != 'causal_detail_v2':
        ap.error('reference-checked distillation requires the v2 fixed reconstruction floor')
    if args.out.exists():
        ap.error('preserve existing training output; use a fresh directory')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest, data = load_bank(args.bank)
    targets = None
    if args.teacher_cache:
        from teacher_distillation import load_teacher_cache, reference_checked_loss
        targets, _ = load_teacher_cache(args.teacher_cache, digest(args.bank/'manifest.json'), data, digest)
    if args.crop > manifest['lr_patch']:
        ap.error('crop exceeds bank dimensions')
    rng = np.random.default_rng(args.seed)
    model = make_model(args.architecture, args.channels, args.blocks, scale=2).to(device)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=False)
        expected = {'architecture': args.architecture, 'channels': args.channels,
                    'blocks': args.blocks, 'scale': 2, 'no_history': args.no_history}
        if any(initial[k] != value for k, value in expected.items()):
            ap.error('initial checkpoint architecture/history differs from this experiment')
        model.load_state_dict(initial['model'], strict=True)
        del initial
    teacher = None
    if args.dino_weight:
        from dino_supervision import DinoSupervision
        teacher = DinoSupervision(args.dino_repository, args.dino_checkpoint, args.dino_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr*0.01)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {'architecture': args.architecture, 'scale': 2, 'channels': args.channels,
        'parameters': sum(p.numel() for p in model.parameters()),
        'init_sha256': digest(args.init) if args.init else None,
        'dino': teacher.metadata if teacher is not None else None,
        'dino_code_sha256': digest(Path(__file__).with_name('dino_supervision.py')) if teacher is not None else None,
        'teacher_cache_sha256': digest(args.teacher_cache/'manifest.json') if args.teacher_cache else None,
        'distillation_code_sha256': digest(Path(__file__).with_name('teacher_distillation.py')) if args.teacher_cache else None,
        'dino_frame_selection': 'one frame per sequence; zero-based step modulo curriculum frame count',
        'blocks': args.blocks, 'no_history': args.no_history, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank/'manifest.json'), 'torch': str(torch.__version__),
        'device': str(device), 'training_code_sha256': digest(__file__),
        'architecture_sha256': digest(Path(__file__).resolve().parents[1]/'architectures/causal_detail.py'),
        'architecture_v2_sha256': digest(Path(__file__).resolve().parents[1]/'architectures/causal_detail_v2.py')}
    (args.out/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    started = time.monotonic()
    print(json.dumps(metadata), flush=True)
    print(f'{len(data["train"])} train / {len(data["validation"])} source-family-held-out sequences', flush=True)
    for step in range(args.steps):
        # Identical curriculum and frames for the recurrent and control runs.
        frames = min(manifest['frames'], 3 if step < args.steps//5 else (7 if step < args.steps*3//5 else 12))
        sampled = batch(data['train'], rng, args.batch, frames, args.crop, targets)
        lr, hr = sampled[:2]
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
        if teacher is not None:
            selected = step % frames
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and args.precision == 'bf16'):
                feature = teacher(output[:, selected], hr[:, selected], lr[:, selected])
            loss = loss + args.dino_weight*feature
        coverage = output.new_tensor(0)
        if args.teacher_weight:
            selected = step % frames
            restored = sampled[2][:, selected].to(device)
            if args.teacher_policy == 'reference_checked':
                with torch.no_grad():
                    floor = model.floor(lr[:, selected]).clamp(0, 1)
                distilled, coverage = reference_checked_loss(output[:, selected], hr[:, selected], restored, floor)
            else:
                distilled = (output[:, selected] - restored).abs().mean()
                coverage = output.new_tensor(1)
            loss = loss + args.teacher_weight*distilled
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite training loss at step {step+1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step(); scheduler.step()
        if (step+1) % 100 == 0:
            print(f'step {step+1}/{args.steps} frames={frames} loss={float(loss.detach()):.6f} pixel={float(pixel.detach()):.6f} teacher_confidence={float(coverage):.4f} rate={(step+1)/(time.monotonic()-started):.2f} steps/s', flush=True)
        if (step+1) % 2000 == 0 or step+1 == args.steps:
            rows = validate(model, data['validation'], device, args.no_history)
            state = {**metadata, 'step': step+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'validation': rows,
                'rng': {'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
            temporary = args.out/'latest.tmp.pth'; torch.save(state, temporary); temporary.replace(args.out/'latest.pth')
            torch.save(state, args.out/f'step{step+1:06d}.pth')
            print(f'validation step={step+1} psnr={-10*math.log10(max(np.mean([r["mse_rgb"] for r in rows]), 1e-12)):.3f} bilinear={-10*math.log10(max(np.mean([r["bilinear_mse_rgb"] for r in rows]), 1e-12)):.3f} floor={-10*math.log10(max(np.mean([r["floor_mse_rgb"] for r in rows]), 1e-12)):.3f}', flush=True)
    print(f'finished {args.steps} steps in {(time.monotonic()-started)/60:.2f} minutes', flush=True)


if __name__ == '__main__':
    main()
