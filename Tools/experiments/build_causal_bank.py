#!/usr/bin/env python3
"""Build genuine 2x codec sequences from explicit source-family split records.

Decode each HR sequence once. Encode its downsampled frames, then decode the
entire LR sequence once. No independent seeking between LR and HR is permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, data=None):
    result = subprocess.run(args, input=data, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    return result.stdout


def probe(path):
    info = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration', '-of', 'json', str(path)]))
    stream = info['streams'][0]
    return int(stream['width']), int(stream['height']), float(info['format']['duration']), stream['r_frame_rate']


def validate_sources(sources):
    ids, families, hashes = set(), {}, {}
    for s in sources:
        if s['id'] in ids or not s['id'].replace('_', '').isalnum():
            raise ValueError('unique alphanumeric source ids required')
        if s['split'] not in ('train', 'validation'):
            raise ValueError('explicit train or validation split required')
        if s['family'] in families and families[s['family']] != s['split']:
            raise ValueError(f"source family {s['family']} crosses split")
        if s['sha256'] in hashes and hashes[s['sha256']] != s['split']:
            raise ValueError('identical source content crosses split')
        ids.add(s['id']); families[s['family']] = s['split']; hashes[s['sha256']] = s['split']
    if set(families.values()) != {'train', 'validation'}:
        raise ValueError('both training and source-family-held-out validation required')


def build_source(source, index, args):
    path = Path(source['path'])
    sw, sh, duration, rate = source['stream']
    rng = np.random.default_rng(args.seed + index)
    side, frames = args.patch * 2, args.frames
    span = frames / float(Fraction(rate))
    if duration < span + 0.1:
        raise ValueError(f'{path}: too short')
    records = []
    for j in range(args.sequences_per_source):
        start = float(rng.uniform(0, max(0, duration-span-0.05)))
        crop = min(side * int(rng.choice([1, 2, 3])), sw, sh) // 2 * 2
        cx = int(rng.integers(0, (sw-crop)//2+1)) * 2
        cy = int(rng.integers(0, (sh-crop)//2+1)) * 2
        codec = 'h264' if j % 2 == 0 else 'vp9'
        quality = int(rng.choice([20, 26, 32, 38]))
        identity = f"{source['id']}-{j:03d}"
        target = args.out / source['split'] / (identity+'.npz')
        hr_command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(start), '-i', str(path),
            '-an', '-frames:v', str(frames), '-vf', f'crop={crop}:{crop}:{cx}:{cy},scale={side}:{side}:flags=lanczos',
            '-threads', '2', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
        raw = command(hr_command)
        if len(raw) != frames * side * side * 3:
            raise ValueError(f'{identity}: incomplete HR sequence')
        hr = np.frombuffer(raw, np.uint8).reshape(frames, side, side, 3)
        with tempfile.TemporaryDirectory(prefix='lucid-causal-bank-') as temporary:
            encoded = str(Path(temporary)/'lr.mkv')
            options = ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(quality)] if codec == 'h264' else [
                '-c:v', 'libvpx-vp9', '-deadline', 'good', '-cpu-used', '4', '-crf', str(quality), '-b:v', '0']
            encode = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{side}x{side}', '-r', rate, '-i', '-', '-vf',
                f'scale={args.patch}:{args.patch}:flags=lanczos:out_color_matrix=bt709:out_range=tv',
                *options, '-threads', '2', '-g', '60', '-pix_fmt', 'yuv420p',
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', encoded]
            command(encode, raw)
            decoded = command(['ffmpeg', '-nostdin', '-v', 'error', '-i', encoded,
                '-frames:v', str(frames), '-fps_mode', 'passthrough', '-threads', '2', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
        if len(decoded) != frames * args.patch * args.patch * 3:
            raise ValueError(f'{identity}: LR/HR frame-count mismatch')
        lr = np.frombuffer(decoded, np.uint8).reshape(frames, args.patch, args.patch, 3)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.partial.npz')
        np.savez_compressed(partial, lr=lr, hr=hr)
        partial.replace(target)
        records.append({'id': identity, 'file': str(target.relative_to(args.out)),
            'sha256': digest(target), 'source_id': source['id'], 'family': source['family'],
            'source_sha256': source['sha256'], 'split': source['split'], 'scale': 2,
            'frames': frames, 'frame_rate': rate, 'start_seconds': start,
            'crop': [cx, cy, crop, crop], 'reference_size': [side, side],
            'codec': codec, 'crf': quality, 'gop': 60, 'hr_command': hr_command,
            'reference_temporal_change': float(np.abs(np.diff(hr.astype(np.float32), axis=0)).mean()),
            'reference_std': float(hr.std())})
        print(identity, source['split'], codec, quality, flush=True)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--sequences-per-source', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260906)
    args = ap.parse_args()
    if args.patch % 4 or args.patch < 32 or args.frames < 8 or args.sequences_per_source < 2:
        ap.error('patch divisible by 4 and >=32, frames >=8, sequences/source >=2 required')
    args.out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    for source in sources:
        source['path'] = str(Path(source['path']).resolve())
        source['sha256'] = digest(source['path'])
        source['stream'] = probe(source['path'])
    validate_sources(sources)
    # No source can cross a split, regardless of how many crops it contributes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [r for batch in pool.map(lambda item: build_source(item[1], item[0], args), enumerate(sources)) for r in batch]
    manifest = {'schema': 1, 'scale': 2, 'frames': args.frames, 'lr_patch': args.patch,
        'seed': args.seed, 'sources': sources, 'sequences': records,
        'ffmpeg_version': command(['ffmpeg', '-version']).decode().splitlines()[0],
        'limitation': 'prototype training bank; compressed source masters; validation family excludes training but is not an untouched final test set'}
    partial = args.out/'manifest.partial.json'; partial.write_text(json.dumps(manifest, indent=2)+'\n')
    partial.replace(args.out/'manifest.json')
    print(f'complete: {len(records)} sequences', flush=True)


if __name__ == '__main__':
    main()
