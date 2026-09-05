#!/usr/bin/env python3
"""Encode full video frames before taking aligned training patches.

This separate bank tests codec context and stream warmup. It does not replace
the frozen bank used for the v1/v2 architecture comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from build_causal_bank import command, digest, probe, validate_sources


def aligned_patch(lr, hr, start, frames, x, y, side):
    if hr.shape != (lr.shape[0], lr.shape[1]*2, lr.shape[2]*2, 3):
        raise ValueError('full-frame LR/HR alignment mismatch')
    if min(start, x, y) < 0 or start+frames > len(lr) or x+side > lr.shape[2] or y+side > lr.shape[1]:
        raise ValueError('patch extends outside aligned sequence')
    a = lr[start:start+frames, y:y+side, x:x+side].copy()
    b = hr[start:start+frames, y*2:(y+side)*2, x*2:(x+side)*2].copy()
    return a, b


def build_source(source, index, args):
    sw, sh, duration, rate = source['stream']
    rng = np.random.default_rng(args.seed+index)
    records = []
    for window in range(args.windows_per_source):
        width = min(sw//8*8, int(rng.choice(args.widths)))
        height = int(width*sh/sw)//8*8
        if min(width, height) < args.patch*2:
            raise ValueError('source cannot supply the requested patch without upscaling')
        warmup = int(rng.choice([8, 16, 24]))
        count = warmup+args.frames
        span = count/float(Fraction(rate))
        if duration < span+0.1:
            raise ValueError('source too short for codec warmup')
        start = float(rng.uniform(0, duration-span-0.05))
        codec = 'h264' if window % 2 == 0 else 'vp9'
        bpp = float(rng.choice([0.02, 0.05, 0.10]))
        bitrate = round(width*height/4*float(Fraction(rate))*bpp)
        hr_command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(start), '-i', source['path'],
            '-an', '-frames:v', str(count), '-vf', f'scale={width}:{height}:flags=lanczos',
            '-threads', '2', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
        raw = command(hr_command)
        if len(raw) != count*width*height*3:
            raise ValueError('incomplete full-frame reference')
        hr = np.frombuffer(raw, np.uint8).reshape(count, height, width, 3)
        with tempfile.TemporaryDirectory(prefix='lucid-stream-bank-') as directory:
            encoded = str(Path(directory)/'stream.mkv')
            options = ['-c:v', 'libx264', '-preset', 'medium'] if codec == 'h264' else [
                '-c:v', 'libvpx-vp9', '-deadline', 'good', '-cpu-used', '4']
            encode_command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}', '-r', rate, '-i', '-', '-vf',
                f'scale={width//2}:{height//2}:flags=lanczos:out_color_matrix=bt709:out_range=tv',
                *options, '-b:v', str(bitrate), '-maxrate', str(bitrate), '-bufsize', str(bitrate*2),
                '-threads', '2', '-g', '60', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709',
                '-color_primaries', 'bt709', '-color_trc', 'bt709', encoded]
            command(encode_command, raw)
            encoded_hash, encoded_size = digest(encoded), Path(encoded).stat().st_size
            decoded = command(['ffmpeg', '-nostdin', '-v', 'error', '-i', encoded,
                '-frames:v', str(count), '-fps_mode', 'passthrough', '-threads', '2',
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
        if len(decoded) != count*(width//2)*(height//2)*3:
            raise ValueError('incomplete full-frame degraded sequence')
        lr = np.frombuffer(decoded, np.uint8).reshape(count, height//2, width//2, 3)
        for patch in range(args.patches_per_window):
            x = int(rng.integers(0, width//2-args.patch+1))
            y = int(rng.integers(0, height//2-args.patch+1))
            a, b = aligned_patch(lr, hr, warmup, args.frames, x, y, args.patch)
            identity = f"{source['id']}-stream-{window:03d}-{patch:02d}"
            target = args.out/source['split']/(identity+'.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.partial.npz')
            np.savez_compressed(partial, lr=a, hr=b)
            partial.replace(target)
            records.append({'id': identity, 'file': str(target.relative_to(args.out)),
                'sha256': digest(target), 'source_id': source['id'], 'family': source['family'],
                'source_sha256': source['sha256'], 'split': source['split'], 'scale': 2,
                'frames': args.frames, 'frame_rate': rate, 'start_seconds': start,
                'codec_warmup_frames': warmup, 'full_reference_size': [width, height],
                'lr_patch_xy': [x, y], 'reference_size': [args.patch*2, args.patch*2],
                'codec': codec, 'rate_control': 'constrained bitrate', 'target_bitrate': bitrate,
                'target_bits_per_pixel_frame': bpp, 'gop': 60, 'encoded_sha256': encoded_hash,
                'encoded_container_bytes': encoded_size, 'hr_command': hr_command,
                'encode_command': encode_command, 'reference_std': float(b.std()),
                'reference_temporal_change': float(np.abs(np.diff(b.astype(np.float32), axis=0)).mean())})
        print(source['id'], window, codec, f'{width//2}x{height//2}', bitrate, f'warmup={warmup}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--patch', type=int, default=128)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--widths', type=int, nargs='+', default=[1280, 1920])
    parser.add_argument('--windows-per-source', type=int, default=2)
    parser.add_argument('--patches-per-window', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    if args.patch < 32 or args.patch % 4 or args.frames < 8 or args.windows_per_source < 2 or args.patches_per_window < 1:
        parser.error('patch >=32 divisible by 4, frames >=8, windows >=2, patches >=1 required')
    if any(w < args.patch*2 or w % 8 for w in args.widths):
        parser.error('widths must be divisible by 8 and fit an HR patch')
    if args.out.exists():
        parser.error('output directory already exists; preserve completed and partial banks')
    sources = json.loads(args.sources.read_text())
    for source in sources:
        source['path'] = str(Path(source['path']).resolve())
        source['sha256'] = digest(source['path'])
        source['stream'] = probe(source['path'])
    validate_sources(sources)
    args.out.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [row for batch in pool.map(lambda pair: build_source(pair[1], pair[0], args), enumerate(sources)) for row in batch]
    manifest = {'schema': 1, 'scale': 2, 'frames': args.frames, 'lr_patch': args.patch,
        'seed': args.seed, 'sources': sources, 'sequences': records,
        'builder': 'full-frame-codec-before-patch', 'builder_sha256': digest(__file__),
        'ffmpeg_version': command(['ffmpeg', '-version']).decode().splitlines()[0],
        'limitation': 'compressed source masters; short stream warmup; validation remains development data'}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'complete: {len(records)} sequences', flush=True)


if __name__ == '__main__':
    main()
