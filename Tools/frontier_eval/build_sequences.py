#!/usr/bin/env python3
"""Build reproducible, consecutive codec-degraded sequences for evaluation.

Every source is explicitly named. The reference is a declared SDR resize of
that same source, not an unrelated clean image or a different decode timestamp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def probe(path, count=False):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', *(['-count_frames'] if count else []),
        '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames',
        '-of', 'json', str(path)], timeout=120))['streams'][0]


def run(args):
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', *args], check=True, timeout=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', nargs=2, action='append', metavar=('ID', 'PATH'), required=True)
    ap.add_argument('--out', type=Path, default=Path('.build/frontier-sequences'))
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--width', type=int, default=320)
    ap.add_argument('--scale', type=int, choices=(2, 4), default=4)
    ap.add_argument('--bitrates', type=int, nargs='+', default=[120000, 400000])
    args = ap.parse_args()
    height = args.width * 9 // 16
    if args.frames < 12 or args.width % 16 or height % 2 or min(args.bitrates) <= 0:
        ap.error('need at least 12 frames and an even 16:9 model size with width divisible by 16')
    ids = [s[0] for s in args.source]
    if len(ids) != len(set(ids)) or any(not i.replace('_', '').isalnum() for i in ids):
        ap.error('source ids must be unique alphanumeric names (underscores allowed)')
    for _, path in args.source:
        if not Path(path).is_file():
            ap.error(f'missing source {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    # Reject bad inputs before producing any part of a new manifest.
    source_info = {name: probe(path) for name, path in args.source}
    records = []
    for source_id, path in args.source:
        path = Path(path).resolve()
        info = source_info[source_id]
        source_hash = digest(path)
        if int(info['width']) < args.width * args.scale or int(info['height']) < height * args.scale:
            ap.error(f'{source_id}: source cannot supply the declared reference size without upscaling')
        reference = args.out / f'{source_id}-reference.mkv'
        run(['-i', str(path), '-frames:v', str(args.frames), '-an', '-vf',
             f'scale={args.width*args.scale}:{height*args.scale}:flags=lanczos:in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=tv',
             '-pix_fmt', 'yuv420p', '-c:v', 'ffv1', '-threads', '2', str(reference)])
        reference_info = probe(reference, count=True)
        reference_hash = digest(reference)
        if int(reference_info['nb_read_frames']) != args.frames:
            raise RuntimeError(f'{source_id}: truncated reference')
        for codec in ('h264', 'vp9'):
            for bitrate in args.bitrates:
                identity = f'{source_id}-{codec}-{bitrate}'
                degraded = args.out / f'{identity}.mkv'
                encoder = ['-c:v', 'libx264', '-preset', 'medium'] if codec == 'h264' else [
                    '-c:v', 'libvpx-vp9', '-deadline', 'good', '-cpu-used', '4']
                run(['-i', str(reference), '-an', '-vf',
                     f'scale={args.width}:{height}:flags=lanczos:in_range=tv:out_range=tv',
                     *encoder, '-b:v', str(bitrate), '-g', '60', '-threads', '2',
                     '-pix_fmt', 'yuv420p', str(degraded)])
                if int(probe(degraded, count=True)['nb_read_frames']) != args.frames:
                    raise RuntimeError(f'{identity}: frame count mismatch')
                records.append({'id': identity, 'source_id': source_id,
                    'source_path': str(path), 'source_sha256': source_hash,
                    'reference': str(reference.resolve()), 'reference_sha256': reference_hash,
                    'degraded': str(degraded.resolve()), 'degraded_sha256': digest(degraded),
                    'frames': args.frames, 'frame_rate': reference_info['r_frame_rate'],
                    'width': args.width, 'height': height, 'scale': args.scale,
                    'codec': codec, 'bitrate': bitrate, 'gop': 60,
                    'color_contract': 'SDR Rec709 video range; existing source compression remains part of reference',
                    'split': 'development-validation; do not use for training'})
                print(f'{identity}: {args.frames} aligned frames', flush=True)
    (args.out / 'sequences.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
