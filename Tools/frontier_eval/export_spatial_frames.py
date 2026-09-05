#!/usr/bin/env python3
"""Export the same deterministic spatial frames used by sequence screening."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=4)
    args = parser.parse_args()
    if args.stride < 1 or args.out.exists():
        parser.error('positive stride and fresh output directory required')
    records = json.loads(args.manifest.read_text())
    for row in records:
        for side in ('reference', 'degraded'):
            if digest(Path(row[side])) != row[side + '_sha256']:
                raise ValueError('source media changed')
    rows = []
    for side, folder in [('degraded', 'input'), ('reference', 'reference')]:
        directory = args.out / folder
        directory.mkdir(parents=True)
        for sequence in records:
            count = len(range(0, sequence['frames'], args.stride))
            pattern = directory / (sequence['id'] + '-%03d.png')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', sequence[side],
                '-vf', f"select=not(mod(n\\,{args.stride}))", '-fps_mode', 'passthrough',
                '-frames:v', str(count), str(pattern)], check=True, timeout=120)
            for index in range(count):
                path = directory / f"{sequence['id']}-{index + 1:03d}.png"
                if not path.is_file():
                    raise RuntimeError('incomplete spatial export')
                rows.append({'file': str(path.relative_to(args.out)), 'sha256': digest(path),
                    'sequence_id': sequence['id'], 'source_id': sequence['source_id'],
                    'frame': index * args.stride, 'side': side})
    (args.out / 'manifest.json').write_text(json.dumps({
        'sequence_manifest_sha256': digest(args.manifest), 'stride': args.stride,
        'purpose': 'matched spatial development screen, full frame, no crop', 'frames': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
