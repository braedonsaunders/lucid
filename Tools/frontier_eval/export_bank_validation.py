#!/usr/bin/env python3
"""Freeze paired PNGs from declared validation sequences without reading training media."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from build_causal_bank import digest, validate_sources


def export(bank, out, stride=4):
    bank, out = Path(bank), Path(out)
    if out.exists() or stride < 1:
        raise ValueError('fresh output and positive stride required')
    path = bank / 'manifest.json'
    manifest = json.loads(path.read_text())
    validate_sources(manifest['sources'])
    if manifest['scale'] != 2:
        raise ValueError('genuine 2x bank required')
    sources = {s['id']: s for s in manifest['sources']}
    sequences, seen = [], set()
    for row in manifest['sequences']:
        source = sources[row['source_id']]
        if row['id'] in seen or any(row[k] != source[k] for k in ('split', 'family')) or row['source_sha256'] != source['sha256']:
            raise ValueError('sequence identity or split differs from source')
        seen.add(row['id'])
        if row['split'] == 'validation':
            if '/' in row['id'] or '\\' in row['id'] or '..' in row['id']:
                raise ValueError('invalid sequence filename')
            sequences.append(row)
    if not sequences:
        raise ValueError('validation sequences required')
    out.mkdir(parents=True)
    for side in ('reference', 'degraded'):
        (out / side).mkdir()
    frames = []
    for row in sequences:
        path = bank / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('validation sequence bytes changed')
        with np.load(path, allow_pickle=False) as data:
            lr, hr = data['lr'], data['hr']
            n, size = manifest['frames'], manifest['lr_patch']
            if lr.dtype != np.uint8 or hr.dtype != np.uint8 or lr.shape != (n, size, size, 3) or hr.shape != (n, size*2, size*2, 3):
                raise ValueError('invalid paired frame geometry or dtype')
            for index in range(0, n, stride):
                for side, pixels in [('reference', hr[index]), ('degraded', lr[index])]:
                    file = out / side / f'{row["id"]}-{index:03d}.png'
                    Image.fromarray(pixels).save(file)
                    frames.append({'source_id': row['source_id'], 'sequence_id': row['id'], 'frame': index,
                                   'side': side, 'file': str(file.relative_to(out)), 'sha256': digest(file)})
    result = {'split': 'development-validation', 'stride': stride,
              'sequence_manifest_sha256': digest(bank / 'manifest.json'), 'code_sha256': digest(__file__),
              'purpose': 'fixed declared validation patches; no training media, resizing or new degradation',
              'sources': [s for s in manifest['sources'] if s['split'] == 'validation'],
              'sequences': sequences, 'frames': frames}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=4)
    args = ap.parse_args()
    result = export(args.bank, args.out, args.stride)
    print(json.dumps({'sources': len(result['sources']), 'pairs': len(result['frames']) // 2}), flush=True)


if __name__ == '__main__': main()
