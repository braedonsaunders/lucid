#!/usr/bin/env python3
"""Crop full-frame teacher outputs into the trainer's teacher cache for a bank."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--windows', type=Path, required=True, help='windows.json from --dump-full-frames')
    ap.add_argument('--teacher-frames', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', default='NVIDIA VFX SDK VideoSuperRes ULTRA, 2x, full-frame; outputs cropped to the bank patches')
    args = ap.parse_args()
    if args.out.exists(): ap.error('fresh output required')
    manifest = json.loads((args.bank / 'manifest.json').read_text())
    bank_hash = digest(args.bank / 'manifest.json')
    windows = json.loads(args.windows.read_text())
    train_ids = {s['id'] for s in manifest['sequences'] if s['split'] == 'train'}
    args.out.mkdir(parents=True)
    rows = []
    for window in windows['windows']:
        frames = None
        for patch in window['patches']:
            if patch['id'] not in train_ids: continue
            if frames is None:
                frames = np.stack([np.asarray(Image.open(args.teacher_frames / window['window'] / f'{i:03d}.png').convert('RGB'))
                                   for i in range(window['frames'])])
                if frames.shape[1:3] != (2 * window['lr_size'][1], 2 * window['lr_size'][0]):
                    raise ValueError(f"{window['window']}: teacher frame size differs from 2x LR")
            x, y, side = patch['lr_patch_xy'][0], patch['lr_patch_xy'][1], patch['patch']
            teacher = frames[:, 2 * y:2 * (y + side), 2 * x:2 * (x + side)].copy()
            path = args.out / f"{patch['id']}.npz"
            np.savez_compressed(path, teacher=teacher)
            rows.append({'id': patch['id'], 'file': path.name, 'sha256': digest(path)})
    if {r['id'] for r in rows} != train_ids:
        raise ValueError('teacher cache does not cover every training sequence')
    (args.out / 'manifest.json').write_text(json.dumps({'bank_sha256': bank_hash, 'sequences': rows, 'split': 'train only',
        'inference': args.label, 'reference_used_for_prediction': False}, indent=2) + '\n')
    print('teacher cache:', len(rows), 'sequences')


if __name__ == '__main__':
    main()
