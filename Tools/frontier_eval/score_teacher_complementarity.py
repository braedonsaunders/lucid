#!/usr/bin/env python3
"""Fixed output combinations: test teacher/shipping complementarity before training.

Weights and frequency split are fixed for the whole development set. No reference
pixels influence reconstruction. This does not measure deployment performance.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter
import torch

from evaluate_sequences import Scorer, digest, load, present_4x_at_2x, save, summarize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from fold_shipping_head import fold_head


def rgb(array):
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--teacher', type=Path, required=True)
    ap.add_argument('--shipping', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.frames / 'manifest.json').read_text())
    for row in manifest['frames']:
        if digest(args.frames / row['file']) != row['sha256']:
            raise ValueError('input changed')
    device = torch.device('mps')
    model, step, history = load(args.shipping, device)
    if history != 1:
        raise ValueError('spatial comparator requires single-frame shipping weights')
    folded = fold_head(model)
    scorer = Scorer(device)
    report = {'purpose': 'spatial development complementarity; no release claim',
        'manifest_sha256': digest(args.frames / 'manifest.json'),
        'checkpoint_sha256': digest(args.shipping), 'code_sha256': digest(Path(__file__)),
        'fold_sha256': digest(Path(__file__).resolve().parents[1] / 'experiments/fold_shipping_head.py'),
        'weights': [0.25, 0.5, 0.75], 'frequency_split': 'PIL GaussianBlur radius 1.2',
        'reference_used_for_reconstruction': False, 'teacher_hashes': {},
        'rows': [], 'complete': False}
    for row in manifest['frames']:
        if row['side'] != 'degraded':
            continue
        name = Path(row['file']).name
        with Image.open(args.frames / row['file']) as image:
            source = image.convert('RGB')
        with Image.open(args.teacher / name) as image:
            teacher = image.convert('RGB')
        report['teacher_hashes'][name] = digest(args.teacher / name)
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float() / 255
            output = model(x)[0].permute(1, 2, 0).cpu().numpy() * 255
            shipping = present_4x_at_2x(rgb(output), source.size)
            direct = rgb(folded(x)[0].permute(1, 2, 0).cpu().numpy() * 255)
        if shipping.size != teacher.size:
            raise ValueError('teacher/shipping shape mismatch')
        s, t = (np.asarray(i, dtype=np.float32) for i in (shipping, teacher))
        low_s, low_t = (np.asarray(i.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
                        for i in (shipping, teacher))
        variants = {'shipping': shipping, 'teacher': teacher, 'direct2x_area': direct,
                    'teacher_low_shipping_high': rgb(low_t + s - low_s)}
        for weight in report['weights']:
            variants[f'teacher_mix_{weight}'] = rgb(s * (1-weight) + t * weight)
        # References are opened only after every reconstructed variant is fixed.
        with Image.open(args.frames / 'reference' / name) as image:
            reference = image.convert('RGB')
        for label, result in variants.items():
            report['rows'].append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                'frame': row['frame'], 'variant': label, 'metrics': scorer.spatial(result, reference)})
        report['summary'] = summarize(report['rows'])
        save(args.report, report)
        print(name, flush=True)
    report['complete'] = True
    save(args.report, report)


if __name__ == '__main__':
    main()
