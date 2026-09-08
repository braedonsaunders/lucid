#!/usr/bin/env python3
"""Score checkpoints on frozen decoded PNG pixels, independent of FFmpeg version."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate_sequences import Scorer, digest, load, present_4x_at_2x, save, summarize


def flip_ensemble(model, source):
    """Static-content four-orientation reference, with four model evaluations.

    Quantize each unflipped prediction to RGB8 before averaging, as an actual
    image-output model would. This is an idealized static-content mechanism probe for temporal
    cycling, not a claim of native frame accumulation or unchanged latency.
    """
    outputs = []
    for axes in ((), (-1,), (-2,), (-2, -1)):
        x = source.flip(axes) if axes else source
        y = model(x)
        if axes:
            y = y.flip(axes)
        outputs.append((y.clamp(0, 1) * 255).round() / 255)
    return torch.stack(outputs).mean(0)


def paired_references(rows):
    sides = {'reference': {}, 'degraded': {}}
    for row in rows:
        if row['side'] not in sides:
            raise ValueError('unknown frame role')
        key = row['sequence_id'], row['frame']
        side = sides[row['side']]
        if key in side:
            raise ValueError('duplicate sequence frame')
        side[key] = row
    if not sides['reference'] or sides['reference'].keys() != sides['degraded'].keys():
        raise ValueError('complete nonempty reference/degraded pairing required')
    for key, row in sides['reference'].items():
        if row['source_id'] != sides['degraded'][key]['source_id']:
            raise ValueError('paired source identities differ')
    return sides['reference']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--checkpoint', nargs=2, action='append', required=True)
    ap.add_argument('--present-4x-at-2x', nargs='+', default=[])
    ap.add_argument('--flip-ensemble', nargs='+', default=[],
                    help='Labels to evaluate with an actual four-pass static RGB8 flip ensemble')
    ap.add_argument('--interpolation', nargs='+', choices=('lanczos', 'bicubic', 'bilinear'),
                    default=['lanczos'], help='Explicit pixel interpolation comparators; not browser rendering')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    labels = [label for label, _ in args.checkpoint]
    if (len(labels) != len(set(labels)) or set(labels) & {'lanczos', 'bicubic', 'bilinear'}
            or len(args.interpolation) != len(set(args.interpolation))
            or not set(args.present_4x_at_2x) <= set(labels)
            or not set(args.flip_ensemble) <= set(labels)):
        ap.error('distinct model/interpolation labels and registered presentation adapters required')
    manifest_path = args.frames / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    reference_index = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames / row['file']) != row['sha256']:
            raise ValueError('frozen decoded frame changed')
    device = torch.device(args.device)
    models = {label: load(path, device) for label, path in args.checkpoint}
    if any(frames != 1 for _, _, frames in models.values()):
        raise ValueError('this spatial scorer requires independent single-frame models')
    scorer = Scorer(device)
    split = manifest.get('split', 'development-validation')
    report = {'purpose': f'fixed decoded-pixel spatial {split}; no temporal or release claim', 'split': split,
        'manifest_sha256': digest(manifest_path), 'sequence_manifest_sha256': manifest['sequence_manifest_sha256'],
        'checkpoint_sha256': {label: digest(path) for label, path in args.checkpoint},
        'presentation_adapters': args.present_4x_at_2x,
        'flip_ensemble': args.flip_ensemble,
        'flip_ensemble_contract': 'four model evaluations per image, unflip each RGB8 prediction, average; static spatial probe only',
        'interpolation': args.interpolation,
        'code_sha256': digest(__file__), 'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
        'torch': str(torch.__version__), 'device': str(device), 'rows': [], 'complete': False}
    for row in manifest['frames']:
        if row['side'] != 'degraded':
            continue
        with Image.open(args.frames / row['file']) as image:
            source = image.convert('RGB')
        reference_row = reference_index[row['sequence_id'], row['frame']]
        with Image.open(args.frames / reference_row['file']) as image:
            reference = image.convert('RGB')
        results = {label: source.resize(reference.size, getattr(Image.Resampling, label.upper()))
                   for label in args.interpolation}
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float()/255
            for label, (model, _, _) in models.items():
                prediction = flip_ensemble(model, x) if label in args.flip_ensemble else model(x)
                y = prediction.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                output = Image.fromarray(np.rint(y * 255).astype(np.uint8))
                if label in args.present_4x_at_2x:
                    output = present_4x_at_2x(output, source.size)
                results[label] = output
        for label, output in results.items():
            report['rows'].append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                'frame': row['frame'], 'variant': label, 'metrics': scorer.spatial(output, reference)})
        report['summary'] = summarize(report['rows'])
        save(args.report, report)
        print(row['file'], flush=True)
    report['complete'] = True
    save(args.report, report)


if __name__ == '__main__':
    main()
