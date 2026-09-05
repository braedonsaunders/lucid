#!/usr/bin/env python3
"""Locate detail errors with reference-only masks; never changes promotion gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate_sequences import digest, load, present_4x_at_2x, save
from score_checkpoint_frames import paired_references
from eval_checkpoint import bands, correlation


def blur_float(image):
    grid = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-grid*grid/2)
    kernel /= kernel.sum()
    horizontal = np.pad(image, ((0, 0), (3, 3)), mode='edge')
    horizontal = sum(kernel[i]*horizontal[:, i:i+image.shape[1]] for i in range(7))
    vertical = np.pad(horizontal, ((3, 3), (0, 0)), mode='edge')
    return sum(kernel[i]*vertical[i:i+image.shape[0], :] for i in range(7))


def region_metrics(output, reference):
    if output.shape != reference.shape or output.ndim != 2:
        raise ValueError('matching luma planes required')
    reference_blur, output_blur = blur_float(reference), blur_float(output)
    ry, rx = np.gradient(reference_blur)
    oy, ox = np.gradient(output_blur)
    gradient = np.hypot(rx, ry)
    fine_r, fine_o = reference-reference_blur, output-output_blur
    masks = {'all': np.ones(reference.shape, dtype=bool), 'low_gradient': gradient < 1,
             'middle_gradient': (gradient >= 1) & (gradient < 4), 'edges': gradient >= 4}
    result = {'legacy_fine_correlation': correlation(bands(output)[0], bands(reference)[0]), 'regions': {}}
    for label, mask in masks.items():
        count = int(mask.sum())
        row = {'pixels': count, 'coverage': float(mask.mean())}
        if count >= 100:
            reference_energy = float(np.sqrt(np.mean(fine_r[mask]**2)))
            denominator = np.sqrt(np.mean((rx[mask]**2+ry[mask]**2))*np.mean((ox[mask]**2+oy[mask]**2)))
            row.update(fine_correlation=correlation(fine_o[mask], fine_r[mask]),
                       fine_mae=float(np.mean(np.abs(fine_o[mask]-fine_r[mask]))),
                       fine_energy_ratio=float(np.sqrt(np.mean(fine_o[mask]**2))/reference_energy) if reference_energy > 1e-9 else None,
                       gradient_alignment=float(np.mean(ox[mask]*rx[mask]+oy[mask]*ry[mask])/denominator) if denominator > 1e-9 else None)
        result['regions'][label] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--checkpoint', nargs=2, action='append', required=True)
    parser.add_argument('--present-4x-at-2x', nargs='+', default=[])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh diagnostic output required')
    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels) or not set(args.present_4x_at_2x) <= set(labels):
        parser.error('distinct labels and registered output adapters required')
    manifest = json.loads((args.frames/'manifest.json').read_text())
    references = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames/row['file']) != row['sha256']:
            raise ValueError('frozen frame changed')
    models = {label: load(path, args.device) for label, path in args.checkpoint}
    if any(frames != 1 for _, _, frames in models.values()):
        raise ValueError('single-frame models required')
    report = {'purpose': 'detail-region diagnostic; existing quality gates remain unchanged',
              'manifest_sha256': digest(args.frames/'manifest.json'), 'code_sha256': digest(__file__),
              'checkpoint_sha256': {label: digest(path) for label, path in args.checkpoint},
              'blur': 'float64 separable sigma-1 Gaussian, radius 3, edge padding; distinct from legacy uint8 PIL blur',
              'regions': 'reference blurred-luma gradient <1, 1..4, >=4 levels per output pixel; not semantic or noise labels',
              'rows': [], 'complete': False}
    with torch.inference_mode():
        for row in manifest['frames']:
            if row['side'] != 'degraded':
                continue
            source = Image.open(args.frames/row['file']).convert('RGB')
            ref = references[row['sequence_id'], row['frame']]
            reference = Image.open(args.frames/ref['file']).convert('L')
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(args.device).float()/255
            for label, (model, _, _) in models.items():
                output = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                output = Image.fromarray(np.rint(output*255).astype(np.uint8))
                if label in args.present_4x_at_2x:
                    output = present_4x_at_2x(output, source.size)
                result = region_metrics(np.asarray(output.convert('L'), dtype=np.float64), np.asarray(reference, dtype=np.float64))
                report['rows'].append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                                       'frame': row['frame'], 'variant': label, **result})
            save(args.out, report)
            print(row['file'], flush=True)
    report['complete'] = True
    save(args.out, report)


if __name__ == '__main__':
    main()
