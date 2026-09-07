#!/usr/bin/env python3
"""Core ML package vs torch checkpoint on real frames: does conversion change the score on grainy sources?"""
import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import load  # noqa: E402
from evaluate_sequences import Scorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--pair', nargs=3, action='append', required=True, metavar=('LABEL', 'CHECKPOINT', 'MLPACKAGE'))
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.frames / 'manifest.json').read_text())
    refs = {(r['sequence_id'], r['frame']): r for r in manifest['frames'] if r['side'] == 'reference'}
    inputs = [r for r in manifest['frames'] if r['side'] == 'degraded' and r['source_id'] in args.source][::args.every]
    device = torch.device(args.device)
    scorer = Scorer(device)
    rows = []
    for label, checkpoint, package in args.pair:
        model = load(checkpoint, device)[0]
        for units in ('CPU_AND_GPU', 'ALL'):
            native = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
            for row in inputs:
                source = Image.open(args.frames / row['file']).convert('RGB')
                reference = Image.open(args.frames / refs[row['sequence_id'], row['frame']]['file']).convert('RGB')
                x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float() / 255
                with torch.inference_mode():
                    t = np.rint(model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                c = np.asarray(native.predict({'input': source})['output'].convert('RGB'))
                diff = np.abs(c.astype(np.int16) - t.astype(np.int16))
                rows.append({'label': label, 'units': units, 'source_id': row['source_id'], 'sequence_id': row['sequence_id'], 'frame': row['frame'],
                             'torch': scorer.spatial(Image.fromarray(t), reference), 'coreml': scorer.spatial(Image.fromarray(c), reference),
                             'max_abs': int(diff.max()), 'mean_abs': float(diff.mean())})
            print(label, units, len(inputs), 'frames', flush=True)
    args.out.write_text(json.dumps({'rows': rows}, indent=1) + '\n')
    import collections
    agg = collections.defaultdict(list)
    for r in rows: agg[(r['label'], r['units'], r['source_id'])].append(r)
    for (l, u, s), rs in sorted(agg.items()):
        print(f"SUMMARY {l:8s} {u:12s} {s:15s} torch LPIPS {np.mean([r['torch']['lpips'] for r in rs]):.4f} coreml {np.mean([r['coreml']['lpips'] for r in rs]):.4f} | DISTS {np.mean([r['torch']['dists'] for r in rs]):.4f} vs {np.mean([r['coreml']['dists'] for r in rs]):.4f} | max|d| {max(r['max_abs'] for r in rs)} mean|d| {np.mean([r['mean_abs'] for r in rs]):.3f}")


if __name__ == '__main__':
    main()
