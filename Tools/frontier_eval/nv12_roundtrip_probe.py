#!/usr/bin/env python3
"""Score torch outputs before and after an RGB -> NV12 (4:2:0) -> RGB round trip.

The native sender packs model output as NV12. A checkpoint whose synthesized
detail is chromatic loses it in 4:2:0 packing; this measures how much per model.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import load  # noqa: E402
from evaluate_sequences import Scorer  # noqa: E402


def roundtrip(rgb):
    h, w = rgb.shape[:2]
    common = ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-i', '-']
    nv12 = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', *common, '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-pix_fmt', 'nv12', '-f', 'rawvideo', '-'],
                          input=rgb.tobytes(), capture_output=True, check=True).stdout
    back = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'nv12', '-s', f'{w}x{h}', '-color_range', 'tv', '-colorspace', 'bt709', '-i', '-',
                           '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], input=nv12, capture_output=True, check=True).stdout
    return np.frombuffer(back, np.uint8).reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--checkpoint', nargs=2, action='append', required=True)
    ap.add_argument('--every', type=int, default=6)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    manifest = json.loads((args.frames / 'manifest.json').read_text())
    refs = {(r['sequence_id'], r['frame']): r for r in manifest['frames'] if r['side'] == 'reference'}
    inputs = [r for r in manifest['frames'] if r['side'] == 'degraded' and r['source_id'] in args.source][::args.every]
    device = torch.device(args.device)
    scorer = Scorer(device)
    import collections
    agg = collections.defaultdict(list)
    for label, checkpoint in args.checkpoint:
        model = load(checkpoint, device)[0]
        for row in inputs:
            source = Image.open(args.frames / row['file']).convert('RGB')
            reference = Image.open(args.frames / refs[row['sequence_id'], row['frame']]['file']).convert('RGB')
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float() / 255
            with torch.inference_mode():
                out = np.rint(model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            direct = scorer.spatial(Image.fromarray(out), reference)
            packed = scorer.spatial(Image.fromarray(roundtrip(out)), reference)
            ref_packed = scorer.spatial(Image.fromarray(roundtrip(np.asarray(reference))), reference)
            agg[(label, row['source_id'])].append((direct, packed, ref_packed))
    for (l, s), rs in sorted(agg.items()):
        d = np.mean([r[0]['lpips'] for r in rs]); p = np.mean([r[1]['lpips'] for r in rs]); rp = np.mean([r[2]['lpips'] for r in rs])
        dd = np.mean([r[0]['dists'] for r in rs]); pd = np.mean([r[1]['dists'] for r in rs])
        print(f"SUMMARY {l:8s} {s:15s} LPIPS direct {d:.4f} -> packed {p:.4f} ({100*(p/d-1):+.1f}%) | DISTS {dd:.4f} -> {pd:.4f} | reference itself packed LPIPS {rp:.4f}")


if __name__ == '__main__':
    main()
