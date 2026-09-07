#!/usr/bin/env python3
"""Does a checkpoint's score depend on how 4:2:0 chroma is upsampled before it?

The torch holdout decodes yuv420p sources with FFmpeg's default chroma
interpolation; the app converts NV12 with VideoToolbox. Re-decode a subset with
several swscale chroma filters and score checkpoints on each. A checkpoint whose
grain-source score moves with the filter is exposed to the native path.
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

FILTERS = {'bicubic': 'bicubic', 'bilinear': 'bilinear', 'neighbor': 'neighbor', 'lanczos': 'lanczos', 'full_chroma_int': 'bicubic+full_chroma_int+accurate_rnd'}


def decode(path, frames, flags, width, height):
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vf', f"select='not(mod(n\\,{frames}))',scale={width}:{height}:flags={flags}",
           '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    count = len(raw) // (width * height * 3)
    return np.frombuffer(raw[:count * width * height * 3], np.uint8).reshape(count, height, width, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequences', type=Path, required=True)
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--checkpoint', nargs=2, action='append', required=True)
    ap.add_argument('--stride', type=int, default=30)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    models = {label: load(path, device)[0] for label, path in args.checkpoint}
    scorer = Scorer(device)
    rows = []
    for seq in json.loads(args.sequences.read_text()):
        if seq['source_id'] not in args.source:
            continue
        w, h = seq['width'], seq['height']
        reference = decode(seq['reference'], args.stride, 'bicubic', 2 * w, 2 * h)
        for name, flags in FILTERS.items():
            degraded = decode(seq['degraded'], args.stride, flags, w, h)
            n = min(len(reference), len(degraded))
            for i in range(n):
                ref = Image.fromarray(reference[i])
                x = torch.from_numpy(degraded[i].copy()).permute(2, 0, 1)[None].to(device).float() / 255
                with torch.inference_mode():
                    for label, model in models.items():
                        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                        out = Image.fromarray(np.rint(y * 255).astype(np.uint8))
                        rows.append({'source_id': seq['source_id'], 'sequence_id': seq['id'], 'frame': i * args.stride,
                                     'chroma_filter': name, 'variant': label, 'metrics': scorer.spatial(out, ref)})
            print(seq['id'], name, n, 'frames', flush=True)
    args.out.write_text(json.dumps({'purpose': 'chroma upsampling sensitivity probe; torch level', 'filters': FILTERS,
                                    'stride': args.stride, 'rows': rows}, indent=1) + '\n')
    # Summary: per model × filter, mean LPIPS/DISTS per source
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['variant'], r['chroma_filter'], r['source_id'])].append(r['metrics'])
    for (v, f, s), ms in sorted(agg.items()):
        print(f"{v:14s} {f:16s} {s:16s} LPIPS {sum(m['lpips'] for m in ms)/len(ms):.4f} DISTS {sum(m['dists'] for m in ms)/len(ms):.4f} energy {sum(m['detail_energy'] for m in ms)/len(ms):.3f}")


if __name__ == '__main__':
    main()
