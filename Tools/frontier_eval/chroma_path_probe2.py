#!/usr/bin/env python3
"""NV12 → RGB with explicit chroma upsamplers (nearest / bilinear / bicubic), then score checkpoints.

Emulates the range of converters a native pipeline may use. A checkpoint whose
grain-source score moves with the upsampler is exposed to the native path.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import load  # noqa: E402
from evaluate_sequences import Scorer  # noqa: E402


def decode_nv12(path, stride, width, height):
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vf', f"select='not(mod(n\\,{stride}))'",
           '-fps_mode', 'passthrough', '-pix_fmt', 'nv12', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    frame_bytes = width * height * 3 // 2
    count = len(raw) // frame_bytes
    frames = []
    for i in range(count):
        chunk = np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], np.uint8)
        y = chunk[:width * height].reshape(height, width).astype(np.float32)
        uv = chunk[width * height:].reshape(height // 2, width // 2, 2).astype(np.float32)
        frames.append((y, uv))
    return frames


def to_rgb(y, uv, mode):
    # Chroma upsampling with left-sited (MPEG-2/H.264) horizontal alignment approximated by
    # align_corners=False bilinear/bicubic and nearest; all are candidates for what a converter does.
    t = torch.from_numpy(uv).permute(2, 0, 1)[None]
    up = F.interpolate(t, size=y.shape, mode={'nearest': 'nearest', 'bilinear': 'bilinear', 'bicubic': 'bicubic'}[mode],
                       **({} if mode == 'nearest' else {'align_corners': False}))[0].permute(1, 2, 0).numpy()
    yy = (y - 16.0) * (255.0 / 219.0)
    cb = (up[..., 0] - 128.0) * (255.0 / 224.0)
    cr = (up[..., 1] - 128.0) * (255.0 / 224.0)
    r = yy + 1.5748 * cr
    g = yy - 0.1873 * cb - 0.4681 * cr
    b = yy + 1.8556 * cb
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def decode_rgb(path, stride, width, height):
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vf', f"select='not(mod(n\\,{stride}))'",
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
        reference = decode_rgb(seq['reference'], args.stride, 2 * w, 2 * h)
        ffmpeg_rgb = decode_rgb(seq['degraded'], args.stride, w, h)
        nv12 = decode_nv12(seq['degraded'], args.stride, w, h)
        n = min(len(reference), len(ffmpeg_rgb), len(nv12))
        variants = {'ffmpeg_rgb': [ffmpeg_rgb[i] for i in range(n)]}
        for mode in ('nearest', 'bilinear', 'bicubic'):
            variants[mode] = [to_rgb(*nv12[i], mode) for i in range(n)]
        for name, frames in variants.items():
            for i in range(n):
                ref = Image.fromarray(reference[i])
                x = torch.from_numpy(frames[i].copy()).permute(2, 0, 1)[None].to(device).float() / 255
                with torch.inference_mode():
                    for label, model in models.items():
                        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                        rows.append({'source_id': seq['source_id'], 'sequence_id': seq['id'], 'frame': i * args.stride,
                                     'chroma': name, 'variant': label, 'metrics': scorer.spatial(Image.fromarray(np.rint(y * 255).astype(np.uint8)), ref)})
        print(seq['id'], n, 'frames', flush=True)
    args.out.write_text(json.dumps({'purpose': 'explicit NV12 chroma upsampler sensitivity; torch level', 'rows': rows}, indent=1) + '\n')
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['variant'], r['chroma'], r['source_id'])].append(r['metrics'])
    for (v, f, s), ms in sorted(agg.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        print(f"SUMMARY {v:8s} {f:11s} {s:15s} LPIPS {sum(m['lpips'] for m in ms)/len(ms):.4f} DISTS {sum(m['dists'] for m in ms)/len(ms):.4f} energy {sum(m['detail_energy'] for m in ms)/len(ms):.3f}")


if __name__ == '__main__':
    main()
