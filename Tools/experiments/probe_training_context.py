#!/usr/bin/env python3
"""Measure crop-context mismatch on training sources, without optimizing weights."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from train_causal_detail import load_bank, digest
from fold_shipping_head import fold_head
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from architectures.subspace_adapter import fuse_convolutions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'checkpoint', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): ap.error('fresh report required')
    if digest(args.checkpoint) != 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65':
        raise ValueError('shipping checkpoint required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, data = load_bank(args.bank)
    if digest(args.bank/'manifest.json') != 'b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0':
        raise ValueError('fixed training bank required')
    model, _, frames = load(args.checkpoint, 'cuda')
    if frames != 1: raise ValueError('single-frame model required')
    model = fold_head(model)
    fuse_convolutions(model)
    model.eval()
    index = {s['id']: s for s in manifest['sequences']}
    selected = {}
    for item in sorted(data['train'], key=lambda x: x[2]):
        selected.setdefault(index[item[2]]['source_id'], item)
    rows = []
    with torch.inference_mode():
        for source, (lr, hr, identity) in sorted(selected.items()):
            pixels = lr[0]
            h, w = pixels.shape[:2]
            y, x = ((h-96)//4)*2, ((w-96)//4)*2
            if min(h, w) < 128: raise ValueError('full-context source too small')
            tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].cuda().float()/255
            full = model(tensor)[:, :, y*2:(y+96)*2, x*2:(x+96)*2]
            cropped = model(tensor[:, :, y:y+96, x:x+96].contiguous())
            reference = torch.from_numpy(hr[0, y*2:(y+96)*2, x*2:(x+96)*2].copy()).permute(2,0,1)[None].cuda().float()/255
            margins = {}
            for margin in (8, 32, 64, 80):
                a, b, r = [v[:, :, margin:-margin, margin:-margin] for v in (full, cropped, reference)]
                delta = (a-b).abs()
                error = (a-r).abs().mean()
                margins[str(margin)] = dict(mean_rgb_levels=float(delta.mean()*255),
                    max_rgb_levels=float(delta.max()*255),
                    context_mae_over_reference_mae=float(delta.mean()/error.clamp_min(1e-9)),
                    full_reference_mae_rgb_levels=float(error*255))
            rows.append(dict(source_id=source, sequence_id=identity, frame=0,
                             full_size=[w,h], crop=[x,y,96,96], margins=margins))
            print(source, margins['8'], flush=True)
    report = dict(complete=True, purpose='Training-only context diagnostic; no quality promotion or training',
        selection='Lexicographically first training sequence per source, first frame, even-aligned center crop96',
        bank_sha256=digest(args.bank/'manifest.json'), checkpoint_sha256=digest(args.checkpoint),
        code_sha256=digest(__file__), torch=str(torch.__version__), device=torch.cuda.get_device_name(),
        precision='FP32, TF32 disabled; same area-folded frozen model in both contexts', rows=rows,
        source_balanced={m:{k:sum(row['margins'][m][k] for row in rows)/len(rows)
                           for k in rows[0]['margins'][m]} for m in ('8','32','64','80')})
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
