"""Measure reparameterized training/evaluation forward differences on fixed crops.

No optimization or quality claims: this isolates arithmetic and fusion effects.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from train_causal_detail import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh receipt required')
    torch.set_num_threads(2)
    model, _, frames = load(args.checkpoint, 'cuda')
    if frames != 1:
        raise ValueError('single-frame checkpoint required')
    manifest = json.loads((args.frames/'manifest.json').read_text())
    rows = [r for r in manifest['frames'] if r['side'] == 'degraded']
    variants = [(mode, dtype) for mode in ('train', 'eval') for dtype in
                (torch.float32, torch.bfloat16, torch.float16)]
    measurements = []
    with torch.no_grad():
        for row in rows:
            path = args.frames/row['file']
            if digest(path) != row['sha256']:
                raise ValueError('fixed development input changed')
            image = np.array(Image.open(path).convert('RGB'))
            h, w = image.shape[:2]
            pixels = image[h//2-64:h//2+64, w//2-64:w//2+64].copy()
            x = torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda().float()/255
            outputs = {}
            for mode, dtype in variants:
                model.train(mode == 'train')
                with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                    outputs[mode+'_'+str(dtype).split('.')[-1]] = model(x).float()[...,16:-16,16:-16]
            baseline = outputs['eval_float32']
            scores = {}
            for name, value in outputs.items():
                delta = (value-baseline).abs()*255
                rounded = (value.clamp(0, 1)*255).round()-(baseline.clamp(0, 1)*255).round()
                scores[name] = {'mean_rgb': float(delta.mean()), 'max_rgb': float(delta.max()),
                                'rounded_rgb_rmse': float(rounded.square().mean().sqrt())}
            for dtype in ('bfloat16', 'float16'):
                delta = (outputs['train_'+dtype]-outputs['eval_'+dtype]).abs()*255
                scores['train_vs_eval_'+dtype] = {'mean_rgb': float(delta.mean()), 'max_rgb': float(delta.max())}
            measurements.append({'source_id': row['source_id'], 'file': row['file'], 'scores': scores})
    summary = {name: {key: float(np.mean([r['scores'][name][key] for r in measurements]))
                     for key in measurements[0]['scores'][name]} for name in measurements[0]['scores']}
    result = {'complete': True, 'scope': '128px center crops from fixed development inputs; forward arithmetic only, not native quality or training ablation',
              'checkpoint_sha256': digest(args.checkpoint), 'manifest_sha256': digest(args.frames/'manifest.json'),
              'code_sha256': digest(__file__), 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
              'reference': 'fused evaluation graph FP32, 16 output-pixel border excluded', 'summary': summary, 'rows': measurements}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
