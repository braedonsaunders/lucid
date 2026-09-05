#!/usr/bin/env python3
"""Test a fixed spatial frequency split before training another detail branch."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from evaluate_sequences import Scorer, digest, load, present_4x_at_2x, save, summarize
from score_checkpoint_frames import paired_references


def split_residual(base, candidate):
    """Fixed sigma-1, radius-3 Gaussian residual; no reference or temporal input."""
    if base.shape != candidate.shape or base.ndim != 4 or base.shape[1] != 3:
        raise ValueError('matching BCHW RGB reconstructions required')
    grid = torch.arange(-3, 4, dtype=base.dtype, device=base.device)
    kernel = torch.exp(-grid.square()/2)
    kernel = kernel/kernel.sum()
    residual = candidate-base
    residual = F.conv2d(F.pad(residual, (3,3,0,0), mode='replicate'), kernel.reshape(1,1,1,7).expand(3,1,1,7), groups=3)
    residual = F.conv2d(F.pad(residual, (0,0,3,3), mode='replicate'), kernel.reshape(1,1,7,1).expand(3,1,7,1), groups=3)
    return base+residual


def tensor(image, device):
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].to(device).float()/255


def rgb8(value):
    pixels = value.clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    return Image.fromarray(np.rint(pixels*255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--shipping', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.out.exists(): ap.error('fresh diagnostic output required')
    torch.set_num_threads(4)
    manifest = json.loads((args.frames/'manifest.json').read_text())
    references = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames/row['file']) != row['sha256']: raise ValueError('frozen frame changed')
    shipping, _, frames = load(args.shipping, args.device)
    candidate, _, candidate_frames = load(args.candidate, args.device)
    if frames != 1 or candidate_frames != 1: raise ValueError('independent single-frame checkpoints required')
    scorer = Scorer(args.device)
    report = {'purpose': 'fixed spatial decomposition diagnostic; two-model cost is not a deployment proposal',
              'manifest_sha256': digest(args.frames/'manifest.json'),
              'checkpoint_sha256': {'shipping': digest(args.shipping), 'candidate': digest(args.candidate)},
              'code_sha256': digest(__file__), 'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
              'operation': 'RGB8 shipping 4x bicubic at 2x + sigma-1 radius-3 separable Gaussian(candidate RGB8 - shipping); replicate padding; RGB8 round/clamp',
              'per_image_reference_conditioning': False, 'protocol_selected_on_development_data': True, 'torch': str(torch.__version__), 'device': args.device,
              'split': 'development only; repeated model-selection sources', 'rows': [], 'complete': False}
    with torch.inference_mode():
        for row in manifest['frames']:
            if row['side'] != 'degraded': continue
            source = Image.open(args.frames/row['file']).convert('RGB')
            reference = Image.open(args.frames/references[row['sequence_id'],row['frame']]['file']).convert('RGB')
            x = tensor(source, args.device)
            base = present_4x_at_2x(rgb8(shipping(x)), source.size)
            detail = rgb8(candidate(x))
            if base.size != detail.size or detail.size != reference.size: raise ValueError('genuine matched 2x outputs required')
            output = rgb8(split_residual(tensor(base,args.device), tensor(detail,args.device)))
            for label, image in [('shipping',base), ('candidate',detail), ('frequency_split',output)]:
                report['rows'].append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                                      'frame': row['frame'], 'variant': label, 'metrics': scorer.spatial(image,reference)})
            report['summary'] = summarize(report['rows'])
            save(args.out,report)
            print(row['file'],flush=True)
    report['complete'] = True
    save(args.out,report)


if __name__ == '__main__': main()
