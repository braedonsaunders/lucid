#!/usr/bin/env python3
"""Uniform weight average of same-architecture 2x checkpoints (a model soup)."""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if len(args.checkpoint) < 2 or args.out.exists():
        ap.error('at least two checkpoints and a fresh output path required')
    states = [torch.load(p, map_location='cpu', weights_only=False) for p in args.checkpoint]
    first = states[0]
    for s in states[1:]:
        for key in ('channels', 'scale', 'frames', 'version'):
            if s.get(key, first.get(key)) != first.get(key):
                raise ValueError(f'architecture metadata differs: {key}')
        if s['model'].keys() != first['model'].keys():
            raise ValueError('model keys differ')
    soup = {}
    for key, value in first['model'].items():
        tensors = [s['model'][key] for s in states]
        if any(t.shape != value.shape or t.dtype != value.dtype for t in tensors):
            raise ValueError(f'tensor schema mismatch: {key}')
        if value.is_floating_point():
            soup[key] = torch.stack([t.float() for t in tensors]).mean(0).to(value.dtype)
        elif all(torch.equal(t, value) for t in tensors):
            soup[key] = value.clone()
        else:
            raise ValueError(f'non-floating buffer differs: {key}')
    torch.save({'model': soup, 'channels': first.get('channels', 32), 'scale': first.get('scale', 2), 'frames': first.get('frames', 1),
                'version': first.get('version', 1), 'step': first.get('step'), 'architecture': first.get('architecture'),
                'soup': [str(p) for p in args.checkpoint]}, args.out)
    print(args.out, 'uniform soup of', len(states), 'checkpoints')


if __name__ == '__main__':
    main()
