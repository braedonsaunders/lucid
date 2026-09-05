#!/usr/bin/env python3
"""Fixed parameter interpolation for the measured quality/fidelity tradeoff.

No additional inference branches. Evaluate actual resulting networks: nonlinear
networks do not generally equal a weighted average of their output images.
"""
import argparse
import hashlib
import math
from pathlib import Path

import torch


def interpolate(anchor, candidate, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('finite interpolation weight in [0, 1] required')
    for key, default in [('channels', 32), ('frames', 1), ('scale', 4), ('version', 1)]:
        if anchor.get(key, default) != candidate.get(key, default):
            raise ValueError(f'architecture metadata mismatch: {key}')
    if anchor.get('scale') != 2:
        raise ValueError('requires explicit 2x checkpoints')
    a, b = anchor['model'], candidate['model']
    if a.keys() != b.keys():
        raise ValueError('model keys differ')
    result = {}
    for key, value in a.items():
        other = b[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'tensor schema mismatch: {key}')
        if value.is_floating_point():
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                raise ValueError(f'nonfinite parameter: {key}')
            result[key] = value.clone() if alpha == 0 else other.clone() if alpha == 1 else torch.lerp(value, other, alpha)
        elif torch.equal(value, other):
            result[key] = value.clone()
        else:
            raise ValueError(f'non-floating buffer differs: {key}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--anchor', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=[.25, .5, .75])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or len(set(args.weights)) != len(args.weights):
        ap.error('fresh output directory and distinct weights required')
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in (args.anchor, args.candidate)]
    states = [interpolate(a, b, weight) for weight in args.weights]
    args.out.mkdir(parents=True)
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for weight, state in zip(args.weights, states):
        metadata = {'anchor_sha256': digest(args.anchor), 'candidate_sha256': digest(args.candidate),
            'candidate_weight': weight, 'code_sha256': digest(__file__),
            'purpose': 'fixed development calibration; no reference-dependent weights or extra inference branches'}
        checkpoint = {k: v for k, v in a.items() if k not in ('model', 'experiment')}
        checkpoint.update(model=state, experiment=metadata, interpolation=metadata)
        path = args.out / f'blend-{weight:g}.pth'
        torch.save(checkpoint, path)
        print(path)


if __name__ == '__main__':
    main()
