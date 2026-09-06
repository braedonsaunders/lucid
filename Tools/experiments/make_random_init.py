#!/usr/bin/env python3
"""Write a randomly initialized direct-2x Unshuffled checkpoint in the trainer's format.

For capacity experiments that cannot start from the folded shipping weights.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_span import Unshuffled  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.channels < 8 or args.channels > 96 or args.out.exists():
        ap.error('channels in [8, 96] and a fresh output path required')
    torch.manual_seed(args.seed)
    model = Unshuffled(args.channels, scale=2).eval()
    torch.save({'model': model.state_dict(), 'channels': args.channels, 'scale': 2, 'frames': 1,
                'version': model.version, 'step': 0, 'architecture': 'shipping_direct2x_area',
                'initialization': f'random, seed {args.seed}'}, args.out)
    print(args.out, sum(p.numel() for p in model.parameters()), 'parameters')


if __name__ == '__main__':
    main()
