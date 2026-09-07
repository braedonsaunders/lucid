#!/usr/bin/env python3
"""Widen a trained Unshuffled SPAN checkpoint while preserving its function exactly.

Net2Net-style: the first (new - old) feature channels are duplicated. Duplicated
outgoing columns carry half the original weight plus an antisymmetric
perturbation, so every downstream sum is unchanged at initialization while the
two copies receive different gradients and can specialise. Multiplicative
attention in SPAB is per channel, so duplicate channels stay identical through
it. Capacity experiment on top of pretrained weights; not a new architecture.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.span_arch import Conv3XC  # noqa: E402
from eval_checkpoint import load  # noqa: E402
from train_span import Unshuffled  # noqa: E402


def duplication(old, new):
    if new < old:
        raise ValueError('can only widen')
    return list(range(old)) + list(range(new - old))


def widen(model, channels, seed=20260912, noise=.02):
    source_channels = model.core.conv_1.eval_conv.out_channels
    if channels <= source_channels or channels % 2:
        raise ValueError('target width must exceed the source width and be even')
    target = Unshuffled(channels, frames=model.frames, scale=model.core.upsampler[1].upscale_factor // 2,
                        version=model.version).eval()
    generator = torch.Generator().manual_seed(seed)
    c, c2 = source_channels, channels
    sources = dict(model.named_modules())
    with torch.no_grad():
        for name, layer in target.named_modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            src = sources[name]
            w = src.weight.data
            out_old, in_old = w.shape[:2]
            out_new, in_new = layer.weight.shape[:2]
            rows = duplication(out_old, out_new) if out_new != out_old else list(range(out_old))
            grown = w[rows].clone()
            bias = src.bias.data[rows].clone() if src.bias is not None else None
            if in_new != in_old:
                if in_old == 4 * c and in_new == 4 * c2:
                    cols, halve = [], []
                    inner = duplication(c, c2)
                    for group in range(4):
                        cols += [group * c + i for i in inner]
                        halve += [i >= c for i in range(c2)]
                else:
                    inner = duplication(in_old, in_new)
                    cols = inner
                    halve = [i >= in_old for i in range(in_new)]
                columns = grown[:, cols].clone()
                # Pair each duplicate with its original and split the weight antisymmetrically.
                seen = {}
                for j, (col, dup) in enumerate(zip(cols, halve)):
                    key = (col, j // len(inner) if in_old == 4 * c else 0)
                    if dup:
                        original = seen[key]
                        base = columns[:, original].clone()
                        perturbation = torch.randn(base.shape, generator=generator) * base.abs().mean() * noise
                        columns[:, original] = base * .5 + perturbation
                        columns[:, j] = base * .5 - perturbation
                    else:
                        seen[key] = j
                grown = columns
            layer.weight.copy_(grown)
            if bias is not None:
                layer.bias.copy_(bias)
        for module in target.modules():
            if isinstance(module, Conv3XC):
                module.update_params()
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260912)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    model, step, frames = load(args.checkpoint, 'cpu')
    if frames != 1 or model.core.upsampler[1].upscale_factor != 4:
        raise ValueError('single-frame direct-2x checkpoint required')
    widened = widen(model, args.channels, args.seed)
    torch.manual_seed(0)
    x = torch.rand(2, 3, 48, 64)
    with torch.no_grad():
        a, b = model.eval()(x), widened.eval()(x)
    error = float((a - b).abs().max())
    if error > 1e-3:
        raise ValueError(f'widening changed the function: max abs error {error}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torch.save({'model': widened.state_dict(), 'channels': args.channels, 'scale': 2, 'frames': 1,
                'version': widened.version, 'step': step, 'architecture': 'shipping_direct2x_area',
                'initialization': f'function-preserving widening of {args.checkpoint.name} from {model.core.conv_1.eval_conv.out_channels} to {args.channels} channels; max abs error {error:.2e}',
                'source_experiment': state.get('experiment')}, args.out)
    print(args.out, f'{sum(p.numel() for p in widened.parameters())} parameters, max abs error {error:.2e}')


if __name__ == '__main__':
    main()
