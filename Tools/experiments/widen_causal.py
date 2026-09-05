#!/usr/bin/env python3
"""Double causal-v2 width while preserving the initial reconstruction function.

Duplicated feature channels get complementary outgoing weights. Their sum is
unchanged, but their gradients can diverge, allowing the added capacity to learn.
This is function-preserving model expansion, not a new architecture claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail_v2 import CausalDetailV2


def widen(model, seed=20260911):
    c = model.channels
    expanded = CausalDetailV2(c * 2, len(model.blocks), model.scale).eval()
    generator = torch.Generator().manual_seed(seed)
    features = list(range(c)) * 2
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, target in expanded.named_modules():
            if not isinstance(target, torch.nn.Conv2d):
                continue
            source = modules[name]
            out = features if target.out_channels == source.out_channels * 2 else list(range(source.out_channels))
            if name.endswith('.project'):
                out = features + [i + c for i in features]
            if name == 'gate':
                inp = features + [i + c for i in features]
            elif name == 'head':
                inp = features + list(range(c, source.in_channels))
            elif target.groups == target.in_channels:
                inp = [0]
            elif target.in_channels == source.in_channels * 2:
                inp = features
            else:
                inp = list(range(source.weight.shape[1]))
            rows = source.weight[out].clone()
            weights = rows[:, inp].clone()
            for old in sorted(set(inp)):
                destinations = [i for i, value in enumerate(inp) if value == old]
                if len(destinations) == 2:
                    noise = torch.randn(rows[:, old].shape, generator=generator) * source.weight.std() * .05
                    weights[:, destinations[0]] = rows[:, old] * .5 + noise
                    weights[:, destinations[1]] = rows[:, old] * .5 - noise
            target.weight.copy_(weights)
            if target.bias is not None:
                target.bias.copy_(source.bias[out])
    return expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['architecture'] != 'causal_detail_v2':
        parser.error('only causal v2 checkpoints are supported')
    model = CausalDetailV2(source['channels'], source['blocks'], source['scale']).eval()
    model.load_state_dict(source['model'], strict=True)
    expanded = widen(model)
    # Validate the trained function, not just tensor shapes, including reset.
    torch.manual_seed(19)
    x = torch.rand(1, 3, 32, 48)
    a, b = model.initial_state(x), expanded.initial_state(x)
    errors = []
    with torch.inference_mode():
        for valid in [0, 1, 1, 0]:
            x = x.roll(1, -1)
            y, a = model(x, a, torch.full((1, 1, 1, 1), float(valid)))
            z, b = expanded(x, b, torch.full((1, 1, 1, 1), float(valid)))
            errors.append(float((y-z).abs().max()))
    if max(errors) > 1e-5:
        raise RuntimeError(f'expanded initialization changed the model: {errors}')
    receipt = {'source_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'method': '2x duplicated channels; complementary outgoing-weight perturbations',
        'seed': 20260911, 'maximum_output_error': max(errors),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result = {k: source[k] for k in ('architecture', 'blocks', 'scale', 'no_history')}
    result.update(channels=expanded.channels, parameters=sum(p.numel() for p in expanded.parameters()),
                  step=0, model=expanded.state_dict(), initialization=receipt)
    torch.save(result, args.out)
    args.out.with_suffix('.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'model'}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
