#!/usr/bin/env python3
"""Measure a Core ML deployment prototype; untrained weights imply no quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from PIL import Image
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail import CausalDetail


class ImageOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, state, valid):
        image, next_state = self.model(frame, state, valid)
        return image * 255, next_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=['640x360', '1280x720'])
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--out', type=Path, default=Path('.build/causal-detail'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260904)
    model = CausalDetail(args.channels, args.blocks, args.scale).eval()
    # A zero head can be constant-folded away. Use nonzero probe weights so the
    # entire architecture is measured. These weights are not a trained model.
    torch.nn.init.normal_(model.head.weight, std=0.002)
    wrapper = ImageOutput(model).eval()
    report = {'untrained': True, 'purpose': 'Core ML graph correctness and latency feasibility only',
        'channels': args.channels, 'blocks': args.blocks, 'scale': args.scale,
        'parameters': sum(p.numel() for p in model.parameters()),
        'coremltools': str(ct.__version__), 'torch': str(torch.__version__),
        'architecture_sha256': hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('architectures/causal_detail.py').read_bytes()).hexdigest(),
        'includes': 'synchronous Python Core ML prediction with PIL image and explicit feature-state copies',
        'excludes': 'video decode, browser transport, app rendering; no trained quality result', 'rows': []}
    for size in args.sizes:
        width, height = map(int, size.split('x'))
        pixels = np.random.default_rng(71).integers(0, 256, (height, width, 3), dtype=np.uint8)
        frame = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float() / 255
        initial = model.initial_state(frame)
        valid = torch.ones(1, 1, 1, 1)
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, (frame, initial, valid))
            reference, state_reference = wrapper(frame, initial, valid)
        path = args.out / f'causal_ch{args.channels}_x{args.scale}_{size}.mlpackage'
        converted = ct.convert(traced, inputs=[
            ct.ImageType(name='input', shape=frame.shape, scale=1/255, color_layout=ct.colorlayout.RGB),
            ct.TensorType(name='history_features', shape=initial.shape),
            ct.TensorType(name='valid', shape=valid.shape)],
            outputs=[ct.ImageType(name='output', color_layout=ct.colorlayout.RGB), ct.TensorType(name='next_state')],
            convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15)
        converted.save(str(path))
        for units in (ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.ALL):
            native = ct.models.MLModel(str(path), compute_units=units)
            inputs = {'input': Image.fromarray(pixels), 'history_features': initial.numpy(), 'valid': valid.numpy()}
            first = native.predict(inputs)
            expected = reference[0].permute(1, 2, 0).numpy().clip(0, 255)
            actual = np.asarray(first['output'].convert('RGB'), dtype=np.float32)
            max_error = float(np.max(np.abs(actual-expected)))
            state_error = float(np.max(np.abs(first['next_state']-state_reference.numpy())))
            if max_error > 2 or state_error > 0.02:
                raise RuntimeError(f'conversion mismatch: image={max_error}, state={state_error}')
            samples = []
            for i in range(15):
                started = time.perf_counter()
                result = native.predict(inputs)
                inputs['history_features'] = result['next_state']
                elapsed = (time.perf_counter()-started)*1000
                if i >= 5:
                    samples.append(elapsed)
            row = {'input': size, 'output': f'{width*args.scale}x{height*args.scale}',
                'compute_units': units.name, 'mean_ms': float(np.mean(samples)),
                'p95_ms': float(np.percentile(samples, 95)), 'samples_ms': samples,
                'conversion_max_pixel_error_255': max_error,
                'conversion_max_state_error': state_error,
                'state_bytes_fp32': initial.numel()*4}
            report['rows'].append(row)
            (args.out/'profile.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
