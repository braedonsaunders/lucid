#!/usr/bin/env python3
"""Measure the pinned 2026 author release at actual video dimensions on Core ML."""
import argparse
import json
from pathlib import Path
import platform
import time

import coremltools as ct
import numpy as np
from PIL import Image
import torch

from plksr_rep_release import load_release, digest, PINNED, WEIGHTS


class ImageRange(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).clamp(0, 1) * 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'weights', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--compute-units', nargs='+', choices=['CPU_AND_GPU', 'CPU_AND_NE'],
                        default=['CPU_AND_GPU', 'CPU_AND_NE'])
    args = parser.parse_args()
    if args.out.exists() or min(args.width, args.height) <= 16 or args.samples < 20:
        parser.error('fresh output, dimensions above reflect padding, and >=20 samples required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(6)
    model = ImageRange(load_release(args.repository, args.weights, deploy=True)).eval()
    rng = np.random.default_rng(20260905)
    pictures = [Image.fromarray(rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8))
                for _ in range(4)]
    example = torch.from_numpy(np.asarray(pictures[0]).copy()).permute(2, 0, 1)[None].float()/255
    report = {'complete': False, 'weights_sha256': WEIGHTS, 'source_sha256': PINNED,
        'profiler_sha256': digest(__file__), 'loader_sha256': digest(Path(__file__).with_name('plksr_rep_release.py')),
        'platform': platform.platform(), 'torch': str(torch.__version__), 'coremltools': ct.__version__,
        'input_size': [args.width, args.height], 'output_scale': 4,
        'scope': 'warmed native prediction only; excludes capture, transport, downsampling and display',
        'precision': args.precision+'; published reflect16/crop64; author deployment fusion', 'rows': []}
    def save():
        (args.out/'profile.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    with torch.inference_mode():
        expected = model(example)[0].permute(1, 2, 0).numpy()
        traced = torch.jit.trace(model, example, check_trace=False)
    print('FP32 reference and trace ready', flush=True)
    converted = ct.convert(traced, inputs=[ct.ImageType(name='input', shape=example.shape,
        color_layout=ct.colorlayout.RGB, scale=1/255)],
        outputs=[ct.ImageType(name='output', color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT16 if args.precision == 'fp16' else ct.precision.FLOAT32, convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS15)
    path = args.out/'plksr_rep.mlpackage'
    converted.save(str(path))
    report['package_sha256'] = {str(p.relative_to(path)): digest(p) for p in path.rglob('*') if p.is_file()}
    for units in args.compute_units:
        native = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
        pixels = np.asarray(native.predict({'input': pictures[0]})['output'].convert('RGB')).astype(np.float32)
        if pixels.shape != expected.shape or not np.isfinite(pixels).all():
            raise ValueError('invalid native output')
        delta = np.abs(pixels-expected)
        row = {'compute_units': units, 'max_rgb_error': float(delta.max()),
               'mean_rgb_error': float(delta.mean()), 'samples_ms': []}
        row['correctness_pass'] = row['max_rgb_error'] <= 3 and row['mean_rgb_error'] <= .6
        report['rows'].append(row)
        save()
        if not row['correctness_pass']:
            print('Rejected precision', row, flush=True)
            continue
        for iteration in range(args.samples+10):
            start = time.perf_counter()
            native.predict({'input': pictures[iteration % 4]})
            elapsed = (time.perf_counter()-start)*1000
            if iteration >= 10:
                row['samples_ms'].append(elapsed)
        row['mean_ms'] = float(np.mean(row['samples_ms']))
        row['p95_ms'] = float(np.percentile(row['samples_ms'], 95))
        save()
        print(units, row['mean_ms'], row['p95_ms'], flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
