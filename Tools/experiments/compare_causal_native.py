#!/usr/bin/env python3
"""Interleave equivalent native graphs to reduce timing-order bias."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import coremltools as ct
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', nargs=2, action='append', required=True, metavar=('LABEL', 'PACKAGE'))
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 30 or len({r[0] for r in args.model}) != len(args.model):
        parser.error('unique labels and >=30 samples required')
    models, states, profiles, histories = {}, {}, {}, {}
    width = height = 0
    for label, package in args.model:
        profile = json.loads(Path(package).with_name('profile.json').read_text())
        if not profile['complete']:
            raise ValueError('graph must pass conversion checks first')
        profiles[label] = profile
        model = ct.models.MLModel(package, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        models[label] = model
        description = model.get_spec().description
        image = next(x for x in description.input if x.name == 'input').type.imageType
        if width and (width, height) != (image.width, image.height):
            raise ValueError('different graph sizes')
        width, height = image.width, image.height
        if profile.get('stateful'):
            states[label] = model.make_state()
        else:
            shape = next(x for x in description.input if x.name == 'history_features').type.multiArrayType.shape
            histories[label] = np.zeros(tuple(shape), dtype=np.float32)
    if any(not p['checkpoint_sha256'] for p in profiles.values()) or len({p['checkpoint_sha256'] for p in profiles.values()}) != 1:
        raise ValueError('different checkpoint weights')
    rng = np.random.default_rng(913)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    images = [Image.fromarray(np.roll(pixels, i, axis=1)) for i in range(4)]
    samples = {label: [] for label in models}
    max_error = {label: 0 for label in models}
    order = list(models)
    reference_label = order[0]
    orders = []
    for index in range(args.samples + 10):
        # Rotate order deterministically: every graph gets each position.
        current = order[index % len(order):] + order[:index % len(order)]
        results = {}
        for label in current:
            inputs = {'input': images[index % len(images)],
                      'valid': np.full((1, 1, 1, 1), float(index > 0), dtype=np.float32)}
            if label in histories:
                inputs['history_features'] = histories[label]
            started = time.perf_counter()
            result = models[label].predict(inputs, state=states[label]) if label in states else models[label].predict(inputs)
            if label in histories:
                histories[label] = result['next_state']
            elapsed = (time.perf_counter() - started) * 1000
            if index >= 10:
                samples[label].append(elapsed)
            results[label] = result['output']
        if index >= 10:
            orders.append(current)
        if index in (0, 10, args.samples + 9):
            reference = np.asarray(results[reference_label], dtype=np.int16)
            for label in models:
                error = int(np.abs(np.asarray(results[label], dtype=np.int16) - reference).max())
                max_error[label] = max(max_error[label], error)
                if error > 2:
                    raise RuntimeError(f'non-equivalent native output: {label}: {error}')
    report = {'purpose': 'paired native graph timing, not app playback or quality promotion',
        'platform': platform.platform(), 'coremltools': str(ct.__version__),
        'checkpoint_sha256': profiles[reference_label]['checkpoint_sha256'],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'size': [width, height], 'compute_units': 'CPU_AND_GPU', 'warmup_rounds': 10,
        'order': orders, 'max_pixel_error_255': max_error,
        'limitations': ['other host workloads remain active', 'synchronous Python image predictions; browser/decode/render excluded'],
        'rows': {label: {'mean_ms': float(np.mean(values)), 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'samples_ms': values,
            'package': str(Path(dict(args.model)[label]).resolve()),
            'stateful': profiles[label].get('stateful', False),
            'separable_floor': profiles[label].get('separable_floor', False)} for label, values in samples.items()}}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({label: {key: value for key, value in row.items() if key in ('mean_ms', 'p50_ms', 'p95_ms')}
                      for label, row in report['rows'].items()}))


if __name__ == '__main__':
    main()
