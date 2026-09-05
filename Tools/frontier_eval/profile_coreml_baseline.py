#!/usr/bin/env python3
"""Measure a published Core ML baseline without importing its application code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import coremltools as ct
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh output required')
    receipt = json.loads(args.receipt.read_text())
    package_files = {}
    for path in args.model.rglob('*'):
        if path.is_file():
            relative = str(path.relative_to(args.model.parent))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if receipt['files'][relative]['sha256'] != digest:
                raise ValueError('published package changed')
            package_files[relative] = digest
    report = {'purpose': 'published baseline graph cost; not end-to-end playback or quality',
              'provenance': receipt, 'files': package_files, 'platform': platform.platform(),
              'coremltools': ct.__version__, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'compute_note': 'requested units allow CPU fallback; this does not prove ANE-only execution',
              'rows': [], 'complete': False}
    units = ['CPU_AND_NE', 'CPU_AND_GPU', 'ALL']
    models = {unit: ct.models.MLModel(str(args.model), compute_units=getattr(ct.ComputeUnit, unit)) for unit in units}
    spec = models[units[0]].get_spec()
    if len(spec.description.input) != 1 or len(spec.description.output) != 1:
        raise ValueError('one input and output required')
    input_name, output_name = spec.description.input[0].name, spec.description.output[0].name
    for width, height in [(640, 360), (854, 480), (1280, 720)]:
        rng = np.random.default_rng(20260905)
        inputs = [rng.random((1, 3, height, width), dtype=np.float32) for _ in range(4)]
        times = {unit: [] for unit in units}
        order_rng = random.Random(20260905)
        ranges = {}
        load = []
        for iteration in range(50):
            order = list(units)
            order_rng.shuffle(order)
            for unit in order:
                start = time.perf_counter()
                output = np.asarray(models[unit].predict({input_name: inputs[iteration % 4]})[output_name])
                elapsed = (time.perf_counter()-start)*1000
                if output.shape != (1, 3, height*2, width*2) or not np.isfinite(output).all():
                    raise ValueError('invalid baseline output geometry or values')
                if iteration >= 10:
                    times[unit].append(elapsed)
                if iteration == 0:
                    ranges[unit] = [float(output.min()), float(output.max())]
            if iteration % 10 == 0:
                load.append({'iteration': iteration, 'load_average': list(os.getloadavg())})
        row = {'input_size': [width, height], 'output_size': [width*2, height*2], 'output_ranges': ranges,
               'system_load': load, 'timings': {unit: {'mean_ms': float(np.mean(values)),
                    'p95_ms': float(np.percentile(values, 95)), 'samples_ms': values} for unit, values in times.items()}}
        report['rows'].append(row)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print([width, height], {u: round(v['mean_ms'], 2) for u, v in row['timings'].items()}, flush=True)
    report['complete'] = True
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
