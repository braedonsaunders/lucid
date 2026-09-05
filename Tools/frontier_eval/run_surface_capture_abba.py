#!/usr/bin/env python3
"""Run a paired installed-extension transport experiment with shipping weights."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

from analyze_browser_trace import analyze, percentile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['app', 'fixture', 'config', 'control', 'candidate', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--samples', type=int, default=60)
    args = parser.parse_args()
    if args.out.exists() or args.samples < 30:
        parser.error('fresh output and at least 30 samples per arm required')
    args.out.mkdir(parents=True)
    result = {'purpose': 'short installed transport ABBA; not a sustained release gate', 'runs': [], 'complete': False}
    pooled = {label: {'draws': 0, 'seconds': 0, 'latencies': []} for label in ['control', 'candidate']}
    for index, label in enumerate(['control', 'candidate', 'candidate', 'control']):
        out = args.out/f'{index}-{label}'
        load_before = list(os.getloadavg())
        subprocess.run([sys.executable, str(Path(__file__).with_name('run_browser_candidate.py')),
            '--headed', '--app', str(args.app), '--fixture', str(args.fixture), '--config', str(args.config),
            '--extension', str(getattr(args, label)), '--presentation-trace', '--samples', str(args.samples),
            '--order', 'shipping', '--out', str(out)], check=True)
        analysis = analyze(out/'report.json', minimum_seconds=args.samples)
        report = json.loads((out/'report.json').read_text())
        measurement = report['runs'][0]['measurement']
        if label == 'candidate' and not all('surface-transfer' in row['frameStats'] for row in measurement['samples']):
            raise ValueError('candidate silently fell back to service-worker capture')
        trace = json.loads(gzip.decompress((out/measurement['presentation_trace']['file']).read_bytes()))
        latencies = [row['latencyMilliseconds'] for row in trace['surfaces'][0]['samples']
                     if measurement['startedEpoch'] <= row['at'] < measurement['endedEpoch']]
        metric = analysis['runs'][0]
        pooled[label]['draws'] += metric['actual_draws']
        pooled[label]['seconds'] += metric['seconds']
        pooled[label]['latencies'].extend(latencies)
        result['runs'].append({'label': label, 'analysis': analysis, 'load_before': load_before,
                               'load_after': list(os.getloadavg())})
        (args.out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    result['pooled'] = {label: {'actual_fps': row['draws']/row['seconds'], 'seconds': row['seconds'],
        'draws': row['draws'], 'p95_ms': percentile(row['latencies'], .95),
        'p99_ms': percentile(row['latencies'], .99)} for label, row in pooled.items()}
    result['complete'] = True
    (args.out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['pooled'], flush=True)


if __name__ == '__main__':
    main()
