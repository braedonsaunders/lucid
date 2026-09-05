#!/usr/bin/env python3
"""Gate sustained installed-extension playback using actual draw acknowledgments."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def analyze(report_path, minimum_seconds=600, minimum_fps=57, maximum_p95_ms=50):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if report.get('complete') is not True or not report.get('runs'):
        raise ValueError('complete nonempty report required')
    results = []
    for run in report['runs']:
        measurement = run['measurement']
        start, end = measurement['startedEpoch'], measurement['endedEpoch']
        if not all(map(finite, [start, end])) or end <= start:
            raise ValueError('invalid measurement window')
        duration = (end - start) / 1000
        receipt = measurement['presentation_trace']
        if Path(receipt['file']).name != receipt['file']:
            raise ValueError('trace must be adjacent to report')
        raw = (report_path.parent / receipt['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('trace digest mismatch')
        trace = json.loads(gzip.decompress(raw))
        surfaces = trace['surfaces']
        installed_id = run['installation']['installed']['id']
        session = run['setup']['status']['activeSession']
        if len(surfaces) != 1 or surfaces[0]['url'] != f'chrome-extension://{installed_id}/surface.html#{session}':
            raise ValueError('unexpected presentation surface')
        records = surfaces[0]['samples']
        if not records or len(records) >= 250000:
            raise ValueError('empty or capacity-limited trace')
        previous_seq, previous_at = -1, -math.inf
        for row in records:
            seq, at, latency = row['seq'], row['at'], row['latencyMilliseconds']
            if (row['session'] != session or type(seq) is not int or seq <= previous_seq
                    or not finite(at) or at < previous_at or not finite(latency) or latency <= 0):
                raise ValueError('invalid, duplicate, reordered, or foreign acknowledgment')
            previous_seq, previous_at = seq, at
        if records[0]['at'] > start or records[-1]['at'] < end:
            raise ValueError('trace does not bracket the measurement window')
        selected = [row for row in records if start <= row['at'] < end]
        if not selected:
            raise ValueError('no draws in measurement window')
        latency = [row['latencyMilliseconds'] for row in selected]
        fps = len(selected) / duration
        p95 = percentile(latency, .95)
        off = run.get('off', {})
        gates = {'duration': duration >= minimum_seconds, 'cadence': fps >= minimum_fps,
                 'latency': p95 <= maximum_p95_ms,
                 'off': off.get('enabled') is False and off.get('enhancing') is False}
        bins = []
        for offset in range(0, math.ceil(duration), 60):
            lo, hi = start + offset * 1000, min(end, start + (offset + 60) * 1000)
            rows = [row for row in selected if lo <= row['at'] < hi]
            bins.append({'start_second': offset, 'seconds': (hi-lo)/1000,
                         'draws': len(rows), 'fps': len(rows)*1000/(hi-lo),
                         'p95_ms': percentile([row['latencyMilliseconds'] for row in rows], .95) if rows else None})
        memory = measurement.get('native_memory_samples', [])
        rss = [row['rss_kib'] / 1024 for row in memory]
        results.append({'variant': run['variant'], 'seconds': duration, 'actual_draws': len(selected),
                        'actual_fps': fps, 'median_ms': statistics.median(latency), 'p95_ms': p95,
                        'p99_ms': percentile(latency, .99), 'max_ms': max(latency),
                        'gates': gates, 'pass': all(gates.values()), 'minute_bins': bins,
                        'native_rss_mib': {'samples': len(rss), 'first': rss[0], 'last': rss[-1], 'peak': max(rss)} if rss else None})
    return {'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'metric': 'capture to canvas draw submission acknowledgment; not physical scanout',
            'limits': {'minimum_seconds': minimum_seconds, 'minimum_fps': minimum_fps, 'maximum_p95_ms': maximum_p95_ms},
            'runs': results, 'pass': all(row['pass'] for row in results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.report), indent=2) + '\n')
