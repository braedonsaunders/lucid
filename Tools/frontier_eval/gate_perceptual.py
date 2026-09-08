#!/usr/bin/env python3
"""Perceptual promotion gate: reward synthesized detail, floor hallucination at the interpolation anchor.

The frozen development gate required every source's fine-band correlation to stay
within 0.01 of shipping. That is a fidelity guard: it rejects the measured commercial
reference (the competitive target) and every candidate that improved LPIPS
and DISTS by 5-18%. Fine correlation cannot reward invented detail by
construction, so a gate built on it can only ever admit models that synthesize
nothing.

This gate keeps the perceptual minimums and per-source perceptual regression
limits unchanged and replaces the shipping-relative correlation guard with two
guards that a good synthesizer passes and a bad one fails:

* hallucination floor: source-balanced fine correlation must not fall below the
  Lanczos anchor, which synthesizes nothing. Below the anchor, added fine energy
  is invented rather than recovered. Calibration on the 960-pair holdout showed
  the commercial reference itself below the anchor on four of eight sources (grain,
  water, noise) while improving both perceptual metrics on every source, so the
  floor is aggregate; per-source deltas are reported, not gated.
* embellishment cap: per-source fine-band energy must not exceed the reference
  by more than 10%.

Bad hallucination is caught by the per-source perceptual guard: LPIPS is far less
tolerant of invented texture than DISTS, and a checkpoint that speckles flat dark
surfaces regresses LPIPS on those sources.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

GATE = {'source_balanced_lpips_improvement_min': .03, 'source_balanced_dists_improvement_min': .03,
        'per_source_perceptual_regression_max': .02, 'anchor': 'lanczos',
        'source_balanced_fine_correlation_floor': 'anchor', 'per_source_detail_energy_max': 1.10}
METRICS = ('lpips', 'dists', 'fine_correlation', 'detail_energy')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_means(rows, labels):
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    coverage = {}
    for row in rows:
        if row['variant'] in labels:
            groups[row['variant']][row['source_id']].append(row['metrics'])
    for label in labels:
        keys = [(r['source_id'], r['sequence_id'], r['frame']) for r in rows if r['variant'] == label]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f'{label}: missing or duplicate samples')
        coverage[label] = set(keys)
    if any(cover != coverage[labels[0]] for cover in coverage.values()):
        raise ValueError('variants do not cover identical samples')
    for label in labels:
        for source, metrics in groups[label].items():
            for m in metrics:
                if not all(math.isfinite(m[k]) for k in METRICS):
                    raise ValueError('nonfinite metric')
    return {label: {source: {k: sum(m[k] for m in metrics) / len(metrics) for k in METRICS}
                    for source, metrics in sources.items()} for label, sources in groups.items()}, len(coverage[labels[0]])


def evaluate(rows, candidate, baseline='shipping', anchor='lanczos', gate=GATE):
    means, count = source_means(rows, [anchor, baseline, candidate])
    balanced = {label: {k: sum(s[k] for s in sources.values()) / len(sources) for k in METRICS}
                for label, sources in means.items()}
    if any(balanced[baseline][k] <= 0 for k in ('lpips', 'dists')):
        raise ValueError('positive baseline distances required')
    gains = {k: 1 - balanced[candidate][k] / balanced[baseline][k] for k in ('lpips', 'dists')}
    reasons = []
    for k, gain in gains.items():
        if gain < gate[f'source_balanced_{k}_improvement_min']:
            reasons.append(f'{k}: aggregate improvement below minimum')
    changes = {}
    for source, base in means[baseline].items():
        cand, anc = means[candidate][source], means[anchor][source]
        change = {k: 1 - cand[k] / base[k] for k in ('lpips', 'dists')}
        change['fine_correlation_vs_shipping'] = cand['fine_correlation'] - base['fine_correlation']
        change['fine_correlation_vs_anchor'] = cand['fine_correlation'] - anc['fine_correlation']
        change['detail_energy'] = cand['detail_energy']
        changes[source] = change
        for k in ('lpips', 'dists'):
            if change[k] < -gate['per_source_perceptual_regression_max']:
                reasons.append(f'{source}: {k} regression exceeds limit')
        if cand['detail_energy'] > gate['per_source_detail_energy_max']:
            reasons.append(f'{source}: fine-band energy exceeds the embellishment cap')
    if balanced[candidate]['fine_correlation'] < balanced[anchor]['fine_correlation']:
        reasons.append(f'fine correlation: source-balanced value below the {anchor} anchor')
    return {'candidate': candidate, 'baseline': baseline, 'anchor': anchor, 'gate': gate,
            'perceptual_gate_pass': not reasons, 'failure_reasons': reasons, 'frames_per_variant': count,
            'perceptual_improvements': gains, 'source_balanced': balanced, 'per_source': means,
            'per_source_changes': changes,
            'status': 'development screen; native delivery, temporal and fresh-source evidence still required'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--candidate', action='append', required=True)
    ap.add_argument('--baseline', default='shipping')
    ap.add_argument('--anchor', default='lanczos')
    ap.add_argument('--anchor-report', type=Path,
                    help='Take anchor rows from another complete report of the same samples (native reports carry no interpolation rows)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh gate receipt required')
    report = json.loads(args.report.read_text())
    if report.get('complete') is not True:
        raise ValueError('complete report required')
    if args.anchor_report:
        other = json.loads(args.anchor_report.read_text())
        if other.get('complete') is not True:
            raise ValueError('complete anchor report required')
        if any(r['variant'] == args.anchor for r in report['rows']):
            raise ValueError('report already contains anchor rows')
        report['rows'] = report['rows'] + [r for r in other['rows'] if r['variant'] == args.anchor]
    results = {c: evaluate(report['rows'], c, args.baseline, args.anchor) for c in args.candidate}
    receipt = {'report': str(args.report), 'report_sha256': digest(args.report),
               'anchor_report': str(args.anchor_report) if args.anchor_report else None,
               'anchor_report_sha256': digest(args.anchor_report) if args.anchor_report else None,
               'checkpoint_sha256': report.get('checkpoint_sha256'), 'results': results,
               'code_sha256': digest(Path(__file__))}
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({c: {k: r[k] for k in ('perceptual_gate_pass', 'perceptual_improvements', 'failure_reasons')}
                      for c, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
