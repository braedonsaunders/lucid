#!/usr/bin/env python3
"""Apply the frozen spatial-quality gate without selecting new model parameters."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(report, spec, frozen, manifest):
    if not report.get('complete') or report.get('split') != 'quality-holdout':
        raise ValueError('complete quality-holdout report required')
    if report['checkpoint_sha256'] != {'blend80': frozen['candidate_sha256'], 'shipping': frozen['shipping_sha256']}:
        raise ValueError('checkpoints differ from frozen candidate and shipping')
    if report['presentation_adapters'] != ['shipping']:
        raise ValueError('shipping requires the declared 4x-to-2x presentation adapter')
    if manifest['stride'] != spec['conditions']['spatial_stride'] or manifest['split'] != 'quality-holdout':
        raise ValueError('frame export differs from frozen split/stride')
    expected = set()
    for source in spec['sources']:
        for codec in spec['conditions']['codecs']:
            for bitrate in spec['conditions']['bitrates']:
                for frame in range(0, source['frames'], manifest['stride']):
                    expected.add((source['id'], f"{source['id']}-{codec}-{bitrate}", frame))
    for side in ['reference', 'degraded']:
        rows = [r for r in manifest['frames'] if r['side'] == side]
        keys = [(r['source_id'], r['sequence_id'], r['frame']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('frame export has missing, duplicate or unexpected samples')
    return compare_rows(report['rows'], expected, spec['gate'])


def compare_rows(rows, expected, gate, candidate='blend80', baseline='shipping', other_labels=('lanczos',)):
    report_rows = rows
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for label in [*other_labels, baseline, candidate]:
        rows = [r for r in report_rows if r['variant'] == label]
        keys = [(r['source_id'], r['sequence_id'], r['frame']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('report has missing, duplicate or unexpected samples')
        for row in rows:
            metrics = row['metrics']
            if not all(math.isfinite(metrics[key]) for key in ['lpips', 'dists', 'fine_correlation']):
                raise ValueError('nonfinite quality score')
            groups[label][row['source_id']].append(metrics)
    means = {label: {source: {key: sum(r[key] for r in rows)/len(rows)
        for key in ['lpips', 'dists', 'fine_correlation']} for source, rows in sources.items()}
        for label, sources in groups.items()}
    balanced = {label: {key: sum(r[key] for r in sources.values())/len(sources)
        for key in ['lpips', 'dists', 'fine_correlation']} for label, sources in means.items()}
    if any(row[key] <= 0 for row in means[baseline].values() for key in ['lpips', 'dists']):
        raise ValueError('positive baseline distances required')
    gains = {key: 1-balanced[candidate][key]/balanced[baseline][key] for key in ['lpips', 'dists']}
    reasons = []
    for key, gain in gains.items():
        if gain < gate[f'source_balanced_{key}_improvement_min']:
            reasons.append(f'{key}: aggregate improvement below minimum')
    deltas = {}
    for source, values in means[baseline].items():
        delta = {key: 1-means[candidate][source][key]/values[key] for key in ['lpips', 'dists']}
        delta['fine_correlation'] = means[candidate][source]['fine_correlation']-values['fine_correlation']
        deltas[source] = delta
        for key in ['lpips', 'dists']:
            if delta[key] < -gate['per_source_perceptual_regression_max']:
                reasons.append(f'{source}: {key} regression exceeds limit')
        if delta['fine_correlation'] < -gate['per_source_fine_correlation_drop_max']:
            reasons.append(f'{source}: fine correlation drop exceeds limit')
    return {'spatial_gate_pass': not reasons, 'failure_reasons': reasons,
        'frames_per_variant': len(expected), 'source_balanced': balanced, 'per_source': means,
        'perceptual_improvements': gains, 'per_source_changes': deltas,
        'status': 'spatial holdout only; native/browser quality and release requirements remain'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['report', 'spec', 'frozen', 'manifest', 'out']:
        ap.add_argument('--'+key, type=Path, required=True)
    args = ap.parse_args()
    report, spec, frozen, manifest = [json.loads(p.read_text()) for p in [args.report,args.spec,args.frozen,args.manifest]]
    if digest(args.spec) != frozen['holdout_spec_sha256'] or digest(args.manifest) != report['manifest_sha256']:
        raise ValueError('spec or manifest bytes changed')
    if manifest['sequence_manifest_sha256'] != report['sequence_manifest_sha256']:
        raise ValueError('sequence manifest changed')
    result = evaluate(report, spec, frozen, manifest)
    result['evidence_sha256'] = {str(p): digest(p) for p in [args.report,args.spec,args.frozen,args.manifest,Path(__file__)]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['spatial_gate_pass','failure_reasons','perceptual_improvements']},indent=2))


if __name__ == '__main__':
    main()
