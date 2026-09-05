#!/usr/bin/env python3
"""Reproduce the frozen local-fidelity experiment's development gates, never promotion."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools/frontier_eval'))
from gate_frozen_holdout import compare_rows

SHIPPING = 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
MANIFESTS = {
    'development': ('aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f', 48),
    'bank': ('eb8ad3792cc1999f4dc593494f959656be28b815ee21d3a9094c47b683f3a7fe', 96),
}
GATE = {
    'source_balanced_lpips_improvement_min': .03,
    'source_balanced_dists_improvement_min': .03,
    'per_source_perceptual_regression_max': .02,
    'per_source_fine_correlation_drop_max': .01,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['source_id'], row['sequence_id'], row['frame']


def score(report, control, mode, checkpoints):
    manifest, count = MANIFESTS[mode]
    for value in (report, control):
        if value.get('complete') is not True or value['split'] != 'development-validation':
            raise ValueError('complete development report required')
        if value['manifest_sha256'] != manifest or value['presentation_adapters'] != ['shipping']:
            raise ValueError('frozen pixels or presentation adapter changed')
        if value['checkpoint_sha256']['shipping'] != SHIPPING:
            raise ValueError('shipping weights changed')
    if report['sequence_manifest_sha256'] != control['sequence_manifest_sha256']:
        raise ValueError('source manifest changed')
    if report['checkpoint_sha256'] != dict(checkpoints, shipping=SHIPPING):
        raise ValueError('evaluated candidate differs from downloaded final checkpoint')
    if {r['variant'] for r in report['rows']} != {'unconstrained', 'constrained', 'shipping', 'lanczos'}:
        raise ValueError('unexpected variants')
    prior = [r for r in control['rows'] if r['variant'] == 'shipping']
    expected = {key(r) for r in prior}
    if len(prior) != count or len(expected) != count:
        raise ValueError('control sample coverage changed')
    max_error = 0.
    for variant in ('shipping', 'lanczos'):
        old = {key(r): r['metrics'] for r in control['rows'] if r['variant'] == variant}
        new = {key(r): r['metrics'] for r in report['rows'] if r['variant'] == variant}
        if set(old) != expected or set(new) != expected:
            raise ValueError('control rows changed')
        for sample in expected:
            if set(old[sample]) != set(new[sample]):
                raise ValueError('control metrics changed')
            for metric, value in old[sample].items():
                delta = abs(value-new[sample][metric])
                if not math.isfinite(delta) or delta > 1e-5:
                    raise ValueError(f'{variant} control failed to reproduce: {metric}')
                max_error = max(max_error, delta)
    results = {label: compare_rows(report['rows'], expected, GATE,
        candidate=label, other_labels=('lanczos',)) for label in ('unconstrained', 'constrained')}
    for result in results.values():
        result['status'] = 'development screen only; no independent holdout or release admission'
    return {
        'mode': mode, 'gate': GATE, 'controls_max_absolute_delta': max_error,
        'results': results,
        'constrained_minus_unconstrained': {
            metric: results['constrained']['source_balanced']['constrained'][metric]
                - results['unconstrained']['source_balanced']['unconstrained'][metric]
            for metric in ('lpips', 'dists', 'fine_correlation')},
        'promotion_authorized': False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=MANIFESTS, required=True)
    ap.add_argument('--checkpoints', type=Path, required=True)
    args = ap.parse_args()
    directory = Path(__file__).resolve().parent
    report = directory / f'local-fidelity-{args.mode}.json'
    control = directory / f'anchored-lowpass-{args.mode}.json'
    paths = {label: args.checkpoints / label / 'step008000.pth' for label in ('unconstrained', 'constrained')}
    result = score(json.loads(report.read_text()), json.loads(control.read_text()),
        args.mode, {label: sha(path) for label, path in paths.items()})
    result['evidence_sha256'] = {str(path.relative_to(ROOT)): sha(path)
        for path in (report, control, Path(__file__).resolve(), ROOT/'Tools/frontier_eval/gate_frozen_holdout.py')}
    (directory/f'local-fidelity-{args.mode}-gate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({label: {k: value[k] for k in ('spatial_gate_pass', 'perceptual_improvements', 'failure_reasons')}
        for label, value in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
