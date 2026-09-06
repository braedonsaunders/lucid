#!/usr/bin/env python3
"""Apply the existing frozen development checks to both prespecified outputs."""
import argparse
import copy
import json
from pathlib import Path

from gate_controller_candidate import evaluate, digest, SHIPPING


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('development', 'bank'), required=True)
    for name in ('report', 'control', 'raw', 'blend', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh gate receipt required')
    report, control = [json.loads(p.read_text()) for p in (args.report, args.control)]
    checkpoints = {'paired_raw': digest(args.raw), 'paired80': digest(args.blend), 'shipping': SHIPPING}
    if report['checkpoint_sha256'] != checkpoints:
        raise ValueError('reported final checkpoints differ from supplied files')
    if {r['variant'] for r in report['rows']} != {*checkpoints, 'lanczos'}:
        raise ValueError('unexpected or missing output variants')
    for field in ('device', 'torch', 'scorer_sha256'):
        if report[field] != control[field]:
            raise ValueError('comparison backend/scorer changed: '+field)
    results = {}
    for candidate in ('paired_raw', 'paired80'):
        # Reuse the tested complete-coverage, frozen-control and threshold logic.
        # Its historical internal candidate name is 'coarse'; no pixels change.
        selected = copy.deepcopy(report)
        selected['checkpoint_sha256'] = {'coarse': checkpoints[candidate], 'shipping': SHIPPING}
        selected['rows'] = [dict(r, variant='coarse' if r['variant'] == candidate else r['variant'])
                            for r in report['rows'] if r['variant'] in (candidate, 'shipping', 'lanczos')]
        result = evaluate(selected, control, args.mode, checkpoints[candidate])
        for field in ('source_balanced', 'per_source'):
            result[field][candidate] = result[field].pop('coarse')
        result['status'] = 'Repeated spatial development screen; independent native/release evidence required'
        results[candidate] = result
    paths = (args.report, args.control, args.raw, args.blend, Path(__file__),
             Path(__file__).with_name('gate_controller_candidate.py'),
             Path(__file__).with_name('gate_frozen_holdout.py'))
    receipt = {'mode': args.mode, 'results': results,
               'evidence_sha256': {str(p): digest(p) for p in paths}}
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {key: value[key] for key in
        ('spatial_gate_pass', 'perceptual_improvements', 'failure_reasons')}
        for name, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
