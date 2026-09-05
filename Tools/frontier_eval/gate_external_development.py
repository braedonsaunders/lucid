#!/usr/bin/env python3
"""Compare exported competitor frames with matched-backend shipping controls."""
import argparse
import json
import math
from pathlib import Path
from gate_frozen_holdout import compare_rows, digest

MANIFEST = 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'
SHIPPING = 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
GATE = {'source_balanced_lpips_improvement_min': .03, 'source_balanced_dists_improvement_min': .03,
        'per_source_perceptual_regression_max': .02, 'per_source_fine_correlation_drop_max': .01}


def score(candidate, shipping, outputs, label):
    if not all(report.get('complete') is True for report in (candidate, shipping, outputs)):
        raise ValueError('complete reports required')
    if (candidate['input_manifest_sha256'] != MANIFEST or shipping['manifest_sha256'] != MANIFEST
            or outputs['input_manifest_sha256'] != MANIFEST):
        raise ValueError('frozen development inputs changed')
    if shipping['checkpoint_sha256'] != {'shipping': SHIPPING} or shipping['presentation_adapters'] != ['shipping']:
        raise ValueError('shipping comparator changed')
    for field in ('device', 'torch', 'sequence_manifest_sha256', 'scorer_sha256'):
        if candidate[field] != shipping[field]:
            raise ValueError('scorer backend or inputs differ: '+field)
    hashes = {row['output_file']: row['output_sha256'] for row in outputs['rows']}
    if len(outputs['rows']) != 48 or len(hashes) != 48 or hashes != candidate['output_sha256']:
        raise ValueError('scored outputs differ from pinned export')
    def keyed(report, variant):
        rows = [r for r in report['rows'] if r['variant'] == variant]
        result = {(r['source_id'], r['sequence_id'], r['frame']): r['metrics'] for r in rows}
        if len(rows) != 48 or len(result) != 48:
            raise ValueError('missing or duplicate rows')
        return result
    old, new = keyed(shipping, 'lanczos'), keyed(candidate, 'lanczos')
    if old.keys() != new.keys():
        raise ValueError('control coverage differs')
    maximum = 0.
    for key in old:
        if old[key].keys() != new[key].keys():
            raise ValueError('control metrics differ')
        for metric in old[key]:
            delta = abs(old[key][metric]-new[key][metric])
            if not math.isfinite(delta) or delta > 1e-5:
                raise ValueError('interpolation control failed: '+metric)
            maximum = max(maximum, delta)
    rows = [r for r in candidate['rows'] if r['variant'] in (label, 'lanczos')]
    rows += [r for r in shipping['rows'] if r['variant'] == 'shipping']
    result = compare_rows(rows, set(old), GATE, candidate=label)
    result.update(status='spatial development screen only; upstream overlap unknown; no release admission',
                  controls_max_absolute_delta=maximum, gate=GATE, promotion_authorized=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'shipping', 'outputs', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    if args.label in ('shipping', 'lanczos', 'bicubic'):
        ap.error('distinct candidate label required')
    paths = [args.candidate, args.shipping, args.outputs]
    result = score(*(json.loads(p.read_text()) for p in paths), args.label)
    result['evidence_sha256'] = {str(p): digest(p) for p in [*paths, Path(__file__), Path(__file__).with_name('gate_frozen_holdout.py')]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('spatial_gate_pass', 'perceptual_improvements', 'failure_reasons')}, indent=2))


if __name__ == '__main__':
    main()
