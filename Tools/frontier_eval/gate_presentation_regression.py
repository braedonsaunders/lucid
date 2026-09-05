#!/usr/bin/env python3
"""Check the fixed 960-pair native presentation regression, without promotion."""
import argparse
import json
from pathlib import Path

from gate_frozen_holdout import compare_rows, digest

SHIPPING = 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
FRAMES = '307d6670ef1c4918798172a16292dd55ade37db61d953efeddb5d60e21154b3b'
SEQUENCES = '788123b0973b0a636dda6a21c1e2f8e9a5e2d58fc565f3dcbaad0a0853815076'
GATE = {
    'source_balanced_lpips_improvement_min': -.005,
    'source_balanced_dists_improvement_min': -.005,
    'per_source_perceptual_regression_max': .005,
    'per_source_fine_correlation_drop_max': .002,
}


def evaluate(report, manifest, config, sequences):
    for item in (report, manifest):
        if item.get('complete') is not True or item.get('split') != 'development-presentation':
            raise ValueError('complete presentation regression required')
        if item['checkpoint_sha256'] != dict(shipping4x=SHIPPING, candidate=SHIPPING):
            raise ValueError('shipping weights changed')
    if (manifest['source_manifest_sha256'] != SEQUENCES
            or manifest['frozen_frames_manifest_sha256'] != FRAMES
            or manifest.get('presentation_corpus') != 'regression960'
            or manifest.get('preserve_display_gain') is not True):
        raise ValueError('fixed corpus or nominal gain changed')
    if (config['candidate_sha256'] != SHIPPING or config['radius'] != 2
            or config.get('preserve_display_gain') is not True
            or config['frames_manifest_sha256'] != FRAMES):
        raise ValueError('configuration differs from frozen presentation')
    expected = {(s['source_id'], s['id'], frame) for s in sequences
                for frame in range(0, s['frames'], 10)}
    if len(expected) != 960:
        raise ValueError('wrong sample count')
    for label in ('shipping4x', 'candidate'):
        keys = [(r['source_id'], r['sequence_id'], r['frame'])
                for r in manifest['rows'] if r['variant'] == label]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('missing, duplicate or unexpected native frames')
    result = compare_rows(report['rows'], expected, GATE, candidate='candidate',
                          baseline='shipping4x', other_labels=())
    result.update(thresholds=GATE, promotion_authorized=False,
                  status='Previously evaluated eight-source native regression; browser cadence and fresh release validation remain')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'manifest', 'config', 'sequences', 'out'):
        ap.add_argument('--'+name, required=True, type=Path)
    args = ap.parse_args()
    report, manifest, config, sequences = [json.loads(p.read_text()) for p in
                                           (args.report, args.manifest, args.config, args.sequences)]
    if digest(args.sequences) != SEQUENCES:
        raise ValueError('sequence manifest changed')
    if report['manifest_sha256'] != digest(args.manifest):
        raise ValueError('scored RGB manifest changed')
    if report['native_config_sha256'] != digest(args.config) or manifest['native_config_sha256'] != digest(args.config):
        raise ValueError('scored native config changed')
    result = evaluate(report, manifest, config, sequences)
    result['evidence_sha256'] = {str(p): digest(p) for p in
                                (args.report, args.manifest, args.config, args.sequences, Path(__file__))}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('spatial_gate_pass', 'failure_reasons', 'perceptual_improvements')}, indent=2))


if __name__ == '__main__':
    main()
