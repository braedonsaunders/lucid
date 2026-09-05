#!/usr/bin/env python3
"""Score exported full frames against a supplied restoration output directory."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from evaluate_sequences import Scorer, digest, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--label', default='external')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.label or args.label in ('lanczos', 'bicubic'):
        parser.error('external label must be distinct from interpolation baselines')
    manifest = json.loads((args.frames / 'manifest.json').read_text())
    for row in manifest['frames']:
        if digest(args.frames / row['file']) != row['sha256']:
            raise ValueError('exported frame changed')
    scorer = Scorer(torch.device('mps'))
    records, output_hashes = [], {}
    inputs = [r for r in manifest['frames'] if r['side'] == 'degraded']
    for row in inputs:
        name = Path(row['file']).name
        output_path = args.outputs / name
        output_hashes[name] = digest(output_path)
        with Image.open(args.frames / row['file']) as image:
            source = image.convert('RGB')
        with Image.open(args.frames / 'reference' / name) as image:
            reference = image.convert('RGB')
        with Image.open(output_path) as image:
            output = image.convert('RGB')
        variants = {'lanczos': source.resize(reference.size, Image.Resampling.LANCZOS),
                    'bicubic': source.resize(reference.size, Image.Resampling.BICUBIC), args.label: output}
        for label, result in variants.items():
            metrics = scorer.spatial(result, reference)
            records.append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                            'frame': row['frame'], 'variant': label, 'metrics': metrics})
        print(name, flush=True)
    report = {'purpose': 'full-frame spatial development baseline; no temporal or native speed claim',
        'input_manifest_sha256': digest(args.frames / 'manifest.json'),
        'sequence_manifest_sha256': manifest['sequence_manifest_sha256'],
        'output_sha256': output_hashes, 'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
        'export_scorer_sha256': digest(Path(__file__)), 'rows': records, 'summary': summarize(records),
        'limitations': ['48 sampled frames from three source identities; not an untouched final test',
                        'upstream training overlap not verified; generalization is not established']}
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
