#!/usr/bin/env python3
"""Compare native motion history on/off with identical clips and settings.

This small legacy-clip check catches regressions; Dinner is a training source,
and these results are explicitly not a source-disjoint promotion gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image
import torch
from evaluate_sequences import Scorer, average, digest, save, temporal_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--app', nargs=2, action='append', metavar=('LABEL', 'EXECUTABLE'), required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=12)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = json.loads((root / 'Tools/tuning.json').read_text())
    scorer = Scorer(torch.device('mps'))
    report = {'purpose': 'legacy native motion regression screening', 'complete': False,
        'limitations': ['Dinner overlaps training; no independent quality promotion claim',
            '1440p native output resized to legacy 1080p reference with Lanczos, consistently for all variants',
            'app builds and concurrent work differ; timings must not be used for performance claims'],
        'tuning': base, 'rows': []}
    for label, app in args.app:
        for scene in ('crowdrun', 'dinner'):
            clip = root / 'TestSite' / f'{scene}-360p-350k.mp4'
            reference = root / 'TestSite' / f'{scene}-1080p.mp4'
            for motion in (0, 1):
                with tempfile.TemporaryDirectory(prefix='lucid-native-motion-') as directory:
                    out = Path(directory)
                    tuning = out / 'tuning.json'
                    tuning.write_text(json.dumps({**base, 'stageMotion': motion}))
                    env = dict(os.environ, LUCID_TUNING=str(tuning), LUCID_LEARNED='1',
                               LUCID_MODEL_STEM='SPAN_x4_ch32utc_')
                    result = subprocess.run([app, '--bench', str(clip), str(reference), directory,
                        '0', str(args.frames)], env=env, capture_output=True, text=True, timeout=180)
                    if result.returncode:
                        raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])
                    files = sorted(out.glob('*-detail.png'))
                    if len(files) != args.frames:
                        raise ValueError(f'expected {args.frames} native frames, got {len(files)}')
                    outputs, references = [], []
                    for file in files:
                        with Image.open(out / file.name.replace('-detail', '-reference')) as image:
                            ref = image.convert('RGB').copy()
                        with Image.open(file) as image:
                            output = image.convert('RGB').copy()
                        outputs.append(output.resize(ref.size, Image.Resampling.LANCZOS))
                        references.append(ref)
                    metrics = average([scorer.spatial(outputs[i], references[i]) for i in range(0, args.frames, 3)])
                    metrics.update(temporal_metrics(outputs, references))
                    row = {'build': label, 'executable_sha256': digest(app), 'scene': scene,
                        'source_sha256': digest(clip), 'reference_sha256': digest(reference),
                        'motion': motion, 'metrics': metrics,
                        'output_fingerprint': digest(files[0]), 'bench_log': result.stdout}
                    report['rows'].append(row)
                    save(args.report, report)
                    print(label, scene, motion, json.dumps(metrics), flush=True)
    report['complete'] = True
    save(args.report, report)


if __name__ == '__main__':
    main()
