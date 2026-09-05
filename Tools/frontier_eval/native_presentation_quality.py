#!/usr/bin/env python3
"""Compare two native presentations of unchanged shipping weights on frozen pixels."""
import argparse
import json
from pathlib import Path
import coremltools as ct
import torch
from PIL import Image
from evaluate_sequences import Scorer, digest, summarize
from score_checkpoint_frames import paired_references
from gate_frozen_holdout import compare_rows

SHIPPING = 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
MANIFEST = 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'
GATE = {'source_balanced_lpips_improvement_min': -.005, 'source_balanced_dists_improvement_min': -.005,
        'per_source_perceptual_regression_max': .005, 'per_source_fine_correlation_drop_max': .002}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'frames', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output directory required')
    if digest(args.frames/'manifest.json') != MANIFEST:
        raise ValueError('frozen 48-pair development inputs required')
    manifest = json.loads((args.frames/'manifest.json').read_text())
    references = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames/row['file']) != row['sha256']:
            raise ValueError('input pixels changed')
    models, provenance = {}, {}
    for label, name in [('shipping', 'shipping4x'), ('quantized', 'quantized_bicubic2x')]:
        package = args.profile/f'{name}_640x360.mlpackage'
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        if model.user_defined_metadata['lucid.checkpoint_sha256'] != SHIPPING:
            raise ValueError('shipping weight identity differs')
        models[label] = model
        provenance[label] = {str(p.relative_to(package)): digest(p) for p in package.rglob('*') if p.is_file()}
    args.out.mkdir(parents=True)
    for label in models:
        (args.out/label).mkdir()
    scorer = Scorer(torch.device('mps'))
    rows, outputs, expected = [], [], set()
    for row in manifest['frames']:
        if row['side'] != 'degraded':
            continue
        reference_row = references[row['sequence_id'], row['frame']]
        with Image.open(args.frames/row['file']) as image:
            source = image.convert('RGB')
        with Image.open(args.frames/reference_row['file']) as image:
            reference = image.convert('RGB')
        if source.size != (640, 360) or reference.size != (1280, 720):
            raise ValueError('frozen native geometry changed')
        expected.add((row['source_id'], row['sequence_id'], row['frame']))
        variants = {'lanczos': source.resize(reference.size, Image.Resampling.LANCZOS)}
        for label, model in models.items():
            output = model.predict({'input': source})['output'].convert('RGB')
            size = (2560, 1440) if label == 'shipping' else reference.size
            if output.size != size:
                raise ValueError('native output scale differs')
            if label == 'shipping':
                output = output.resize(reference.size, Image.Resampling.BICUBIC)
            path = args.out/label/Path(row['file']).name
            output.save(path)
            outputs.append(dict(row, variant=label, output_file=str(path.relative_to(args.out)), output_sha256=digest(path)))
            variants[label] = output
        for label, output in variants.items():
            rows.append({'source_id': row['source_id'], 'sequence_id': row['sequence_id'],
                         'frame': row['frame'], 'variant': label, 'metrics': scorer.spatial(output, reference)})
        print(row['file'], flush=True)
    if len(expected) != 48:
        raise ValueError('unexpected pair coverage')
    gate = compare_rows(rows, expected, GATE, candidate='quantized')
    gate['status'] = 'performance-only native presentation noninferiority; no trained quality or release admission'
    report = {'complete': True, 'manifest_sha256': MANIFEST, 'shipping_sha256': SHIPPING,
        'code_sha256': digest(__file__), 'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
        'gate_sha256': digest(Path(__file__).with_name('gate_frozen_holdout.py')),
        'compute_units': 'CPU_AND_GPU', 'scoring_device': 'mps', 'torch': str(torch.__version__),
        'coremltools': ct.__version__, 'model_packages': provenance, 'rows': rows,
        'output_pixels': outputs, 'summary': summarize(rows), 'gate_thresholds': GATE,
        'noninferiority': gate, 'promotion_authorized': False,
        'scope': '48 repeated development frames, excludes native detail postprocessing and delivery'}
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: gate[k] for k in ('spatial_gate_pass', 'failure_reasons', 'perceptual_improvements')}, indent=2))


if __name__ == '__main__':
    main()
