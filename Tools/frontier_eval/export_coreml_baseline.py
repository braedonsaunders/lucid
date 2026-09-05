#!/usr/bin/env python3
"""Export a pinned native RGB tensor model on frozen paired development frames."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import coremltools as ct
import numpy as np
from PIL import Image
from score_checkpoint_frames import paired_references


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'receipt', 'frames', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    receipt = json.loads(args.receipt.read_text())
    expected = {p: v['sha256'] for p, v in receipt['files'].items()
                if p.startswith(args.model.name+'/')}
    actual = {str(p.relative_to(args.model.parent)): digest(p) for p in args.model.rglob('*') if p.is_file()}
    if not expected or actual != expected:
        raise ValueError('pinned model package is incomplete or changed')
    manifest_path = args.frames/'manifest.json'
    if digest(manifest_path) != 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f':
        raise ValueError('frozen 48-pair development manifest required')
    manifest = json.loads(manifest_path.read_text())
    references = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames/row['file']) != row['sha256']:
            raise ValueError('changed frozen image')
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE)
    spec = model.get_spec()
    if len(spec.description.input) != 1 or len(spec.description.output) != 1:
        raise ValueError('single RGB tensor input/output required')
    input_name, output_name = spec.description.input[0].name, spec.description.output[0].name
    args.out.mkdir(parents=True)
    records = []
    for row in manifest['frames']:
        if row['side'] != 'degraded':
            continue
        with Image.open(args.frames/row['file']) as image:
            image = np.asarray(image.convert('RGB'), dtype=np.float32)/255
        height, width = image.shape[:2]
        if (width, height) != (640, 360):
            raise ValueError('this benchmark requires the supported 360p tensor shape')
        tensor = image.transpose(2, 0, 1)[None].copy()
        started = time.perf_counter()
        output = np.asarray(model.predict({input_name: tensor})[output_name])
        elapsed = (time.perf_counter()-started)*1000
        if output.shape != (1, 3, 2*height, 2*width) or not np.isfinite(output).all():
            raise ValueError('invalid model output')
        reference = references[row['sequence_id'], row['frame']]
        with Image.open(args.frames/reference['file']) as image:
            if image.size != (2*width, 2*height):
                raise ValueError('native output/reference geometry differs')
        pixels = np.rint(output[0].astype(np.float32).transpose(1, 2, 0).clip(0, 1)*255).astype(np.uint8)
        path = args.out/Path(row['file']).name
        Image.fromarray(pixels).save(path)
        records.append(dict(row, output_file=path.name, output_sha256=digest(path),
                            prediction_and_materialization_ms=elapsed))
    (args.out/'manifest.json').write_text(json.dumps({
        'complete': True, 'model_files': actual, 'upstream': receipt,
        'input_manifest_sha256': digest(manifest_path), 'code_sha256': digest(Path(__file__)),
        'coremltools': ct.__version__, 'compute_units': 'CPU_AND_NE; CPU fallback allowed',
        'representation': 'RGB NCHW float32 /255 input; float output clamp and nearest RGB8 rounding; direct2x',
        'timing_note': 'quality export timings include first prediction; use separate warmed runtime benchmark',
        'rows': records}, indent=2)+'\n')
    print(f'Exported {len(records)} pinned native outputs', flush=True)


if __name__ == '__main__':
    main()
