#!/usr/bin/env python3
"""Export pinned author PLKSR-Rep FP32 outputs with the frozen 4x-to-2x presentation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from plksr_rep_release import load_release, digest, PINNED, WEIGHTS
from score_checkpoint_frames import paired_references


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'weights', 'frames', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    if digest(args.frames/'manifest.json') != 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f':
        raise ValueError('expected frozen 48-pair development manifest')
    manifest = json.loads((args.frames/'manifest.json').read_text())
    references = paired_references(manifest['frames'])
    for row in manifest['frames']:
        if digest(args.frames/row['file']) != row['sha256']:
            raise ValueError('changed input pixels')
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_release(args.repository, args.weights, deploy=False).to(args.device).eval()
    args.out.mkdir(parents=True)
    report = {'complete': False, 'device': args.device, 'torch': str(torch.__version__),
        'weights_sha256': WEIGHTS, 'source_sha256': PINNED,
        'code_sha256': digest(__file__), 'loader_sha256': digest(Path(__file__).with_name('plksr_rep_release.py')),
        'input_manifest_sha256': digest(args.frames/'manifest.json'),
        'presentation': 'author original FP32 with reflect16/crop64; clamp and nearest RGB8 rounding at 4x; bicubic to 2x',
        'rows': []}
    def save():
        (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    with torch.inference_mode():
        for row in manifest['frames']:
            if row['side'] != 'degraded':
                continue
            with Image.open(args.frames/row['file']) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            height, width = pixels.shape[:2]
            tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None].float().to(args.device)/255
            start = time.perf_counter()
            output = model(tensor).cpu().numpy()
            elapsed = (time.perf_counter()-start)*1000
            if output.shape != (1, 3, 4*height, 4*width) or not np.isfinite(output).all():
                raise ValueError('invalid model output')
            reference = references[row['sequence_id'], row['frame']]
            with Image.open(args.frames/reference['file']) as image:
                if image.size != (2*width, 2*height):
                    raise ValueError('unexpected reference geometry')
            image = Image.fromarray(np.rint(output[0].transpose(1, 2, 0).clip(0, 1)*255).astype(np.uint8))
            image = image.resize((2*width, 2*height), Image.Resampling.BICUBIC)
            path = args.out/Path(row['file']).name
            image.save(path)
            report['rows'].append(dict(row, output_file=path.name, output_sha256=digest(path),
                                      prediction_and_materialization_ms=elapsed))
            save()
            print(path.name, elapsed, flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
