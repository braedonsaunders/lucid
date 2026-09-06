#!/usr/bin/env python3
"""Render every variant of chosen frozen frames for eyes-first comparison (TestSite/compare.html).

Numbers are shown beside the pictures, never used to hide one. Includes external
outputs (e.g. NVIDIA VFX ULTRA PNGs) when present for a frame.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_checkpoint import load  # noqa: E402
from evaluate_sequences import Scorer, present_4x_at_2x  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, action='append', required=True, help='frozen frames dir (input/, reference/, manifest.json)')
    ap.add_argument('--sample', action='append', required=True, help='input stem, e.g. rush_hour-h264-1000000-100')
    ap.add_argument('--checkpoint', nargs=2, action='append', default=[])
    ap.add_argument('--present-4x-at-2x', nargs='*', default=[])
    ap.add_argument('--external', nargs=3, action='append', default=[], help='LABEL DIR SUFFIX: DIR/<stem>SUFFIX.png or DIR/<seq>-<frame-1:03d>SUFFIX.png')
    ap.add_argument('--native', nargs=3, action='append', default=[],
                    help='LABEL DIR VARIANT: decoded native-pipeline outputs (decode_native_holdout.py manifest) for holdout frames')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', type=Path, default=Path('output/compare'))
    ap.add_argument('--append', action='store_true', help='merge into an existing manifest instead of replacing it')
    args = ap.parse_args()
    device = torch.device(args.device)
    models = {label: load(path, device) for label, path in args.checkpoint}
    scorer = Scorer(device)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if args.append and manifest_path.exists() else {'frames': []}
    existing = {f['id']: f for f in manifest['frames']}
    for stem in args.sample:
        root = next((r for r in args.frames if (r / 'input' / f'{stem}.png').exists()), None)
        if root is None:
            raise FileNotFoundError(stem)
        rows = json.loads((root / 'manifest.json').read_text())['frames']
        row = next((r for r in rows if r['file'] == f'input/{stem}.png'), None)
        true_frame = row['frame'] if row else None
        source = Image.open(root / 'input' / f'{stem}.png').convert('RGB')
        reference = Image.open(root / 'reference' / f'{stem}.png').convert('RGB')
        panels = [('source', source.resize(reference.size, Image.Resampling.NEAREST)),
                  ('lanczos', source.resize(reference.size, Image.Resampling.LANCZOS))]
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float() / 255
            for label, (model, _, _) in models.items():
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(np.rint(y * 255).astype(np.uint8))
                if label in args.present_4x_at_2x:
                    image = present_4x_at_2x(image, source.size)
                panels.append((label, image))
        for label, folder, suffix in args.external:
            head, index = stem.rsplit('-', 1)
            names = [f'{stem}{suffix}.png', f'{head}-{int(index) - 1:03d}{suffix}.png']
            if true_frame is not None:
                names.insert(0, f'{head}-{true_frame:03d}{suffix}.png')
            for candidate in (Path(folder) / n for n in names):
                if candidate.exists():
                    panels.append((label, Image.open(candidate).convert('RGB')))
                    break
        for label, folder, variant in args.native:
            native = json.loads((Path(folder) / 'manifest.json').read_text())
            head = stem.rsplit('-', 1)[0]
            match = next((r for r in native['rows'] if r['sequence_id'] == head and r['frame'] == true_frame and r['variant'] == variant), None)
            if match is not None:
                panels.append((label, Image.open(Path(folder) / match['file']).convert('RGB')))
        panels.append(('reference', reference))
        frame_dir = args.out / stem
        frame_dir.mkdir(exist_ok=True)
        variants = []
        for label, image in panels:
            if image.size != reference.size:
                raise ValueError(f'{label}: size {image.size} differs from reference {reference.size}')
            file = f'{stem}/{label}.png'
            image.save(args.out / file)
            metrics = scorer.spatial(image, reference) if label != 'reference' else None
            variants.append({'name': label, 'file': file, 'metrics': metrics})
        record = {'id': stem, 'source': stem.split('-')[0], 'frame': true_frame, 'variants': variants}
        existing[stem] = record
        print(stem, [v['name'] for v in variants], flush=True)
    manifest['frames'] = [existing[k] for k in sorted(existing)]
    manifest['note'] = 'Metrics are per-frame against the frozen reference; lower LPIPS/DISTS is better. Judge with eyes; the numbers are context.'
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
    print('wrote', manifest_path, len(manifest['frames']), 'frames')


if __name__ == '__main__':
    main()
