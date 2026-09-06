#!/usr/bin/env python3
"""Render side-by-side zoomed crops of fixed development frames for visual review.

Pixels only; no metric is computed here. Crops are chosen from the reference's
fine-band energy so the model outputs cannot influence where we look.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load  # noqa: E402
from evaluate_sequences import present_4x_at_2x  # noqa: E402


def busiest_window(reference, size):
    gray = np.asarray(reference.convert('L'), dtype=np.float64)
    fine = gray - np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64)
    energy = Image.fromarray((fine ** 2).astype(np.float32), mode='F').resize(
        (reference.width // 8, reference.height // 8), Image.Resampling.BOX)
    energy = np.asarray(energy)
    step = size // 8
    best, where = -1, (0, 0)
    for y in range(2, energy.shape[0] - step - 2):
        for x in range(2, energy.shape[1] - step - 2):
            value = energy[y:y + step, x:x + step].mean()
            if value > best:
                best, where = value, (x * 8, y * 8)
    return where


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=Path, required=True)
    ap.add_argument('--sample', action='append', required=True, help='input stem, e.g. crowd_run-h264-1000000-001')
    ap.add_argument('--checkpoint', nargs=2, action='append', default=[])
    ap.add_argument('--present-4x-at-2x', nargs='*', default=[])
    ap.add_argument('--external', nargs=2, action='append', default=[], help='LABEL DIR with <stem-with-000>-ULTRA.png files')
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--zoom', type=int, default=3)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    models = {label: load(path, device) for label, path in args.checkpoint}
    manifest = json.loads((args.frames / 'manifest.json').read_text())
    for stem in args.sample:
        source = Image.open(args.frames / 'input' / f'{stem}.png').convert('RGB')
        reference = Image.open(args.frames / 'reference' / f'{stem}.png').convert('RGB')
        panels = [('lanczos', source.resize(reference.size, Image.Resampling.LANCZOS))]
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1)[None].to(device).float() / 255
            for label, (model, _, _) in models.items():
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(np.rint(y * 255).astype(np.uint8))
                if label in args.present_4x_at_2x:
                    image = present_4x_at_2x(image, source.size)
                panels.append((label, image))
        rows = manifest['frames']
        row = next((r for r in rows if r['file'] == f'input/{stem}.png'), None)
        for label, folder in args.external:
            head, index = stem.rsplit('-', 1)
            names = [f'{head}-{int(index) - 1:03d}-ULTRA.png']
            if row is not None:
                names.insert(0, f"{head}-{row['frame']:03d}-ULTRA.png")
            path = next((Path(folder) / n for n in names if (Path(folder) / n).exists()), None)
            if path is None:
                raise FileNotFoundError(f'{label}: none of {names} in {folder}')
            panels.append((label, Image.open(path).convert('RGB')))
        panels.append(('reference', reference))
        for label, image in panels:
            if image.size != reference.size:
                raise ValueError(f'{label}: size {image.size} differs from reference {reference.size}')
        x0, y0 = busiest_window(reference, args.crop)
        size = args.crop * args.zoom
        sheet = Image.new('RGB', (size * len(panels), size + 22), 'black')
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(panels):
            crop = image.crop((x0, y0, x0 + args.crop, y0 + args.crop)).resize((size, size), Image.Resampling.NEAREST)
            sheet.paste(crop, (i * size, 22))
            draw.text((i * size + 6, 5), label, fill='white')
        sheet.save(args.out / f'{stem}-crop.png')
        print(stem, 'crop at', (x0, y0), '->', args.out / f'{stem}-crop.png')


if __name__ == '__main__':
    main()
