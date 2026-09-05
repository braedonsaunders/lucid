#!/usr/bin/env python3
"""Run an image-restoration baseline at its trained patch size without cropping the evaluation frame."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def positions(length, tile=512, overlap=128):
    if length < tile or not 0 <= overlap < tile:
        raise ValueError('image must fit one tile and overlap must be smaller than tile')
    return sorted(set([*range(0, length - tile + 1, tile - overlap), length - tile]))


def merge_patches(patches, width, height, tile=512):
    value = np.zeros((height, width, 3), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    window = np.sin(np.pi * (np.arange(tile) + .5) / tile) ** 2
    window = window[:, None] * window[None, :]
    for x, y, patch in patches:
        if patch.shape != (tile, tile, 3) or x < 0 or y < 0 or x + tile > width or y + tile > height:
            raise ValueError('invalid restored tile size or bounds')
        value[y:y + tile, x:x + tile] += patch * window[..., None]
        weight[y:y + tile, x:x + tile] += window
    if (weight <= 0).any():
        raise ValueError('tiles do not cover the complete frame')
    return (value / weight[..., None]).round().clip(0, 255).astype(np.uint8)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split(source, output):
    (output / 'input').mkdir(parents=True)
    records = []
    for path in sorted(source.glob('*.png')):
        with Image.open(path) as image:
            full = image.convert('RGB').resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)
        tiles = []
        for y in positions(full.height):
            for x in positions(full.width):
                name = f'{path.stem}__x{x}_y{y}.png'
                target = output / 'input' / name
                full.crop((x, y, x + 512, y + 512)).save(target)
                tiles.append({'file': name, 'x': x, 'y': y, 'input_sha256': digest(target)})
        records.append({'file': path.name, 'source_sha256': digest(path),
                        'width': full.width, 'height': full.height, 'tiles': tiles})
    if not records:
        raise ValueError('no source frames')
    (output / 'manifest.json').write_text(json.dumps({'tile_size': 512, 'nominal_overlap': 128,
        'preprocessing': '2x PIL bicubic before tiling; no reference access', 'frames': records}, indent=2) + '\n')


def merge(manifest, restored, output):
    records = json.loads(manifest.read_text())
    output.mkdir(parents=True)
    hashes = {}
    for frame in records['frames']:
        patches = []
        for tile in frame['tiles']:
            path = restored / tile['file']
            hashes[tile['file']] = digest(path)
            with Image.open(path) as image:
                patches.append((tile['x'], tile['y'], np.asarray(image.convert('RGB'))))
        pixels = merge_patches(patches, frame['width'], frame['height'], records['tile_size'])
        Image.fromarray(pixels).save(output / frame['file'])
    (output / 'merge-receipt.json').write_text(json.dumps({'manifest_sha256': digest(manifest),
        'restored_tile_sha256': hashes, 'blending': 'normalized separable sin^2 window, half-pixel centered',
        'code_sha256': digest(Path(__file__))}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    split_parser = sub.add_parser('split')
    split_parser.add_argument('--input', type=Path, required=True)
    split_parser.add_argument('--out', type=Path, required=True)
    merge_parser = sub.add_parser('merge')
    merge_parser.add_argument('--manifest', type=Path, required=True)
    merge_parser.add_argument('--restored', type=Path, required=True)
    merge_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be a fresh directory')
    if args.action == 'split':
        split(args.input, args.out)
    else:
        merge(args.manifest, args.restored, args.out)


if __name__ == '__main__':
    main()
