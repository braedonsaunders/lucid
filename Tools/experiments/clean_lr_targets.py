"""Derive clean RGB LR supervision using the bank's FFmpeg Lanczos geometry.

This target precedes codec loss and 4:2:0 conversion. It is NOT a recovered
bit-exact pre-encode YUV plane. Exclude four LR border pixels when supervising
the cropped target, because the original full-frame resize had more context.
"""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from train_causal_detail import digest


def clean_lr_rgb(reference):
    if reference.dtype != np.uint8 or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError('THWC RGB8 reference sequence required')
    frames, height, width, _ = reference.shape
    if height % 2 or width % 2:
        raise ValueError('even reference geometry required')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-i', 'pipe:0', '-frames:v', str(frames),
               '-vf', f'scale={width//2}:{height//2}:flags=lanczos', '-threads', '1',
               '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    result = subprocess.run(command, input=reference.tobytes(), capture_output=True, check=True, timeout=60)
    expected = frames * (height // 2) * (width // 2) * 3
    if len(result.stdout) != expected:
        raise ValueError('incomplete clean LR conversion')
    return np.frombuffer(result.stdout, np.uint8).reshape(frames, height // 2, width // 2, 3).copy()


def load_clean_targets(directory, data, bank_hash):
    """Build once or validate an immutable target cache for both source splits."""
    directory = Path(directory)
    expected = {identity: (lr, hr, split) for split, sequences in data.items()
                for lr, hr, identity in sequences}
    targets = {}
    if directory.exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not manifest.get('complete') or manifest['bank_sha256'] != bank_hash or manifest['target'] != 'clean RGB Lanczos 2x downsample, pre-codec/pre-420':
            raise ValueError('clean cache provenance differs')
        rows = manifest['sequences']
        if len(rows) != len(expected) or {r['id'] for r in rows} != expected.keys():
            raise ValueError('clean cache coverage differs')
        for row in rows:
            path = directory / row['file']
            if digest(path) != row['sha256'] or row['split'] != expected[row['id']][2]:
                raise ValueError('clean cache changed or crossed source splits')
            with np.load(path, allow_pickle=False) as values:
                target = values['clean_lr'].copy()
            if target.dtype != np.uint8 or target.shape != expected[row['id']][0].shape:
                raise ValueError('clean LR target geometry differs')
            targets[row['id']] = target
        return targets, manifest
    directory.mkdir(parents=True)
    rows = []
    for identity, (lr, hr, split) in expected.items():
        target = clean_lr_rgb(hr)
        if target.shape != lr.shape:
            raise ValueError('clean LR target does not align with decoded LR')
        path = directory / (identity + '.npz')
        np.savez_compressed(path, clean_lr=target)
        targets[identity] = target
        rows.append({'id': identity, 'file': path.name, 'sha256': digest(path), 'split': split})
        if len(rows) % 100 == 0:
            print('clean targets', len(rows), '/', len(expected), flush=True)
    manifest = {'complete': True, 'bank_sha256': bank_hash, 'code_sha256': digest(__file__),
                'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
                'target': 'clean RGB Lanczos 2x downsample, pre-codec/pre-420',
                'crop_boundary_exclusion_lr': 4, 'sequences': rows}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return targets, manifest


def batch_clean_lr(data, targets, rng, count, frames, crop):
    """Sample the same time, crop, flip and transpose for decoded/clean/HR data."""
    batches = [[], [], []]
    for _ in range(count):
        lr, hr, identity = data[int(rng.integers(len(data)))]
        t = int(rng.integers(lr.shape[0] - frames + 1))
        y, x = (int(rng.integers(lr.shape[d] - crop + 1)) for d in (1, 2))
        arrays = [lr[t:t+frames, y:y+crop, x:x+crop],
                  targets[identity][t:t+frames, y:y+crop, x:x+crop],
                  hr[t:t+frames, 2*y:2*(y+crop), 2*x:2*(x+crop)]]
        for axis in (1, 2):
            if rng.random() < .5:
                arrays = [np.flip(a, axis) for a in arrays]
        if rng.random() < .5:
            arrays = [np.swapaxes(a, 1, 2) for a in arrays]
        for batch, array in zip(batches, arrays):
            batch.append(array.copy())
    return tuple(torch.from_numpy(np.stack(batch)).permute(0, 1, 4, 2, 3).float() / 255 for batch in batches)
