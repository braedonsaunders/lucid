"""Compare Torch emulation to production Metal kernels on deterministic probes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from native_stages import deband, motion_blocks, taa
from native_output_stages import sharpen_luma, grade_luma


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    rows = []
    for kind in ('translated-texture', 'plateau', 'cut', 'stationary'):
        p = args.out / kind
        p.mkdir()
        a = rng.uniform(.1, .9, (32, 48)).astype('float32')
        a[:16, :24] = rng.integers(126, 129, (16, 24)) / 255
        b = np.roll(a, 2, 1)
        b[:16, :24] = rng.integers(126, 129, (16, 24)) / 255
        if kind == 'plateau':
            a[:] = .5
            b[:] = .5
        elif kind == 'cut':
            b = rng.uniform(.05, .3, a.shape).astype('float32')
        elif kind == 'stationary':
            b = a.copy()
        history = (a + rng.uniform(-.01, .01, a.shape)).astype('float32')
        for name, value in [('current', b), ('previous', a), ('history', history)]:
            value.tofile(p / (name + '.bin'))
        current, previous, hist = (torch.tensor(v)[None, None] for v in (b, a, history))
        clean = deband(current, frame=3)
        field = motion_blocks(current, previous)
        output = taa(clean, hist, current, previous, field)
        sharpened = sharpen_luma(current)
        graded = grade_luma(sharpened, .5)
        subprocess.run([str(args.executable.resolve()), str(args.source.resolve()), str(p.resolve()), '48', '32'], check=True)
        for stage, tensor in [('deband', clean), ('motion', field), ('taa', output),
                              ('sharpen', sharpened), ('grade', graded)]:
            expected = tensor[0].permute(1, 2, 0).numpy().flatten()
            actual = np.fromfile(p / ('metal-' + stage + '.bin'), dtype='float32')
            error = np.abs(actual - expected)
            rows.append({'fixture': kind, 'stage': stage, 'max_absolute_error': float(error.max()),
                         'mean_absolute_error': float(error.mean()),
                         'pass': bool(np.isfinite(error).all() and error.max() <= 1e-5)})
    receipt = {'complete': True, 'pass': all(r['pass'] for r in rows), 'tolerance': 1e-5,
               'scope': 'FP32 kernel math only; not NV12 conversion, native fast math, frame history duration, or image-quality equivalence',
               'source_sha256': digest(args.source), 'executable_sha256': digest(args.executable),
               'emulator_sha256': digest(Path(__file__).with_name('native_stages.py')),
               'output_emulator_sha256': digest(Path(__file__).with_name('native_output_stages.py')), 'rows': rows}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['pass']:
        raise SystemExit('Metal parity failed')


if __name__ == '__main__':
    main()
