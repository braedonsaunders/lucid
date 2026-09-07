#!/usr/bin/env python3
"""Run the NVIDIA VFX SDK ULTRA scaler over dumped full-frame LR windows (any size), one process per frame.

Teacher generation for distillation. Outputs 2x PNGs beside a receipt per frame.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    receipt = {'complete': False}
    try:
        import numpy as np
        from PIL import Image
        import torch
        import nvvfx
        from nvvfx.effects import QualityLevel
        source = np.array(Image.open(spec['input']).convert('RGB'))
        h, w = source.shape[:2]
        tensor = torch.from_numpy(source).permute(2, 0, 1).float().div(255).contiguous().cuda()
        sr = nvvfx.VideoSuperRes(getattr(QualityLevel, spec['mode']))
        sr.output_width = 2 * w; sr.output_height = 2 * h
        sr.load(); torch.cuda.synchronize()
        output = torch.from_dlpack(sr.run(tensor).image).clone(); torch.cuda.synchronize()
        if tuple(output.shape) != (3, 2 * h, 2 * w) or not torch.isfinite(output).all():
            raise ValueError('invalid SDK output')
        if float(output.max()) < .1 or float(output.std()) < .001:
            raise ValueError('black or constant SDK output')
        pixels = output.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(pixels).save(spec['output'])
        receipt.update(complete=True, output_sha256=digest(spec['output']), size=[2 * w, 2 * h])
        Path(spec['receipt']).write_text(json.dumps(receipt))
        sys.stdout.flush(); os._exit(0)
    except BaseException:
        receipt['error'] = traceback.format_exc(); Path(spec['receipt']).write_text(json.dumps(receipt))
        sys.stdout.flush(); os._exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', type=Path)
    ap.add_argument('--frames', type=Path, help='directory with windows.json and <window>/<frame>.png')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--mode', default='ULTRA')
    args = ap.parse_args()
    if args.worker: worker(args.worker); return
    windows = json.loads((args.frames / 'windows.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / 'progress.json'
    done = json.loads(log.read_text()) if log.exists() else {}
    started_all = time.perf_counter(); n = 0
    for window in windows['windows']:
        folder = args.out / window['window']; folder.mkdir(exist_ok=True)
        for i in range(window['frames']):
            key = f"{window['window']}/{i:03d}"
            if done.get(key): continue
            spec = {'mode': args.mode, 'input': str((args.frames / window['window'] / f'{i:03d}.png').resolve()),
                    'output': str((folder / f'{i:03d}.png').resolve()), 'receipt': str((folder / f'{i:03d}.json').resolve())}
            spec_path = folder / f'{i:03d}-input.json'; spec_path.write_text(json.dumps(spec))
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(spec_path.resolve())],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try: child.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=False, capture_output=True)
                child.communicate(); raise RuntimeError('SDK worker timed out')
            receipt = json.loads(Path(spec['receipt']).read_text())
            if child.returncode or not receipt.get('complete'):
                raise RuntimeError(f'SDK worker failed on {key}: {receipt.get("error", "")[-400:]}')
            done[key] = receipt['output_sha256']; n += 1
            if n % 20 == 0:
                log.write_text(json.dumps(done)); print(key, f'{(time.perf_counter()-started_all)/60:.1f} min', flush=True)
    log.write_text(json.dumps(done)); print('TEACHER-FRAMES-DONE', len(done), flush=True)


if __name__ == '__main__': main()
