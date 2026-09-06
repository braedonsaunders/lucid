#!/usr/bin/env python3
"""NVIDIA VFX SDK screen on any frozen frame manifest; calibration only, no fallback, no training use.

Same isolated-worker lifecycle as evaluate_nvidia_vfx.py (the SDK blocks in destroy()
on this host). Scores the SDK modes and Lanczos on CUDA; Lucid rows for the same
samples come from the separate torch/native reports so the comparison stays offline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_nvidia_vfx import worker, WHEEL, digest, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', type=Path)
    for key in ('frames', 'installation', 'out'):
        ap.add_argument('--'+key, type=Path)
    ap.add_argument('--modes', nargs='+', default=['ULTRA'], choices=('BICUBIC', 'HIGH', 'ULTRA'))
    ap.add_argument('--frames-sha256', help='Expected manifest digest; refuse anything else')
    args = ap.parse_args()
    if args.worker:
        worker(args.worker); return
    if any(getattr(args, k) is None for k in ('frames', 'installation', 'out', 'frames_sha256')):
        ap.error('frames, installation, out and frames-sha256 required')
    if args.out.exists(): ap.error('fresh output required')
    manifest_sha = digest(args.frames/'manifest.json')
    if manifest_sha != args.frames_sha256: raise ValueError('frozen frames differ from the declared manifest')
    installation = json.loads(args.installation.read_text())
    if installation['sha256'] != WHEEL: raise ValueError('official wheel identity changed')
    sdk_root = args.installation.parent/'deps'
    for relative, sha in installation['files'].items():
        if digest(sdk_root/relative) != sha: raise ValueError('SDK file changed')
    source = json.loads((args.frames/'manifest.json').read_text())
    refs = {(r['sequence_id'], r['frame']): r for r in source['frames'] if r['side'] == 'reference'}
    inputs = [r for r in source['frames'] if r['side'] == 'degraded']
    if not inputs or {(r['sequence_id'], r['frame']) for r in inputs} != set(refs): raise ValueError('unpaired frames')
    for row in source['frames']:
        if digest(args.frames/row['file']) != row['sha256']: raise ValueError('decoded pixels changed')
    args.out.mkdir(parents=True)
    report = dict(complete=False, inference_complete=False, modes=list(args.modes), split=source.get('split', 'unknown'),
        manifest_sha256=manifest_sha, installation_sha256=digest(args.installation), code_sha256=digest(__file__),
        outputs=[], rows=[], limitations=['SDK RGB image screen, not browser-driver VSR or temporal evaluation',
            'Fresh process per image due to verified SDK teardown hang',
            'Calibration of the competitive target on the same holdout; SDK outputs never enter training'])
    save(args.out/'report.json', report)
    for row in inputs:
        for mode in args.modes:
            stem = f"{row['sequence_id']}-{row['frame']:03d}-{mode}"
            image_path, receipt_path, spec_path = (args.out/(stem+s) for s in ('.png', '.json', '-input.json'))
            save(spec_path, dict(mode=mode, input=str((args.frames/row['file']).resolve()), input_sha256=row['sha256'],
                                 output=str(image_path.resolve()), receipt=str(receipt_path.resolve())))
            started = time.perf_counter()
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(spec_path.resolve())],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try: stdout, stderr = child.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=False, capture_output=True)
                stdout, stderr = child.communicate(); raise RuntimeError('isolated SDK worker timed out')
            (args.out/(stem+'.log')).write_bytes(stdout+stderr)
            if child.returncode: raise RuntimeError('SDK worker failed; no scaler fallback')
            receipt = json.loads(receipt_path.read_text())
            if not receipt['complete'] or digest(image_path) != receipt['output_sha256']:
                raise ValueError('incomplete or changed SDK output')
            report['outputs'].append(dict(row, variant='nvvfx_'+mode, file=image_path.name,
                image_sha256=receipt['output_sha256'], receipt=receipt_path.name,
                process_wall_seconds=time.perf_counter()-started))
            if len(report['outputs']) % 20 == 0: save(args.out/'report.json', report)
            print(stem, 'frozen', flush=True)
    report['inference_complete'] = True; save(args.out/'report.json', report)
    from PIL import Image
    import torch
    from evaluate_sequences import Scorer, summarize
    scorer = Scorer(torch.device('cuda'))
    outputs = {(r['sequence_id'], r['frame'], r['variant']): r for r in report['outputs']}
    for row in inputs:
        key = row['sequence_id'], row['frame']
        reference = Image.open(args.frames/refs[key]['file']).convert('RGB')
        variants = {'lanczos': Image.open(args.frames/row['file']).convert('RGB').resize(reference.size, Image.Resampling.LANCZOS)}
        for mode in args.modes:
            variants['nvvfx_'+mode] = Image.open(args.out/outputs[(*key, 'nvvfx_'+mode)]['file']).convert('RGB')
        for variant, image in variants.items():
            report['rows'].append(dict(source_id=row['source_id'], sequence_id=key[0], frame=key[1], variant=variant,
                                       metrics=scorer.spatial(image, reference)))
        if len(report['rows']) % 40 == 0: save(args.out/'report.json', report)
        print(key, 'scored', flush=True)
    report.update(complete=True, summary=summarize(report['rows']), scoring_device='cuda', torch=str(torch.__version__),
                  scorer_sha256=digest(Path(__file__).with_name('evaluate_sequences.py')))
    save(args.out/'report.json', report)


if __name__ == '__main__': main()
