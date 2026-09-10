#!/usr/bin/env python3
"""Score a frozen baseline or trained refiner head on the pinned validation set.

The phase-1 control is the frozen trunk itself: no training, just this
evaluation of the init checkpoint with a zero-width head. Trained arms
write their own validation.json; this script produces the identical
schema for any checkpoint so the comparison is apples to apples.

Source-proxy diagnostics only; native export and timing unverified.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

from train_causal_detail import digest, load_bank
from train_ssm_span import select_validation
from eval_checkpoint import load
from train_span import Unshuffled
from architectures.output_refiner import OutputRefinerSPAN, load_output_refiner_checkpoint
from train_output_refiner import validate


def load_any(path, device):
    try:
        return load_output_refiner_checkpoint(path, device)
    except ValueError:
        pass
    model, step, frames = load(path, device)
    if type(model) is not Unshuffled or frames != 1:
        raise ValueError('plain baseline must be a single-frame 2x SPAN checkpoint')
    wrapped = OutputRefinerSPAN(model, head_channels=0).to(device)
    return wrapped.eval(), {'step': step}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--reds-bank', type=Path, required=True)
    ap.add_argument('--init', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--source-motion-policy', choices=('raw', 'taa', 'search'), default='raw')
    args = ap.parse_args()
    if args.out.exists() or args.frames < 3:
        ap.error('fresh output and frames >=3 required')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] < args.frames:
        ap.error('adequate bank sequence geometry required')
    validation_data, validation_sources = select_validation(args.bank, args.reds_bank, manifest, data)
    model, _ = load_any(str(args.init), 'cuda')
    initial = copy.deepcopy(model.sr).eval().requires_grad_(False)
    result = validate(model, validation_data, validation_sources,
                      use_head=True, baseline=initial,
                      source_motion_policy=args.source_motion_policy)
    result['checkpoint_sha256'] = digest(args.init)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    files = ['Tools/architectures/output_refiner.py',
             'Tools/experiments/train_output_refiner.py',
             'Tools/experiments/eval_output_refiner.py',
             'Tools/train_span.py',
             'Tools/frontier_eval/evaluate_sequences.py', 'Tools/eval_checkpoint.py']
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'eval.json').write_text(json.dumps({
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'),
        'reds_bank_sha256': digest(args.reds_bank / 'manifest.json'),
        'checkpoint_sha256': digest(args.init),
        'source_hashes': {p: digest(source_root / p) for p in files},
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}, indent=2) + '\n')
    for label in ('base', 'current_only', 'refined', 'initial'):
        balanced = result['summary'].get(label, {}).get('source_balanced', {})
        print(json.dumps({'variant': label, 'source_balanced': balanced}), flush=True)
    (args.out / 'complete.json').write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    main()
