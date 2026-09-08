"""Evaluate matched recurrent checkpoints on unused REDS validation windows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.recurrent_span import RecurrentSPAN
from build_causal_bank import validate_sources
from eval_checkpoint import load
from mmap_training_bank import validate_pair
from recurrent_frame_feeding import load_recurrent_checkpoint
from train_causal_detail import digest
from train_recurrent_span import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'training-bank', 'initial', 'control', 'recurrent', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh evaluation receipt required')
    torch.set_num_threads(4)
    manifest = json.loads((args.bank/'manifest.json').read_text())
    training = json.loads((args.training_bank/'manifest.json').read_text())
    validate_sources(manifest['sources'])
    rows = [r for r in manifest['sequences'] if r['split'] == 'validation' and r['id'].endswith('-00')]
    ids = {r['source_id'] for r in rows}
    if ids != {'reds_154', 'reds_073'} or len(rows) != 36:
        raise ValueError('fixed two-source, first-patch-per-window selection changed')
    for field in ('id', 'family', 'sha256'):
        selected = {s[field] for s in manifest['sources'] if s['id'] in ids}
        trained = {s[field] for s in training['sources'] if s['split'] == 'train'}
        if selected & trained:
            raise ValueError('evaluation source overlaps training: '+field)
    pairs = []
    for row in rows:
        path = args.bank/row['file']
        if digest(path) != row['sha256']:
            raise ValueError('validation pair changed')
        with np.load(path, allow_pickle=False) as pair:
            lr, hr = pair['lr'].copy(), pair['hr'].copy()
        validate_pair(lr, hr, 16)
        pairs.append((lr, hr, row['id']))
    control, cs = load_recurrent_checkpoint(args.control, 'cuda')
    candidate, rs = load_recurrent_checkpoint(args.recurrent, 'cuda')
    ce, re = cs['experiment'], rs['experiment']
    keys = ('bank_sha256', 'checkpoint_sha256', 'frozen_backbone_sha256',
            'first_decoded_sequence_sha256', 'first_output_sha256', 'discriminator_initial_sha256')
    if any(ce[k] != re[k] for k in keys):
        raise ValueError('unmatched recurrent arms')
    if (ce['bank_sha256'] != digest(args.training_bank/'manifest.json')
            or ce['checkpoint_sha256'] != digest(args.initial)
            or not ce['args']['no_history'] or re['args']['no_history']
            or not all(e['args']['joint'] and e['args']['native_input_stages'] for e in (ce, re))):
        raise ValueError('declared joint recurrence comparison changed')
    base, _, frames = load(args.initial, 'cuda')
    if frames != 1:
        raise ValueError('single-frame initial checkpoint required')
    initial = RecurrentSPAN(base).sr.eval()
    sources = {r['id']: r['source_id'] for r in rows}
    result = {'complete': False,
              'scope': 'Two held-out REDS sources, first spatial patch from every codec window; full 16-frame source-stage proxy; not native output quality',
              'selection': [r['id'] for r in rows],
              'bank_sha256': digest(args.bank/'manifest.json'),
              'training_bank_sha256': digest(args.training_bank/'manifest.json'),
              'checkpoint_sha256': {name: digest(getattr(args, name)) for name in ('initial', 'control', 'recurrent')},
              'code_sha256': digest(__file__),
              'scorer_sha256': digest(Path(__file__).resolve().parents[1]/'frontier_eval/evaluate_sequences.py'),
              'torch': str(torch.__version__), 'device': 'cuda', 'reports': {}}
    for name, model, history in [('control', control, False), ('recurrent', candidate, True)]:
        print('evaluating', name, flush=True)
        result['reports'][name] = validate(model, pairs, sources, True, use_history=history, baseline=initial)
    result['complete'] = True
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('complete REDS recurrence evaluation', flush=True)


if __name__ == '__main__':
    main()
