"""Replay fixed recurrent weights with explicit legacy and corrected motion seeds."""
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


def validation_pairs(bank, training, *, reds=False):
    manifest = json.loads((bank / 'manifest.json').read_text())
    validate_sources(manifest['sources'])
    rows = [r for r in manifest['sequences'] if r['split'] == 'validation'
            and (not reds or r['id'].endswith('-00'))]
    ids = {r['source_id'] for r in rows}
    if len(rows) != 36 or (reds and ids != {'reds_154', 'reds_073'}):
        raise ValueError('fixed validation selection changed')
    for field in ('id', 'family', 'sha256'):
        selected = {s[field] for s in manifest['sources'] if s['id'] in ids}
        trained = {s[field] for s in training['sources'] if s['split'] == 'train'}
        if selected & trained:
            raise ValueError('evaluation source overlaps training: ' + field)
    sources = {s['id']: s for s in manifest['sources']}
    pairs = []
    for row in rows:
        source = sources[row['source_id']]
        if (any(row[k] != source[k] for k in ('split', 'family'))
                or row['source_sha256'] != source['sha256']):
            raise ValueError('sequence provenance differs from source')
        path = bank / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('validation pair changed')
        with np.load(path, allow_pickle=False) as pair:
            lr, hr = pair['lr'].copy(), pair['hr'].copy()
        validate_pair(lr, hr, 16)
        pairs.append((lr, hr, row['id']))
    return pairs, {r['id']: r['source_id'] for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-bank', 'reds-bank', 'initial', 'control', 'sr', 'decoded', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh evaluation receipt required')
    torch.set_num_threads(4)
    training = json.loads((args.training_bank / 'manifest.json').read_text())
    pairs, sources = validation_pairs(args.training_bank, training)
    reds, reds_sources = validation_pairs(args.reds_bank, training, reds=True)
    if sources.keys() & reds_sources.keys():
        raise ValueError('duplicate validation sequence IDs')
    pairs.extend(reds)
    sources.update(reds_sources)
    models, states = {}, {}
    for name in ('control', 'sr', 'decoded'):
        models[name], states[name] = load_recurrent_checkpoint(getattr(args, name), 'cuda')
    experiments = [s['experiment'] for s in states.values()]
    for key in ('bank_sha256', 'checkpoint_sha256', 'frozen_backbone_sha256',
                'first_decoded_sequence_sha256', 'first_output_sha256', 'discriminator_initial_sha256'):
        if len({e[key] for e in experiments}) != 1:
            raise ValueError('unmatched recurrent arms: ' + key)
    if (experiments[0]['bank_sha256'] != digest(args.training_bank / 'manifest.json')
            or experiments[0]['checkpoint_sha256'] != digest(args.initial)):
        raise ValueError('training bank or initialization changed')
    for name, model in models.items():
        a = states[name]['experiment']['args']
        if (a['no_history'] != (name == 'control') or not a['joint']
                or not a['native_input_stages'] or model.motion_seed != 'taa'
                or model.history_source != ('decoded' if name == 'decoded' else 'sr')):
            raise ValueError('expected legacy-seed matched training declarations')
    base, _, frames = load(args.initial, 'cuda')
    if frames != 1:
        raise ValueError('single-frame initial checkpoint required')
    initial = RecurrentSPAN(base).sr.eval()
    root = Path(__file__).resolve().parents[1]
    result = {'complete': False,
              'scope': 'Fixed weights; explicit motion-seed replay on 72 full 16-frame validation sequences; native-input proxy, not native playback or retraining',
              'selection': sources,
              'checkpoint_sha256': {n: digest(getattr(args, n)) for n in ('initial', 'control', 'sr', 'decoded')},
              'bank_sha256': {n: digest(getattr(args, n) / 'manifest.json') for n in ('training_bank', 'reds_bank')},
              'source_sha256': {str(p.relative_to(root)): digest(p) for p in [
                  Path(__file__).resolve(), root / 'experiments/native_stages.py',
                  root / 'experiments/recurrent_frame_feeding.py', root / 'experiments/train_recurrent_span.py',
                  root / 'architectures/recurrent_span.py', root / 'frontier_eval/evaluate_sequences.py']},
              'torch': str(torch.__version__), 'device': torch.cuda.get_device_name(),
              'trained_motion_seed': {n: m.motion_seed for n, m in models.items()}, 'reports': {}}
    for name, model in models.items():
        for policy in (('taa',) if name == 'control' else ('taa', 'search')):
            label = name + '_' + policy
            print('evaluating', label, flush=True)
            model.motion_seed = policy  # Explicit replay override; checkpoint bytes stay unchanged.
            result['reports'][label] = validate(model, pairs, sources, True,
                use_history=name != 'control', baseline=initial)
            print('finished', label, flush=True)
    result['complete'] = True
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('complete fixed-weight motion replay', flush=True)


if __name__ == '__main__':
    main()
