#!/usr/bin/env python3
"""Compose immutable codec banks and training-only teacher caches without copying media.

Output manifests reference checked absolute paths. Original banks must remain
available. This changes data diversity only; it does not relabel any source split.
"""
import argparse
import json
from pathlib import Path

from build_causal_bank import digest, validate_sources


def compose(pairs):
    sources, sequences, targets, receipts = [], [], [], []
    geometry, teacher_settings = None, None
    seen = set()
    for bank, cache in pairs:
        bank, cache = Path(bank).resolve(), Path(cache).resolve()
        manifest = json.loads((bank / 'manifest.json').read_text())
        teacher = json.loads((cache / 'manifest.json').read_text())
        settings = {k: teacher[k] for k in ('teacher_sha256', 'inference', 'baseline_provenance_sha256', 'code_sha256', 'seed')}
        if teacher_settings is not None and settings != teacher_settings:
            raise ValueError('teacher generation settings differ between banks')
        teacher_settings = settings
        bank_hash = digest(bank / 'manifest.json')
        shape = {k: manifest[k] for k in ('scale', 'frames', 'lr_patch')}
        if geometry is not None and geometry != shape:
            raise ValueError('bank geometry differs')
        geometry = shape
        validate_sources(manifest['sources'])
        if teacher['bank_sha256'] != bank_hash or teacher['split'] != 'train only' or teacher.get('reference_used_for_prediction') is not False:
            raise ValueError('teacher cache does not match training bank')
        source_index = {s['id']: s for s in manifest['sources']}
        expected = set()
        for row in manifest['sequences']:
            identity = row['id']
            if identity in seen:
                raise ValueError('duplicate sequence identity')
            seen.add(identity)
            source = source_index[row['source_id']]
            if any(row[k] != source[k] for k in ('split', 'family')) or row['source_sha256'] != source['sha256']:
                raise ValueError('sequence disagrees with source split or identity')
            path = (bank / row['file']).resolve()
            if digest(path) != row['sha256']:
                raise ValueError('sequence bytes changed')
            sequences.append({**row, 'file': str(path)})
            if row['split'] == 'train':
                expected.add(identity)
        cached = set()
        for row in teacher['sequences']:
            if row['id'] not in expected or row['id'] in cached:
                raise ValueError('teacher contains duplicate, unknown or validation identity')
            cached.add(row['id'])
            path = (cache / row['file']).resolve()
            if digest(path) != row['sha256']:
                raise ValueError('teacher bytes changed')
            targets.append({**row, 'file': str(path)})
        if cached != expected:
            raise ValueError('incomplete training-only teacher cache')
        sources.extend(manifest['sources'])
        receipts.append({'bank': str(bank), 'bank_sha256': bank_hash,
                         'teacher_cache': str(cache), 'teacher_manifest_sha256': digest(cache / 'manifest.json'),
                         'teacher_provenance': {k: v for k, v in teacher.items() if k != 'sequences'}})
    if not receipts:
        raise ValueError('at least one bank/cache pair required')
    # Detect family/content leakage across banks, not just within each input.
    validate_sources(sources)
    return ({**geometry, 'sources': sources, 'sequences': sequences,
             'composition': receipts, 'code_sha256': digest(__file__),
             'storage': 'checked absolute references; preserve original banks'}, targets, receipts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair', nargs=2, type=Path, action='append', required=True, metavar=('BANK', 'TEACHER_CACHE'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve existing output; use a fresh directory')
    bank, targets, receipts = compose(args.pair)
    (args.out / 'bank').mkdir(parents=True)
    (args.out / 'teacher-cache').mkdir()
    manifest = args.out / 'bank/manifest.json'
    manifest.write_text(json.dumps(bank, indent=2) + '\n')
    teacher = {'bank_sha256': digest(manifest), 'split': 'train only',
               'sequences': targets, 'composition': receipts, 'code_sha256': digest(__file__),
               'reference_used_for_prediction': False}
    (args.out / 'teacher-cache/manifest.json').write_text(json.dumps(teacher, indent=2) + '\n')
    print(json.dumps({'sources': len(bank['sources']),
                      'train': sum(s['split'] == 'train' for s in bank['sequences']),
                      'validation': sum(s['split'] == 'validation' for s in bank['sequences']),
                      'bank_sha256': digest(manifest)}), flush=True)


if __name__ == '__main__':
    main()
