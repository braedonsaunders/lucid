"""Materialize checked codec banks as memory-mapped pairs for bounded-RAM training.

Arrays are immutable .npy files; only sampled crops become resident tensors.
Inputs remain intact, split identities are preserved, and publication is atomic
at the final manifest. This does not manufacture additional training examples.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_causal_bank import digest, validate_sources


class MappedSequences:
    def __init__(self, directory, rows):
        self.directory = Path(directory)
        self.rows = tuple(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (np.load(self.directory / row['lr_file'], mmap_mode='r', allow_pickle=False),
                np.load(self.directory / row['hr_file'], mmap_mode='r', allow_pickle=False), row['id'])


def validate_pair(lr, hr, frames):
    if (lr.dtype != np.uint8 or hr.dtype != np.uint8 or lr.ndim != 4 or hr.ndim != 4
            or lr.shape[-1] != 3 or hr.shape[-1] != 3 or lr.shape[0] != frames
            or hr.shape != (frames, lr.shape[1]*2, lr.shape[2]*2, 3)):
        raise ValueError('RGB8 sequence alignment or scale mismatch')


def load_mapped_bank(directory, manifest):
    if manifest.get('storage') != 'mmap-pairs-v1' or not manifest.get('complete') or manifest.get('scale') != 2:
        raise ValueError('completed 2x mapped bank required')
    validate_sources(manifest['sources'])
    sources = {s['id']: s for s in manifest['sources']}
    seen, splits = set(), {'train': [], 'validation': []}
    for row in manifest['sequences']:
        source = sources[row['source_id']]
        if row['id'] in seen or any(row[k] != source[k] for k in ('split', 'family')) or row['source_sha256'] != source['sha256']:
            raise ValueError('sequence provenance disagrees with source split')
        seen.add(row['id'])
        for kind in ('lr', 'hr'):
            path = Path(directory) / row[kind + '_file']
            if digest(path) != row[kind + '_sha256']:
                raise ValueError('mapped training array changed')
        lr, hr, _ = MappedSequences(directory, [row])[0]
        validate_pair(lr, hr, manifest['frames'])
        splits[row['split']].append(row)
    if not all(splits.values()):
        raise ValueError('both source-disjoint splits required')
    return manifest, {split: MappedSequences(directory, rows) for split, rows in splits.items()}


def materialize(banks, out):
    out = Path(out)
    if not banks:
        raise ValueError('at least one input bank required')
    if out.exists():
        raise ValueError('fresh mapped-bank directory required')
    manifests = [json.loads((Path(bank)/'manifest.json').read_text()) for bank in banks]
    sources, index, ids, geometry = [], {}, set(), None
    for manifest in manifests:
        shape = {k: manifest[k] for k in ('scale', 'frames', 'lr_patch')}
        if shape['scale'] != 2 or (geometry is not None and geometry != shape):
            raise ValueError('bank geometry differs')
        geometry = shape
        validate_sources(manifest['sources'])
        for source in manifest['sources']:
            prior = index.get(source['id'])
            if prior is not None:
                if any(prior[k] != source[k] for k in ('sha256', 'split', 'family')):
                    raise ValueError('source identity differs across banks')
            else:
                index[source['id']] = source
                sources.append(source)
        for row in manifest['sequences']:
            source = index[row['source_id']]
            if row['id'] in ids or any(row[k] != source[k] for k in ('split', 'family')) or row['source_sha256'] != source['sha256']:
                raise ValueError('duplicate or conflicting sequence provenance')
            ids.add(row['id'])
    validate_sources(sources)
    out.mkdir(parents=True)
    rows, provenance = [], []
    for bank, manifest in zip(banks, manifests):
        bank = Path(bank)
        provenance.append({'bank': str(bank.resolve()), 'manifest_sha256': digest(bank/'manifest.json')})
        for row in manifest['sequences']:
            path = bank/row['file']
            if digest(path) != row['sha256']:
                raise ValueError('source pair changed')
            with np.load(path, allow_pickle=False) as values:
                lr, hr = values['lr'], values['hr']
                validate_pair(lr, hr, manifest['frames'])
                saved = {k: v for k, v in row.items() if k not in ('file', 'sha256')}
                saved['original_pair_sha256'] = row['sha256']
                for kind, pixels in (('lr', lr), ('hr', hr)):
                    target = out/row['split']/(row['id']+'-'+kind+'.npy')
                    target.parent.mkdir(exist_ok=True)
                    with target.open('xb') as stream:
                        np.save(stream, pixels, allow_pickle=False)
                    saved[kind+'_file'] = str(target.relative_to(out))
                    saved[kind+'_sha256'] = digest(target)
                rows.append(saved)
            if len(rows) % 100 == 0:
                print('mapped', len(rows), 'of', len(ids), flush=True)
    result = {**geometry, 'storage': 'mmap-pairs-v1', 'complete': True,
              'sources': sources, 'sequences': rows, 'composition': provenance,
              'materializer_sha256': digest(__file__)}
    partial = out/'manifest.partial.json'
    partial.write_text(json.dumps(result, indent=2)+'\n')
    partial.replace(out/'manifest.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.bank, args.out)
    print('complete', len(result['sequences']), 'mapped pairs', flush=True)


if __name__ == '__main__':
    main()
