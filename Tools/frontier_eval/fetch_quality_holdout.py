#!/usr/bin/env python3
"""Fetch the frozen quality holdout without inspecting or scoring its images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from fetch_sources import BASE, SVT, digest, fetch

TAURUS = BASE + 'ftp.ldv.e-technik.tu-muenchen.de/pub/test_sequences/1080p/ReadMe_1080p.txt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--training-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    training = json.loads(args.training_manifest.read_text())['sources']
    sources = spec['sources']
    if len({s['id'] for s in sources}) != len(sources):
        raise ValueError('duplicate holdout sources')
    for key in ('id', 'family'):
        if {s[key] for s in sources} & {s[key] for s in training}:
            raise ValueError(f'holdout overlaps training {key}')
    if any(s['frames'] < 300 or s['first_frame'] != 0 for s in sources):
        raise ValueError('frozen fetch requires at least 300 leading frames')
    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in [('SVT_MultiFormat_v10.pdf', SVT), ('Taurus_1080p_readme.txt', TAURUS)]:
        path = args.out / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
    def download(row):
        if row['family'] == 'svt_multiformat':
            terms, url = 'SVT technology testing/development terms; see accompanying PDF', SVT
        elif row['family'] == 'taurus_1080p':
            terms, url = 'Taurus source readme: no restrictions of use, no copyright', TAURUS
        else:
            raise ValueError('unregistered holdout provenance family')
        return {**fetch((row['id'], row['remote'], terms, url), args.out, row['frames'], download_workers=4),
                'family': row['family']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(download, sources))
    result = {'spec_sha256': digest(args.spec), 'training_manifest_sha256': digest(args.training_manifest),
        'sources': receipts, 'images_inspected': False, 'scores_computed': False,
        'purpose': 'quality holdout; never use these sources for training'}
    (args.out / 'holdout-receipts.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
