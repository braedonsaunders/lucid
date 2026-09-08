"""Resumable full-frame codec bank expansion with checked per-source receipts.

Uses the existing codec-before-crop builder unchanged. Completed sources are
immutable; a partial source can be deterministically regenerated. The final
manifest appears only after every requested source has a verified receipt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from build_causal_bank import command, digest, probe, validate_sources
from build_stream_bank import build_source


def write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def checked_rows(out, path, source, expected):
    receipt = json.loads(path.read_text())
    if receipt['source_sha256'] != source['sha256'] or len(receipt['rows']) != expected:
        raise ValueError('source receipt differs from frozen expansion')
    for row in receipt['rows']:
        if row['source_id'] != source['id'] or row['source_sha256'] != source['sha256'] or row['split'] != source['split'] or row['family'] != source['family']:
            raise ValueError('source receipt crosses source identity or split')
        if digest(out/row['file']) != row['sha256']:
            raise ValueError('completed expansion pair changed')
    return receipt['rows']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--windows-per-source', type=int, default=18)
    parser.add_argument('--patches-per-window', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260918)
    args = parser.parse_args()
    if args.windows_per_source < 2 or args.patches_per_window < 1:
        parser.error('at least two windows and one patch required')
    sources = json.loads(args.sources.read_text())
    for source in sources:
        source['path'] = str(Path(source['path']).resolve())
        actual = digest(source['path'])
        if source.get('sha256', actual) != actual:
            raise ValueError('source master changed')
        source['sha256'] = actual
        source['stream'] = list(probe(source['path']))
    validate_sources(sources)
    spec = {'sources': sources, 'seed': args.seed, 'windows_per_source': args.windows_per_source,
            'patches_per_window': args.patches_per_window, 'patch': 128, 'frames': 16,
            'widths': [1280, 1920], 'scale': 2,
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                           ('build_large_stream_bank.py', 'build_stream_bank.py', 'build_causal_bank.py')},
            'ffmpeg': command(['ffmpeg', '-version']).decode().splitlines()[0]}
    args.out.mkdir(parents=True, exist_ok=True)
    spec_path = args.out/'expansion-spec.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError('preserve differently configured expansion')
    elif any(args.out.iterdir()):
        raise ValueError('nonempty expansion without frozen specification')
    else:
        write_json(spec_path, spec)
    receipts = args.out/'receipts'
    receipts.mkdir(exist_ok=True)
    options = SimpleNamespace(**{k: spec[k] for k in
        ('seed', 'windows_per_source', 'patches_per_window', 'patch', 'frames', 'widths')},
        out=args.out, dump_full_frames=None)
    expected = args.windows_per_source*args.patches_per_window
    def build(pair):
        index, source = pair
        receipt = receipts/(source['id']+'.json')
        if not receipt.exists():
            rows = build_source(source, index, options)
            write_json(receipt, {'source_sha256': source['sha256'], 'rows': rows})
        rows = checked_rows(args.out, receipt, source, expected)
        print('source complete', source['id'], len(rows), flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [row for group in pool.map(build, enumerate(sources)) for row in group]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate expanded sequence identity')
    result = {'schema': 1, 'scale': 2, 'frames': 16, 'lr_patch': 128, 'seed': args.seed,
              'sources': sources, 'sequences': rows, 'complete': True,
              'builder': 'full-frame-codec-before-patch', 'builder_sha256': digest(__file__),
              'expansion_spec_sha256': digest(spec_path), 'ffmpeg_version': spec['ffmpeg'],
              'limitation': 'more codec/spatial examples, not independent source footage for each patch; repeated development validation'}
    write_json(args.out/'manifest.json', result)
    print('complete', len(rows), 'sequences', flush=True)


if __name__ == '__main__':
    main()
