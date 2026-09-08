#!/usr/bin/env python3
"""Verify and compile the exact shipping models. Used by every Xcode build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'Lucid/Resources/Models.json'


def digest(package):
    h = hashlib.sha256()
    for p in sorted(package.rglob('*')):
        if p.is_file() and p.name != '.DS_Store':
            h.update(p.relative_to(package).as_posix().encode() + b'\0')
            h.update(p.read_bytes())
    return h.hexdigest()


def verify():
    manifest = json.loads(MANIFEST.read_text())
    names = set()
    for item in all_models(manifest):
        name = item['name']
        if not re.fullmatch(r'[A-Za-z0-9_]+', name) or name in names:
            raise ValueError('Invalid model name')
        names.add(name)
        package = ROOT / 'Model' / (name + '.mlpackage')
        if not (package / 'Manifest.json').is_file() or digest(package) != item['sha256']:
            raise ValueError(f'{name}: missing model or checksum mismatch; update the model manifest deliberately')
    return manifest


def all_models(manifest):
    return manifest['models']


def package(destination):
    manifest = verify()
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / 'ModelBuild.json'
    try:
        prior = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        prior = {}
    toolchain = subprocess.check_output(['xcodebuild', '-version'], text=True).strip()
    for item in all_models(manifest):
        name = item['name']
        target = destination / (name + '.mlmodelc')
        identity = item['sha256'] + ':' + toolchain
        if prior.get(name) == identity and (target / 'coremldata.bin').is_file():
            continue
        with tempfile.TemporaryDirectory(prefix='lucid-model-') as tmp:
            subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(ROOT / 'Model' / (name + '.mlpackage')), tmp], check=True)
            compiled = Path(tmp) / (name + '.mlmodelc')
            if not (compiled / 'coremldata.bin').is_file():
                raise RuntimeError(f'{name}: compiler did not produce a model')
            staged = destination / ('.' + name + '.staging')
            if staged.exists():
                shutil.rmtree(staged)
            shutil.copytree(compiled, staged)
            if target.exists():
                shutil.rmtree(target)
            staged.rename(target)
        prior[name] = identity
    # The bundle carries exactly the ladder in the manifest: anything else
    # left over from an earlier build (a former family, a lab candidate) goes.
    wanted = {x['name'] + '.mlmodelc' for x in all_models(manifest)}
    for stale in destination.glob('*.mlmodelc'):
        if stale.name not in wanted:
            shutil.rmtree(stale)
            prior.pop(stale.name[:-len('.mlmodelc')], None)
    # Only write the receipt once the complete ladder has compiled successfully.
    receipt_path.write_text(json.dumps({x['name']: prior[x['name']] for x in all_models(manifest)}, indent=2) + '\n')
    print(f"Verified {len(all_models(manifest))} packaged SR models")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.destination:
        package(args.destination)
    else:
        print(f"Verified {len(all_models(verify()))} source models")
