#!/usr/bin/env python3
"""Generate the Safari companion resources from the shared browser implementation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'BrowserExtension'
TARGET = ROOT / 'SafariCompanion/Lucid Companion/Lucid Companion Extension/Resources'
FILES = ['background.js', 'bridge-auth.js', 'stream-policy.js', 'content.js', 'surface.html', 'surface.js']


def expected():
    result = {f: (SOURCE / f).read_bytes() for f in FILES}
    m = json.loads((SOURCE / 'manifest.json').read_text())
    m.pop('minimum_chrome_version', None)
    m.pop('message_serialization', None)
    m['background'] = {'scripts': ['bridge-auth.js', 'background.js'], 'persistent': False}
    result['manifest.json'] = (json.dumps(m, indent=2) + '\n').encode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    drift = []
    for name, data in expected().items():
        target = TARGET / name
        if args.check:
            if not target.exists() or target.read_bytes() != data: drift.append(name)
        else: target.write_bytes(data)
    if drift: raise SystemExit('Safari resource drift: ' + ', '.join(drift))
    print('Safari resources match the shared browser implementation')


if __name__ == '__main__': main()
