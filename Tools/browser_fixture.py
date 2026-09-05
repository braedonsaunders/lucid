#!/usr/bin/env python3
"""Build an isolated browser fixture exercising the real companion scripts.

Runtime ports use real MessageChannels. This verifies serialization and iframe
presentation, but is not an extension installation or third-party CSP test.
"""
import argparse
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.build/browser-test')
    parser.add_argument('--clip', default='TestSite/crowdrun-360p-350k.mp4')
    parser.add_argument('--bridge-port', type=int, default=47911)
    parser.add_argument('--token-port', type=int, default=47912)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for folder in [ROOT / 'BrowserExtension', ROOT / 'Tools/browser-fixture']:
        for path in folder.iterdir():
            if path.suffix not in ('.js', '.html'): continue
            content = path.read_text().replace('47811', str(args.bridge_port)).replace('47812', str(args.token_port))
            if folder.name == 'browser-fixture':
                content = content.replace('47911', str(args.bridge_port)).replace('47912', str(args.token_port))
            (out / path.name).write_text(content)
    shutil.copyfile(args.clip, out / 'video.mp4')
    print(f'Fixture ready: {out.resolve()}')
if __name__ == '__main__': main()
