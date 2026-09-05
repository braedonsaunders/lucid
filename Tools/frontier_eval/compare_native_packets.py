#!/usr/bin/env python3
"""Compare native sender payloads and parsed headers without JSON key-order noise."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packets', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh report required')
    raw = (args.packets / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if not manifest['complete']:
        raise ValueError('complete capture required')
    groups = {}
    for row in manifest['rows']:
        key = row['sequence_id'], row['frame']
        group = groups.setdefault(key, {})
        if row['variant'] not in ('shipping4x', 'candidate') or row['variant'] in group:
            raise ValueError('unexpected or duplicate arm')
        data = (args.packets / row['packet']).read_bytes()
        if digest(data) != row['sha256'] or data[:4] != b'LUCE':
            raise ValueError('packet changed')
        size = struct.unpack('>I', data[4:8])[0]
        header, pixels = json.loads(data[8:8+size]), data[8+size:]
        if header['format'] != 'NV12' or len(pixels) != header['w'] * header['h'] * 3 // 2:
            raise ValueError('invalid payload')
        group[row['variant']] = header, pixels
    rows = []
    for (sequence, frame), pair in sorted(groups.items()):
        if set(pair) != {'shipping4x', 'candidate'}:
            raise ValueError('incomplete pair')
        a, b = pair['shipping4x'], pair['candidate']
        rows.append(dict(sequence_id=sequence, frame=frame,
                         headers_equal=a[0] == b[0], payloads_equal=a[1] == b[1],
                         shipping_payload_sha256=digest(a[1]), candidate_payload_sha256=digest(b[1])))
    if not rows:
        raise ValueError('empty capture')
    report = dict(complete=True, pairs=len(rows),
                  exact_pairs=sum(r['headers_equal'] and r['payloads_equal'] for r in rows),
                  manifest_sha256=digest(raw), code_sha256=digest(Path(__file__).read_bytes()), rows=rows,
                  scope='Exact native paired output check; no browser cadence or source-general quality claim')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Exact native pairs: {report['exact_pairs']}/{report['pairs']}")


if __name__ == '__main__':
    main()
