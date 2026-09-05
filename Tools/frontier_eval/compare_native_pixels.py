#!/usr/bin/env python3
"""Describe paired delivered RGB differences; not a perceptual admission gate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh report required')
    manifest=args.frames/'manifest.json';m=json.loads(manifest.read_text())
    if not m['complete'] or m.get('output_storage')!='tensor4x_fp32':
        raise ValueError('complete fixed tensor-output native export required')
    groups=defaultdict(dict)
    for row in m['rows']:
        key=row['source_id'],row['sequence_id'],row['frame']
        if row['variant'] in groups[key]:raise ValueError('duplicate frame')
        groups[key][row['variant']]=row
    histogram=defaultdict(lambda:np.zeros(256,dtype=np.int64));rows=[]
    for key,variants in groups.items():
        if set(variants)!={'shipping4x','candidate'}:raise ValueError('unpaired frame')
        pixels={}
        for name,row in variants.items():
            path=args.frames/row['file']
            if digest(path)!=row['image_sha256']:raise ValueError('native pixels changed')
            with Image.open(path) as im:pixels[name]=np.array(im.convert('RGB'),dtype=np.int16)
        if pixels['shipping4x'].shape!=(720,1280,3) or pixels['candidate'].shape!=(720,1280,3):
            raise ValueError('native presentation geometry changed')
        error=np.abs(pixels['candidate']-pixels['shipping4x'])
        counts=np.bincount(error.ravel(),minlength=256)
        histogram[key[0]]+=counts
        rows.append(dict(source_id=key[0],sequence_id=key[1],frame=key[2],
                         mean_rgb_levels=float(error.mean()),max_rgb_levels=int(error.max())))
    sources={}
    for source,counts in histogram.items():
        total=int(counts.sum())
        sources[source]=dict(histogram_channels=counts.tolist(),mean_rgb_levels=float(counts@np.arange(256)/total),
            channels_equal_fraction=float(counts[0]/total),channels_within_one_fraction=float(counts[:2].sum()/total),
            p99_rgb_levels=int(np.searchsorted(counts.cumsum(),total*.99)),
            max_rgb_levels=int(np.flatnonzero(counts)[-1]))
    report=dict(complete=True,frames=len(rows),rows=rows,sources=sources,manifest_sha256=digest(manifest),
        code_sha256=digest(__file__),scope='Absolute native RGB differences only; no reference quality, temporal or perceptual gate')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k!='histogram_channels'} for s,r in sources.items()},indent=2))


if __name__=='__main__':main()
