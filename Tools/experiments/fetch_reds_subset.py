#!/usr/bin/env python3
"""Fetch selected official REDS training frames from a pinned ZIP via HTTP ranges.

Never fetch REDS4 evaluation sequences. Every ZIP entry is checked against the
pinned central-directory index, including name, decoded size and CRC; completed
PNG files and receipts have SHA256 identities. No GPU or external code runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import struct
import time
import urllib.request
import zlib

REVISION='62dc25d16e6f43d2214f1b365023abda86f7a0ae'
URL=f'https://huggingface.co/datasets/snah/REDS/resolve/{REVISION}/train_sharp.zip'
ARCHIVE_SIZE=34261573976
ARCHIVE_SHA256='620294c1c3f23ed26c5ea228633770469c0b28e57d31eb41dc77deb401c6681b'
EXCLUDED={'000','011','015','020'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_range(start,length):
    # Distinct range URLs prevent intermediary caches returning another range.
    url=URL+f'?lucid-range={start}-{length}'
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}',
        'Accept-Encoding':'identity','User-Agent':'Lucid-training-corpus/1.0'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request,timeout=45) as response:
                expected=f'bytes {start}-{start+length-1}/{ARCHIVE_SIZE}'
                if response.status!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError('server did not honor exact bounded ZIP range')
                data=response.read(length+1)
                if len(data)!=length:raise ValueError('incomplete ZIP range')
                return data
        except (OSError,TimeoutError):
            if attempt==3:raise
            time.sleep(2**attempt)


def unpack_entry(row,header,body):
    if len(header)!=30 or header[:4]!=b'PK\x03\x04':raise ValueError('invalid local ZIP header')
    _,version,flags,method,mtime,mdate,crc,csize,size,nlen,xlen=struct.unpack('<IHHHHHIIIHH',header)
    if flags&1 or method!=row['compression']:raise ValueError('unsupported ZIP entry')
    name=body[:nlen].decode('utf-8')
    if name!=row['name'] or len(body)!=nlen+xlen+row['compressed_size']:
        raise ValueError('ZIP entry identity/size mismatch')
    compressed=body[nlen+xlen:]
    data=zlib.decompress(compressed,-15) if method==8 else compressed if method==0 else None
    if data is None or len(data)!=row['size'] or zlib.crc32(data)!=row['crc']:
        raise ValueError('ZIP entry decoded checksum mismatch')
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('PNG required')
    return data


def fetch(row,out):
    source=Path(row['name']).parent.name
    if source in EXCLUDED:raise ValueError('evaluation source is excluded from training')
    target=out/source/Path(row['name']).name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        data=target.read_bytes()
        if len(data)!=row['size'] or zlib.crc32(data)!=row['crc']:
            raise ValueError('existing frame changed; refusing overwrite')
    else:
        header=read_range(row['offset'],30)
        nlen,xlen=struct.unpack('<HH',header[26:30])
        body=read_range(row['offset']+30,nlen+xlen+row['compressed_size'])
        data=unpack_entry(row,header,body)
        partial=target.with_suffix('.partial.png');partial.write_bytes(data);partial.replace(target)
    return {'file':str(target.relative_to(out)),'source_id':'reds_'+source,
        'frame':int(target.stem),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
        'zip_crc32':row['crc'],'zip_offset':row['offset']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--sequences',type=int,default=8)
    ap.add_argument('--frames',type=int,default=64)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if not 1<=args.sequences<=60 or not 40<=args.frames<=100 or not 1<=args.workers<=4:
        ap.error('1..60 sequences, 40..100 frames, 1..4 workers required')
    index=json.loads(args.index.read_text())
    names=sorted({Path(r['name']).parent.name for r in index if r['name'].endswith('.png')}-EXCLUDED)
    selected=random.Random(20260905).sample(names,60)[:args.sequences]
    rows=[r for r in index if r['name'].endswith('.png') and Path(r['name']).parent.name in selected
          and int(Path(r['name']).stem)<args.frames]
    rows.sort(key=lambda r:r['name'])
    if len(rows)!=args.sequences*args.frames:raise ValueError('incomplete requested source frames')
    spec={'official_page':'https://seungjunnah.github.io/Datasets/reds.html','license':'CC BY 4.0',
        'url':URL,'revision':REVISION,'archive_size':ARCHIVE_SIZE,'archive_lfs_sha256':ARCHIVE_SHA256,
        'index_sha256':digest(args.index),'sequences':selected,'frames_per_sequence':args.frames,
        'excluded_evaluation_sequences':sorted(EXCLUDED),'selection_seed':20260905,
        'source_frame_rate':24,'frame_rate_source':'official REDS page identifies standard release as 24fps',
        'whole_archive_downloaded_or_hashed':False,'entry_validation':'pinned-index CRC32, exact size/name, local SHA256'}
    args.out.mkdir(parents=True,exist_ok=True)
    spec_path=args.out/'subset-spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:
        raise ValueError('preserve existing differently configured subset')
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    receipts=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for receipt in pool.map(lambda row:fetch(row,args.out),rows):
            receipts.append(receipt)
            if len(receipts)%16==0:print('verified',len(receipts),'of',len(rows),flush=True)
    manifest={'spec':spec,'code_sha256':digest(__file__),'frames':receipts,'complete':True}
    (args.out/'frame-receipts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('complete',len(receipts),'frames',flush=True)


if __name__=='__main__':main()
