#!/usr/bin/env python3
"""Freeze native sender packets as decoded RGB PNGs before cross-host scoring."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packets',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh RGB frame directory required')
    source=args.packets/'manifest.json';manifest=json.loads(source.read_text())
    if not manifest['complete']:raise ValueError('complete packet capture required')
    groups=defaultdict(list)
    for row in manifest['rows']:groups[row['sequence_id'],row['variant']].append(row)
    args.out.mkdir(parents=True)
    result={k:v for k,v in manifest.items() if k!='rows'}
    result.update(packet_manifest_sha256=digest(source),decoder_sha256=digest(__file__),
        decoder_ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],rows=[],complete=False)
    for (sequence,label),rows in groups.items():
        rows.sort(key=lambda r:r['frame']);payload=bytearray()
        for row in rows:
            path=args.packets/row['packet'];data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha256'] or data[:4]!=b'LUCE':raise ValueError('packet changed')
            size=struct.unpack('>I',data[4:8])[0];header=json.loads(data[8:8+size]);pixels=data[8+size:]
            if (header['w'],header['h'],header['seq'],header['format'])!=(1280,720,row['frame'],'NV12'):
                raise ValueError('invalid packet geometry/position/format')
            if header['colorSpace']!={'primaries':'bt709','transfer':'bt709','matrix':'bt709','fullRange':False}:
                raise ValueError('explicit Rec709 video-range packet required')
            if len(pixels)!=1280*720*3//2:raise ValueError('invalid native packet size')
            payload.extend(pixels)
        target=args.out/(sequence+'-'+label);target.mkdir()
        command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','nv12','-video_size','1280x720',
            '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-i','pipe:0',
            '-frames:v',str(len(rows)),'-vf','scale=in_color_matrix=bt709:out_range=pc:in_range=tv',
            '-pix_fmt','rgb24','-threads','2',str(target/'%03d.png')]
        subprocess.run(command,input=payload,check=True,timeout=120)
        images=sorted(target.glob('*.png'))
        if len(images)!=len(rows):raise ValueError('incomplete RGB export')
        for image,row in zip(images,rows):
            result['rows'].append({**row,'file':str(image.relative_to(args.out)),'image_sha256':digest(image)})
        print(sequence,label,'RGB frozen',flush=True)
    result['complete']=True
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
