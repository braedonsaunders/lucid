#!/usr/bin/env python3
"""Score actual Lucid NV12 delivery packets from the diagnostic H264 fixture.

Packet YUV is decoded with explicit Rec.709/range metadata. This evaluates the
native sender output, not Chrome's rendering or monitor color management.
"""
import argparse
import io
import json
from pathlib import Path
import struct
import subprocess

from PIL import Image
import torch
from evaluate_sequences import Scorer, decode, digest, save, summarize


def packet_image(path):
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != b'LUCE':
        raise ValueError('invalid Lucid packet')
    size = struct.unpack('>I', data[4:8])[0]
    header = json.loads(data[8:8+size])
    w, h = header['w'], header['h']
    color = header['colorSpace']
    if (header['format'] != 'NV12' or w <= 0 or h <= 0 or w%2 or h%2
        or len(data)-8-size != w*h*3//2):
        raise ValueError('invalid NV12 payload geometry')
    if any(color.get(k) != 'bt709' for k in ['primaries','transfer','matrix']) or not isinstance(color.get('fullRange'),bool):
        raise ValueError('this diagnostic scorer requires explicit Rec.709 SDR color metadata')
    result = subprocess.run(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','nv12',
        '-video_size',f'{w}x{h}','-color_range','pc' if color['fullRange'] else 'tv',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-i','pipe:0',
        '-frames:v','1','-vf',f"scale=in_color_matrix=bt709:out_range=pc:in_range={'pc' if color['fullRange'] else 'tv'}",
        '-pix_fmt','rgb24','-f','image2pipe','-c:v','png','pipe:1'],
        input=data[8+size:],capture_output=True,check=True,timeout=30)
    with Image.open(io.BytesIO(result.stdout)) as image:
        return image.convert('RGB').copy(),header


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--device',default='mps')
    args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text())
    sources={}
    scorer=Scorer(torch.device(args.device))
    report={'purpose':'native delivered H264 development spatial screen; no browser or release claim',
        'manifest_sha256':digest(args.manifest),'code_sha256':digest(__file__),
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'device':args.device,'torch':str(torch.__version__),'rows':[],'complete':False}
    for row in manifest['rows']:
        path=Path(row['packet'])
        if digest(path) != row['sha256']:
            raise ValueError('native packet changed')
        output,header=packet_image(path)
        if header['seq']%16 != row['frame'] or header['seq'] != int(path.stem):
            raise ValueError('diagnostic frame mapping differs')
        if row['reference'] not in sources:
            sources[row['reference']]=decode(row['reference'],16)
        reference=sources[row['reference']][row['frame']]
        report['rows'].append({**row,'header':header,'reference_sha256':digest(row['reference']),
            'metrics':scorer.spatial(output,reference)})
        report['summary']=summarize(report['rows']);save(args.report,report)
        print(row['sequence_id'],row['variant'],row['frame'],flush=True)
    report['complete']=True;save(args.report,report)


if __name__=='__main__':main()
