#!/usr/bin/env python3
"""Make lossless 24fps masters from checked REDS PNGs and verify every RGB frame.

The first two source IDs in the seeded selection are development validation;
all remaining IDs are training. These are sequence-disjoint development splits,
not a claim of independently verified filming-family or upstream model splits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh master directory required')
    manifest_path=args.frames/'frame-receipts.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest['complete'] or len(manifest['spec']['sequences'])<3:
        raise ValueError('completed corpus with at least three sequences required')
    for row in manifest['frames']:
        if digest(args.frames/row['file'])!=row['sha256']:raise ValueError('PNG changed')
    args.out.mkdir(parents=True)
    sources=[];receipts=[]
    for index,sequence in enumerate(manifest['spec']['sequences']):
        selected=sorted([r for r in manifest['frames'] if r['source_id']=='reds_'+sequence],key=lambda r:r['frame'])
        if [r['frame'] for r in selected]!=list(range(manifest['spec']['frames_per_sequence'])):
            raise ValueError('complete consecutive source required')
        output=args.out/('reds_'+sequence+'.mkv')
        command=['ffmpeg','-nostdin','-v','error','-framerate','24','-start_number','0','-i',
            str(args.frames/sequence/'%08d.png'),'-frames:v',str(len(selected)),
            '-threads','2','-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(output)]
        subprocess.run(command,check=True,timeout=180)
        process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(output),
            '-threads','2','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE)
        sequence_hash=hashlib.sha256()
        try:
            for row in selected:
                with Image.open(args.frames/row['file']) as image:expected=image.convert('RGB').tobytes()
                actual=bytearray()
                while len(actual)<len(expected):
                    block=process.stdout.read(len(expected)-len(actual))
                    if not block:raise ValueError('short lossless decode')
                    actual.extend(block)
                if actual!=expected:raise ValueError('RGB mismatch in lossless source master')
                sequence_hash.update(actual)
            if process.stdout.read(1):raise ValueError('extra decoded frames')
            if process.wait(timeout=30):raise RuntimeError('lossless decoder failed')
        finally:
            process.stdout.close()
            if process.poll() is None:process.kill();process.wait()
        split='validation' if index<2 else 'train'
        sources.append({'id':'reds_'+sequence,'family':'reds_sequence_'+sequence,
            'split':split,'path':str(output.resolve()),'sha256':digest(output),
            'license':'CC BY 4.0','origin':manifest['spec']['url']})
        receipts.append({'id':'reds_'+sequence,'frames':len(selected),'sha256':digest(output),
            'rgb_sequence_sha256':sequence_hash.hexdigest(),'command':command})
        print(sequence,split,'all RGB frames exact',flush=True)
    (args.out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    (args.out/'master-receipts.json').write_text(json.dumps({'sources':receipts,
        'frame_receipts_sha256':digest(manifest_path),'code_sha256':digest(__file__),
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'split_rule':'First two seeded selected sequences are development validation; others train.',
        'limitation':'Source family is identified only at sequence level; no independent camera/session-family verification.'},indent=2)+'\n')


if __name__=='__main__':main()
