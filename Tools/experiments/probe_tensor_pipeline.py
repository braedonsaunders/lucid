#!/usr/bin/env python3
"""Freeze identical fixture bytes for a native image/tensor pipeline equality probe."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('video','models','config','executable','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--frames',type=int,default=64)
    args=ap.parse_args()
    if args.out.exists() or not 16<=args.frames<=300:ap.error('fresh output and 16..300 frames required')
    config=json.loads(args.config.read_text())
    if config['radius']!=4 or config['tuning']['sharpness']!=.75 or config['output_storage']!='tensor4x_fp32':
        raise ValueError('fixed full4x Standard tensor contract required')
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,pix_fmt,color_range,color_space,color_transfer,color_primaries',
        '-of','json',str(args.video)],text=True))['streams'][0]
    width,height=stream['width'],stream['height']
    if (width,height) not in ((640,360),(864,480)):raise ValueError('fixed native fixture geometry required')
    for key,expected in [('color_range','tv'),('color_space','bt709'),('color_transfer','bt709'),('color_primaries','bt709')]:
        if stream.get(key,expected)!=expected:raise ValueError('fixture contradicts the explicit Rec709 contract')
    args.out.mkdir()
    raw=args.out/'input.nv12'
    command=['ffmpeg','-nostdin','-v','error','-i',str(args.video),'-frames:v',str(args.frames),
             '-fps_mode','passthrough','-pix_fmt','nv12','-f','rawvideo',str(raw)]
    subprocess.run(command,check=True,timeout=60)
    if raw.stat().st_size!=width*height*3//2*args.frames:raise ValueError('incomplete decoded fixture')
    rate=Fraction(stream['r_frame_rate'])
    descriptor=dict(format='NV12',width=width,height=height,frames=args.frames,
                    fpsNumerator=rate.numerator,fpsDenominator=rate.denominator,data=raw.name,sha256=digest(raw),
                    colorSpace=dict(primaries='bt709',transfer='bt709',matrix='bt709',fullRange=False),
                    chromaLocation='left',spatialStride=4,exportWarmup=True)
    (args.out/'input.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    tuning=args.out/'tuning.json';tuning.write_text(json.dumps(config['tuning'],indent=2)+'\n')
    report=dict(complete=False,rows=[],video_sha256=digest(args.video),stream=stream,descriptor=descriptor,
                decode_command=command,native_config_sha256=digest(args.config),executable_sha256=digest(args.executable),
                code_sha256=digest(__file__),models={},
                scope='Fixture-derived explicit Rec709 native equality probe; not a source-general quality evaluation')
    for label,name in [('shipping4x','image4x'),('candidate','tensor4x_fp32')]:
        model=args.models/(name+'.mlpackage');target=args.out/label
        report['models'][label]={str(p.relative_to(model)):digest(p) for p in model.rglob('*') if p.is_file()}
        if not report['models'][label]:raise ValueError('missing model')
        env={k:v for k,v in os.environ.items() if not k.startswith('LUCID_')}
        env.update(LUCID_COMPUTE_UNITS='gpu',LUCID_PIPELINE_MODEL=str(model.resolve()),
                   LUCID_PIPELINE_PACKETS=str(target.resolve()),LUCID_TUNING=str(tuning.resolve()))
        result=subprocess.run([str(args.executable.resolve()),'--pipeline-ms',str((args.out/'input.json').resolve()),str(args.frames-8)],
                              env=env,capture_output=True,text=True,timeout=120)
        (args.out/(label+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
        if 'pipeline-ms detail radius=4' not in result.stdout or 'pipeline-ms detail referenceRadius=4' not in result.stdout:
            raise ValueError('detail geometry changed')
        if json.loads((target/'tuning.json').read_text())!=config['tuning']:raise ValueError('tuning changed')
        packets=sorted(target.glob('????????.luce'))
        if [int(p.stem) for p in packets]!=list(range(0,args.frames,4)):raise ValueError('missing or reordered packets')
        report['rows'].extend(dict(source_id='browser-fixture',sequence_id=f'fixture-{width}x{height}',frame=int(p.stem),
                                  variant=label,packet=str(p.relative_to(args.out)),sha256=digest(p)) for p in packets)
        (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
