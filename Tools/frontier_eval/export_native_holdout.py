#!/usr/bin/env python3
"""Run the frozen native candidate and shipping on identical decoded NV12 clips.

All consecutive frames traverse native preprocessing/reconstruction/detail;
only fixed spatial positions are exported through the real NV12 packet sender.
Reference PNGs come from the already frozen decoded-pixel holdout manifest.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','frozen-frames','native-config','models','executable','out']:
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh output directory required')
    sequences=json.loads(args.manifest.read_text())
    frames=json.loads((args.frozen_frames/'manifest.json').read_text())
    config=json.loads(args.native_config.read_text())
    if frames['sequence_manifest_sha256']!=digest(args.manifest) or frames['split']!='quality-holdout':
        raise ValueError('frozen sequence/frame manifests differ')
    refs={(r['sequence_id'],r['frame']):r for r in frames['frames'] if r['side']=='reference'}
    # Core ML metadata must identify the exact frozen weight pair.
    import coremltools as ct
    packages={'shipping4x':args.models/'shipping4x_640x360.mlpackage',
              'candidate':args.models/'direct2x_trained_640x360.mlpackage'}
    expected={'shipping4x':'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65',
              'candidate':config['candidate_sha256']}
    for label,package in packages.items():
        model=ct.models.MLModel(str(package),skip_model_load=True)
        if model.user_defined_metadata.get('lucid.checkpoint_sha256')!=expected[label]:
            raise ValueError('Core ML package does not identify frozen checkpoint')
    args.out.mkdir(parents=True)
    tuning=args.out/'candidate-tuning.json';tuning.write_text(json.dumps(config['tuning'],indent=2)+'\n')
    report={'purpose':'frozen native spatial holdout; no browser cadence or release claim',
        'split':'quality-holdout','complete':False,'rows':[],'input_streams':[],
        'source_manifest_sha256':digest(args.manifest),'frozen_frames_manifest_sha256':digest(args.frozen_frames/'manifest.json'),
        'native_config_sha256':digest(args.native_config),'checkpoint_sha256':expected,
        'executable_sha256':digest(args.executable),'code_sha256':digest(__file__),
        'models':{label:{str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file()} for label,package in packages.items()},
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'configuration':'Standard shipping sharpness0.75/radius4; frozen candidate sharpness0.4/radius2. Fixed grain phase0.',
        'limitations':['FFmpeg decode boundary instead of Chrome WebCodecs; byte-identical NV12 inputs for model pair',
            'Native sender packets; browser rendering and monitor color management excluded',
            'Absent SDR primaries/transfer and chroma-location tags use the frozen corpus contract and existing Lucid defaults; other color spaces are not tested',
            'Raw-weight holdout already completed before this native run; native config frozen before raw-weight results were inspected']}
    def save():
        (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    for sequence in sequences:
        if digest(sequence['degraded'])!=sequence['degraded_sha256']:raise ValueError('source clip changed')
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries,chroma_location',
            '-of','json',sequence['degraded']],text=True))['streams'][0]
        if (stream['width'],stream['height'],stream['pix_fmt'],stream.get('color_range'),stream.get('color_space'),
            stream.get('color_transfer','bt709'),stream.get('color_primaries','bt709')) != (640,360,'yuv420p','tv','bt709','bt709','bt709'):
            raise ValueError('source does not match the explicit Rec.709 420 contract')
        if stream.get('chroma_location','left')!='left':raise ValueError('unsupported chroma siting')
        raw=args.out/'working.nv12'
        command=['ffmpeg','-nostdin','-v','error','-i',sequence['degraded'],'-frames:v',str(sequence['frames']),
            '-fps_mode','passthrough','-pix_fmt','nv12','-f','rawvideo',str(raw)]
        subprocess.run(command,check=True,timeout=120)
        rate=Fraction(sequence['frame_rate'])
        descriptor={'format':'NV12','width':640,'height':360,'frames':sequence['frames'],
            'fpsNumerator':rate.numerator,'fpsDenominator':rate.denominator,'data':raw.name,'sha256':digest(raw),
            'colorSpace':{'primaries':'bt709','transfer':'bt709','matrix':'bt709','fullRange':False},
            'chromaLocation':'left','spatialStride':frames['stride'],'exportWarmup':True}
        descriptor_path=args.out/'working.json';descriptor_path.write_text(json.dumps(descriptor,indent=2)+'\n')
        report['input_streams'].append({'sequence_id':sequence['id'],'descriptor':descriptor,'decode_command':command,'probe':stream,
            'metadata_defaults':{key:value for key,value in [('color_primaries','bt709'),('color_transfer','bt709'),('chroma_location','left')] if key not in stream},
            'metadata_default_basis':'Frozen SDR Rec709 corpus contract and existing native left-chroma policy; no explicit contradictory tags accepted'})
        for label,package in packages.items():
            target=args.out/(sequence['id']+'-'+label)
            # Inherited tuning overrides must not alter either frozen arm.
            env={k:v for k,v in os.environ.items() if not k.startswith('LUCID_')}
            env.update(LUCID_COMPUTE_UNITS='gpu',LUCID_PIPELINE_MODEL=str(package.resolve()),LUCID_PIPELINE_PACKETS=str(target.resolve()))
            if label=='candidate':env['LUCID_TUNING']=str(tuning.resolve())
            result=subprocess.run([str(args.executable.resolve()),'--pipeline-ms',str(descriptor_path.resolve()),str(sequence['frames']-8)],
                env=env,capture_output=True,text=True,timeout=180)
            (args.out/(target.name+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
            actual_tuning=json.loads((target/'tuning.json').read_text())
            expected_tuning=dict(config['tuning'],sharpness=.75 if label=='shipping4x' else .4)
            if actual_tuning!=expected_tuning:raise ValueError('actual native tuning differs from frozen arm')
            packets=sorted(target.glob('????????.luce'))
            indices=list(range(0,sequence['frames'],frames['stride']))
            if [int(p.stem) for p in packets]!=indices:raise ValueError('native sample positions differ')
            for packet,index in zip(packets,indices):
                data=packet.read_bytes();size=struct.unpack('>I',data[4:8])[0];header=json.loads(data[8:8+size])
                if data[:4]!=b'LUCE' or (header['w'],header['h'],header['seq'])!=(1280,720,index):
                    raise ValueError('unexpected native packet geometry/index')
                reference=refs[sequence['id'],index]
                if digest(args.frozen_frames/reference['file'])!=reference['sha256']:raise ValueError('reference PNG changed')
                report['rows'].append({'source_id':sequence['source_id'],'sequence_id':sequence['id'],'frame':index,
                    'variant':label,'packet':str(packet.relative_to(args.out)),'sha256':digest(packet),
                    'reference':reference['file'],'reference_sha256':reference['sha256']})
            save();print(sequence['id'],label,len(packets),'packets',flush=True)
        raw.unlink() # Only this utility's decoded scratch file, reproducible from pinned source/command.
    report['complete']=True;save()


if __name__=='__main__':main()
