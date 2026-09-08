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
    ap.add_argument('--development-presentation', action='store_true',
                    help='Use the fixed 48-pair development screen for the quantized shipping presentation')
    ap.add_argument('--presentation-corpus', choices=['development48', 'regression960'], default='development48',
                    help='Previously used, fixed source sets; neither is a fresh holdout')
    ap.add_argument('--tensor-output', action='store_true', help='Fixed FP32 full-4x output-storage regression; Standard radius4 in both arms')
    ap.add_argument('--preserve-display-gain', action='store_true',
                    help='Diagnostic: keep nominal gain when the presentation candidate changes output scale')
    ap.add_argument('--candidate-only', action='store_true', help='Capture the candidate against an already frozen native baseline')
    ap.add_argument('--motion-policy', choices=['taa', 'gain_only', 'search'], default='taa',
                    help='Process-only candidate TAA policy; must match the frozen config')
    ap.add_argument('--use-native-default', action='store_true',
                    help='Do not override candidate motion policy; still require the frozen expected policy in native logs')
    args=ap.parse_args()
    if args.preserve_display_gain and not args.development_presentation:
        ap.error('--preserve-display-gain requires --development-presentation')
    if args.presentation_corpus != 'development48' and not args.development_presentation:
        ap.error('--presentation-corpus requires --development-presentation')
    if args.tensor_output and (args.preserve_display_gain or not args.development_presentation):
        ap.error('--tensor-output requires development presentation and forbids gain modification')
    if args.out.exists():ap.error('fresh output directory required')
    sequences=json.loads(args.manifest.read_text())
    frames=json.loads((args.frozen_frames/'manifest.json').read_text())
    config=json.loads(args.native_config.read_text())
    if config.get('motion_policy', 'taa') != args.motion_policy:
        raise ValueError('motion policy differs from frozen config')
    development = args.development_presentation
    if development and (config['candidate_sha256']!='fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65' or config['radius']!=(4 if args.tensor_output else 2)):
        raise ValueError('development presentation requires unchanged shipping weights and declared output scale')
    if args.tensor_output and (config.get('output_storage') != 'tensor4x_fp32' or config['tuning']['sharpness'] != .75):
        raise ValueError('tensor-output requires frozen FP32 storage and Standard gain')
    corpus_hashes = {
        'development48': 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f',
        'regression960': '307d6670ef1c4918798172a16292dd55ade37db61d953efeddb5d60e21154b3b',
    }
    if development and digest(args.frozen_frames/'manifest.json')!=corpus_hashes[args.presentation_corpus]:
        raise ValueError('fixed development inputs required')
    if frames['sequence_manifest_sha256']!=digest(args.manifest) or (not development and frames['split']!='quality-holdout'):
        raise ValueError('frozen sequence/frame manifests differ')
    refs={(r['sequence_id'],r['frame']):r for r in frames['frames'] if r['side']=='reference'}
    # Core ML metadata must identify the exact frozen weight pair.
    import coremltools as ct
    packages={'shipping4x':args.models/('image4x.mlpackage' if args.tensor_output else 'shipping4x_640x360.mlpackage'),
              'candidate':args.models/('tensor4x_fp32.mlpackage' if args.tensor_output else ('quantized_bicubic2x_640x360.mlpackage' if development else 'direct2x_trained_640x360.mlpackage'))}
    expected={'shipping4x':'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65',
              'candidate':config['candidate_sha256']}
    if args.candidate_only:
        packages = {'candidate': packages['candidate']}
        expected = {'candidate': expected['candidate']}
    for label,package in packages.items():
        model=ct.models.MLModel(str(package),skip_model_load=True)
        if model.user_defined_metadata.get('lucid.checkpoint_sha256')!=expected[label]:
            raise ValueError('Core ML package does not identify frozen checkpoint')
        spec=model.get_spec().description
        image_in=next(x.type.imageType for x in spec.input if x.name=='input')
        output=next(x for x in spec.output if x.name=='output')
        scale=4 if label=='shipping4x' or args.tensor_output else 2
        if args.tensor_output and label=='candidate':
            tensor=output.type.multiArrayType
            if (output.type.WhichOneof('Type') != 'multiArrayType'
                    or list(tensor.shape) != [1,3,1440,2560]
                    or tensor.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
                    or model.user_defined_metadata.get('lucid.output_range') != '0..255'
                    or model.user_defined_metadata.get('lucid.output_scale') != '4'):
                raise ValueError('fixed FP32 tensor output contract differs')
        elif (output.type.WhichOneof('Type') != 'imageType'
              or (output.type.imageType.width,output.type.imageType.height)!=(640*scale,360*scale)):
            raise ValueError('Core ML image output geometry differs from frozen arm')
        if (image_in.width,image_in.height)!=(640,360):
            raise ValueError('Core ML input geometry differs from frozen arm')
    args.out.mkdir(parents=True)
    tuning=args.out/'candidate-tuning.json';tuning.write_text(json.dumps(config['tuning'],indent=2)+'\n')
    # The shipping arm is the frozen SPAN 4x comparator at its own measured gain (0.75); pin it
    # explicitly so the app's current defaults (now the lucid2k_ family at 0.2) cannot change it.
    shipping_tuning=args.out/'shipping-tuning.json';shipping_tuning.write_text(json.dumps(dict(config['tuning'],sharpness=.75),indent=2)+'\n')
    report={'purpose':'frozen native spatial holdout; no browser cadence or release claim',
        'split':'development-presentation' if development else 'quality-holdout','complete':False,'rows':[],'input_streams':[],
        'candidate_only': args.candidate_only, 'motion_policy': args.motion_policy,
        'candidate_uses_native_default': args.use_native_default,
        'source_manifest_sha256':digest(args.manifest),'frozen_frames_manifest_sha256':digest(args.frozen_frames/'manifest.json'),
        'native_config_sha256':digest(args.native_config),'checkpoint_sha256':expected,
        'executable_sha256':digest(args.executable),'code_sha256':digest(__file__),
        'models':{label:{str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file()} for label,package in packages.items()},
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'configuration':f"Standard shipping sharpness0.75/radius4; frozen candidate sharpness{config['tuning']['sharpness']}/radius{config['radius']}. Grain phase{config['tuning']['grainPhase']}.",
        'limitations':['FFmpeg decode boundary instead of Chrome WebCodecs; byte-identical NV12 inputs for model pair',
            'Native sender packets; browser rendering and monitor color management excluded',
            'Absent SDR primaries/transfer and chroma-location tags use the frozen corpus contract and existing Lucid defaults; other color spaces are not tested',
            'Repeated regression sources, not fresh release validation; tuning and weight identities are bound to the frozen config, and raw-model scores do not substitute for native results']}
    if development:
        report['purpose']='native presentation development regression; no fresh holdout or promotion'
        report['configuration']=f"Standard shipping sharpness0.75/radius4; quantized presentation sharpness{config['tuning']['sharpness']}/radius2. Fixed grain phase0."
        report['limitations'][-1]='Repeated development sources; unchanged weights, standard postprocessing and real sender compared before product integration'
        report['preserve_display_gain']=args.preserve_display_gain
        report['presentation_corpus']=args.presentation_corpus
    if args.tensor_output:
        report['configuration']='Identical Standard sharpness0.75/radius4, full 4x RGB8/NV12/detail and 2x sender in both arms. Fixed grain phase0.'
        report['output_storage']='tensor4x_fp32'
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
            env['LUCID_TUNING']=str((tuning if label=='candidate' else shipping_tuning).resolve())
            policy = ['taa', 'gain_only', 'search'].index(args.motion_policy) if label == 'candidate' else 0
            if label != 'candidate' or not args.use_native_default:
                env['LUCID_TAA_MOTION_POLICY'] = str(policy)
            if label=='candidate' and args.preserve_display_gain:env['LUCID_PIPELINE_PRESERVE_GAIN']='1'
            result=subprocess.run([str(args.executable.resolve()),'--pipeline-ms',str(descriptor_path.resolve()),str(sequence['frames']-8)],
                env=env,capture_output=True,text=True,timeout=180)
            (args.out/(target.name+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
            if f'pipeline-ms detail motionPolicy={policy}' not in result.stdout:
                raise ValueError('actual native motion policy differs from frozen arm')
            if args.preserve_display_gain:
                reference_radius=2 if label=='candidate' else 4
                if f'pipeline-ms detail referenceRadius={reference_radius}' not in result.stdout:
                    raise ValueError('diagnostic gain reference radius differs')
            if args.tensor_output and ('pipeline-ms detail radius=4' not in result.stdout or 'pipeline-ms detail referenceRadius=4' not in result.stdout):
                raise ValueError('tensor route must retain full 4x detail geometry and gain')
            actual_tuning=json.loads((target/'tuning.json').read_text())
            expected_tuning=dict(config['tuning'],sharpness=.75 if label=='shipping4x' else config['tuning']['sharpness'])
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
