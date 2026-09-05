#!/usr/bin/env python3
"""Isolated VFX image-content screen; includes every declared mode, never a fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

FRAMES = 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'
WHEEL = 'b6cfaff5f435ad18329a1e1c1ac3ceb36f2aa6cfb0774d271c0bcc3aeaf31c53'
SHIPPING = 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65'
MODES = ('BICUBIC', 'HIGH', 'ULTRA')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def worker(spec_path):
    # The alpha SDK blocks in destroy() on this host. Each image owns a process;
    # synchronize and persist CPU artifacts, then let process exit release CUDA.
    # This is benchmark isolation, not a proposed production integration.
    spec = json.loads(Path(spec_path).read_text())
    receipt = dict(complete=False, mode=spec['mode'])
    try:
        import numpy as np
        from PIL import Image
        import torch
        import nvvfx
        from nvvfx.effects import QualityLevel
        if digest(spec['input']) != spec['input_sha256']:
            raise ValueError('input changed')
        source = np.array(Image.open(spec['input']).convert('RGB'))
        if source.shape != (360, 640, 3): raise ValueError('input geometry changed')
        tensor = torch.from_numpy(source).permute(2,0,1).float().div(255).contiguous().cuda()
        sr = nvvfx.VideoSuperRes(getattr(QualityLevel,spec['mode']))
        sr.output_width=1280; sr.output_height=720
        started=time.perf_counter(); sr.load(); torch.cuda.synchronize()
        receipt['load_ms']=(time.perf_counter()-started)*1000
        started=time.perf_counter()
        output=torch.from_dlpack(sr.run(tensor).image).clone()
        torch.cuda.synchronize()
        receipt['first_run_clone_ms']=(time.perf_counter()-started)*1000
        if tuple(output.shape)!=(3,720,1280) or not torch.isfinite(output).all():
            raise ValueError('invalid SDK output')
        lo,hi,std=float(output.min()),float(output.max()),float(output.std())
        if lo<0 or hi>1.0001 or hi<.1 or std<.001:
            raise ValueError('black, constant or unexpected-range SDK output')
        pixels=output.clamp(0,1).mul(255).round().byte().permute(1,2,0).contiguous().cpu().numpy()
        Image.fromarray(pixels).save(spec['output'])
        receipt.update(complete=True, output_sha256=digest(spec['output']),
            minimum=lo, maximum=hi, std=std, package=nvvfx.__version__,
            sdk_version=str(nvvfx.get_sdk_version()),torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(), cleanup='synchronized fresh worker process exit; SDK destroy excluded because of reproduced hang')
        save(spec['receipt'],receipt)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    except BaseException:
        receipt['error']=traceback.format_exc(); save(spec['receipt'],receipt)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',type=Path)
    for key in ('frames','shipping','installation','out'):
        ap.add_argument('--'+key,type=Path)
    args=ap.parse_args()
    if args.worker: worker(args.worker); return
    if any(getattr(args,k) is None for k in ('frames','shipping','installation','out')):
        ap.error('frames, shipping, installation, out required')
    if args.out.exists(): ap.error('fresh output required')
    if digest(args.frames/'manifest.json')!=FRAMES: raise ValueError('fixed development frames required')
    installation=json.loads(args.installation.read_text())
    if installation['sha256']!=WHEEL: raise ValueError('official wheel identity changed')
    sdk_root=args.installation.parent/'deps'
    for relative,sha in installation['files'].items():
        if digest(sdk_root/relative)!=sha: raise ValueError('SDK file changed')
    source=json.loads((args.frames/'manifest.json').read_text())
    baseline=json.loads((args.shipping/'report.json').read_text())
    if (not baseline['complete'] or baseline['manifest_sha256']!=FRAMES
            or baseline['shipping_sha256']!=SHIPPING): raise ValueError('shipping baseline identity changed')
    refs={(r['sequence_id'],r['frame']):r for r in source['frames'] if r['side']=='reference'}
    inputs=[r for r in source['frames'] if r['side']=='degraded']
    shipping={(r['sequence_id'],r['frame']):r for r in baseline['output_pixels'] if r['variant']=='shipping'}
    if len(inputs)!=48 or len(shipping)!=48: raise ValueError('wrong development coverage')
    for row in source['frames']:
        if digest(args.frames/row['file'])!=row['sha256']: raise ValueError('decoded pixels changed')
    for row in shipping.values():
        if digest(args.shipping/row['output_file'])!=row['output_sha256']: raise ValueError('shipping output changed')
    args.out.mkdir(parents=True)
    report=dict(complete=False, inference_complete=False, modes=list(MODES), split='development-rgb48',
        manifest_sha256=FRAMES, shipping_report_sha256=digest(args.shipping/'report.json'),
        installation_sha256=digest(args.installation), code_sha256=digest(__file__), outputs=[], rows=[],
        limitations=['SDK RGB image screen, not browser-driver VSR or temporal evaluation',
            'Fresh process per image due verified SDK teardown hang; cold run times are not steady video latency',
            'Lucid comparator is native Core ML RGB4x plus fixed PIL bicubic presentation, excludes native NV12/detail/browser',
            'Repeated development sources; external model training overlap unknown; no release promotion'])
    save(args.out/'report.json',report)
    for row in inputs:
        key=(row['sequence_id'],row['frame'])
        for mode in MODES:
            stem=f"{row['sequence_id']}-{row['frame']:03d}-{mode}"
            image_path=args.out/(stem+'.png'); receipt_path=args.out/(stem+'.json')
            spec_path=args.out/(stem+'-input.json')
            spec=dict(mode=mode,input=str((args.frames/row['file']).resolve()),input_sha256=row['sha256'],
                output=str(image_path.resolve()),receipt=str(receipt_path.resolve()))
            save(spec_path,spec)
            started=time.perf_counter()
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(spec_path.resolve())],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try: stdout,stderr=child.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=False,capture_output=True)
                stdout,stderr=child.communicate(); raise RuntimeError('isolated SDK worker timed out')
            (args.out/(stem+'.log')).write_bytes(stdout+stderr)
            if child.returncode: raise RuntimeError('SDK worker failed; no scaler fallback')
            receipt=json.loads(receipt_path.read_text())
            if not receipt['complete'] or digest(image_path)!=receipt['output_sha256']:
                raise ValueError('incomplete or changed SDK output')
            report['outputs'].append(dict(row,variant='nvvfx_'+mode,file=image_path.name,
                image_sha256=receipt['output_sha256'],receipt=receipt_path.name,
                process_wall_seconds=time.perf_counter()-started))
            save(args.out/'report.json',report); print(stem,'frozen',flush=True)
    report['inference_complete']=True; save(args.out/'report.json',report)
    from PIL import Image
    import numpy as np
    import torch
    from evaluate_sequences import Scorer,summarize
    scorer=Scorer(torch.device('cuda'))
    outputs={(r['sequence_id'],r['frame'],r['variant']):r for r in report['outputs']}
    for row in inputs:
        key=row['sequence_id'],row['frame']
        reference=Image.open(args.frames/refs[key]['file']).convert('RGB')
        variants={'shipping':Image.open(args.shipping/shipping[key]['output_file']).convert('RGB'),
                  'lanczos':Image.open(args.frames/row['file']).convert('RGB').resize((1280,720),Image.Resampling.LANCZOS)}
        for mode in MODES:
            variant='nvvfx_'+mode; variants[variant]=Image.open(args.out/outputs[(*key,variant)]['file']).convert('RGB')
        bicubic=np.asarray(variants['nvvfx_BICUBIC']).astype(np.int16)
        for mode in ('HIGH','ULTRA'):
            mae=float(np.abs(np.asarray(variants['nvvfx_'+mode]).astype(np.int16)-bicubic).mean())
            if mae<=1e-6: raise ValueError('AI output indistinguishable from bicubic; refuse fallback')
        for variant,image in variants.items():
            report['rows'].append(dict(source_id=row['source_id'],sequence_id=key[0],frame=key[1],variant=variant,
                metrics=scorer.spatial(image,reference)))
        save(args.out/'report.json',report);print(key,'scored',flush=True)
    report.update(complete=True, summary=summarize(report['rows']),scoring_device='cuda',torch=str(torch.__version__),
                  scorer_sha256=digest(Path(__file__).with_name('evaluate_sequences.py')))
    save(args.out/'report.json',report)


if __name__=='__main__': main()
