#!/usr/bin/env python3
"""Validate official VFX inference, including nonzero output and AI/bicubic distinction."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['BICUBIC','HIGH','ULTRA'], help='Isolate one mode when diagnosing SDK teardown')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    report = dict(complete=False, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='VFX SDK API only; not browser-driver RTX VSR parity or quality ranking', rows=[])
    def save(): args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    faulthandler.dump_traceback_later(30, repeat=True)
    def stage(value):
        report['stage']=value; save()
    try:
        stage('imports')
        import torch
        import nvvfx
        from nvvfx.effects import QualityLevel
        torch.manual_seed(20260905)
        frame = torch.rand(3,360,640,device='cuda',dtype=torch.float32)
        report.update(torch=str(torch.__version__), package=nvvfx.__version__,
                      sdk_version=str(nvvfx.get_sdk_version()), gpu=torch.cuda.get_device_name(),
                      input=[640,360], output=[1280,720])
        results = {}
        for mode in ([args.mode] if args.mode else ('BICUBIC','HIGH','ULTRA')):
            stage(mode+' construct')
            with nvvfx.VideoSuperRes(getattr(QualityLevel,mode)) as sr:
                sr.output_width=1280; sr.output_height=720
                stage(mode+' load')
                started=time.perf_counter(); sr.load(); torch.cuda.synchronize()
                stage(mode+' run')
                load_ms=(time.perf_counter()-started)*1000
                samples=[]
                for i in range(25):
                    torch.cuda.synchronize(); started=time.perf_counter()
                    # The SDK owns returned memory. Clone before the next invocation or close.
                    output=torch.from_dlpack(sr.run(frame).image).clone()
                    torch.cuda.synchronize()
                    if i>=5: samples.append((time.perf_counter()-started)*1000)
                    if tuple(output.shape)!=(3,720,1280) or not torch.isfinite(output).all():
                        raise ValueError('invalid output geometry or nonfinite samples')
                lo,hi,std=float(output.min()),float(output.max()),float(output.std())
                if lo<0 or hi>1.0001 or hi<.1 or std<.01:
                    raise ValueError('black, constant, or unexpected output range')
                results[mode]=output.cpu()
                report['rows'].append(dict(mode=mode,load_ms=load_ms,samples_ms=samples,
                    mean_ms=sum(samples)/len(samples),p95_ms=sorted(samples)[18],minimum=lo,maximum=hi,std=std,
                    output_sha256=hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()))
                stage(mode+' close')
            stage(mode+' closed')
        if args.mode:
            report['complete']=True; stage('isolated mode complete'); faulthandler.cancel_dump_traceback_later(); return
        report['ai_vs_bicubic_mae']={m:float((results[m]-results['BICUBIC']).abs().mean()) for m in ('HIGH','ULTRA')}
        if any(v<=1e-6 for v in report['ai_vs_bicubic_mae'].values()):
            raise ValueError('AI output indistinguishable from bicubic; refuse silent fallback')
        report['complete']=True; stage('complete'); faulthandler.cancel_dump_traceback_later()
    except Exception:
        report['error']=traceback.format_exc(); save(); raise


if __name__=='__main__': main()
