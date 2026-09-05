#!/usr/bin/env python3
"""Separate learned global-conditioning context from convolution crop padding."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh output required')
    manifest=json.loads((args.samples/'manifest.json').read_text())
    if manifest['bank_sha256']!='b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0' or manifest['split']!='training-only':
        raise ValueError('fixed training samples required')
    if len(manifest['rows'])!=26:raise ValueError('26 source identities required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,frames=load(args.checkpoint,args.device)
    if frames!=1 or not hasattr(model,'coefficients') or not model.dynamic:
        raise ValueError('trained global controller required')
    model.eval();rows=[]
    with torch.inference_mode():
        for row in manifest['rows']:
            path=args.samples/row['file']
            if digest(path)!=row['sha256']:raise ValueError('sample changed')
            with np.load(path,allow_pickle=False) as sample:
                lr,hr=sample['lr'].copy(),sample['hr'].copy()
            h,w=lr.shape[:2];y,x=((h-96)//4)*2,((w-96)//4)*2
            if min(h,w)<128 or hr.shape[:2]!=(h*2,w*2):raise ValueError('invalid context geometry')
            tensor=torch.from_numpy(lr).permute(2,0,1)[None].to(args.device).float()/255
            cropped_input=tensor[:,:,y:y+96,x:x+96].contiguous()
            full_features=model.anchor.core.conv_1(model.anchor.unshuffle(tensor))
            crop_features=model.anchor.core.conv_1(model.anchor.unshuffle(cropped_input))
            full_coefficients=model.coefficients(full_features)
            crop_coefficients=model.coefficients(crop_features)
            full=model(tensor)[:,:,y*2:(y+96)*2,x*2:(x+96)*2]
            cropped=model(cropped_input)
            original=model.coefficients
            try:
                model.coefficients=lambda _:full_coefficients
                fixed=model(cropped_input)
            finally:model.coefficients=original
            reference=torch.from_numpy(hr[y*2:(y+96)*2,x*2:(x+96)*2].copy()).permute(2,0,1)[None].to(args.device).float()/255
            margins={}
            for margin in (8,32,64,80):
                a,b,c,r=[v[:,:,margin:-margin,margin:-margin] for v in (full,cropped,fixed,reference)]
                margins[str(margin)]={name:dict(mean_rgb_levels=float(delta.abs().mean()*255),max_rgb_levels=float(delta.abs().max()*255)) for name,delta in [('total_context',a-b),('conditioning_only',b-c),('padding_with_fixed_conditioning',a-c)]}
                margins[str(margin)]['full_reference_mae_rgb_levels']=float((a-r).abs().mean()*255)
            rows.append(dict(source_id=row['source_id'],sequence_id=row['sequence_id'],frame=0,crop=[x,y,96,96],
                coefficient_mean_difference=float((full_coefficients-crop_coefficients).abs().mean()),
                coefficient_max_difference=float((full_coefficients-crop_coefficients).abs().max()),margins=margins))
            print(row['source_id'],margins['64'],flush=True)
    report=dict(complete=True,checkpoint_sha256=digest(args.checkpoint),samples_manifest_sha256=digest(args.samples/'manifest.json'),
        code_sha256=digest(__file__),architecture_sha256=digest(Path(__file__).resolve().parents[1]/'architectures/activation_control.py'),
        torch=str(torch.__version__),device=args.device,rows=rows,
        scope='Training-only context mechanism probe; weights unchanged, no new training or evaluation-set selection')
    report['source_balanced']={m:{kind:sum(r['margins'][m][kind]['mean_rgb_levels'] for r in rows)/len(rows)
        for kind in ('total_context','conditioning_only','padding_with_fixed_conditioning')} for m in ('8','32','64','80')}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
