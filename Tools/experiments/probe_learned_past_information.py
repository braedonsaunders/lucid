#!/usr/bin/env python3
"""Training-bank-only diagnostic of learned subpixel information in past frames.

This fits small diagnostic linear predictors, not a shipping VSR network. DIS
is an alignment probe, not a new architecture or a native runtime claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import bands, correlation, load

BANK = 'b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0'
SHIPPING_CACHE = '3909586f4058c096823074c61f6e4f1ead741925f2beaf70670535908346ce41'
ARMS = ('spatial', 'unaligned', 'aligned')
TIMES = (4, 8, 12)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap(image, flow):
    h, w = flow.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + flow[..., 0], y + flow[..., 1]
    inside = (mx >= 0) & (my >= 0) & (mx < w-1) & (my < h-1)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), inside


def align_prediction(current, past, prediction):
    """Align twelve learned prediction channels using only decoded LR flow."""
    a, b = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (current, past)]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    backward = dis.calc(a, b, None)
    forward = dis.calc(b, a, None)
    warped_forward, inside = remap(forward, backward)
    warped_gray, _ = remap(b.astype(np.float32), backward)
    disagreement = np.square(backward + warped_forward).sum(axis=2)
    energy = np.square(backward).sum(axis=2) + np.square(warped_forward).sum(axis=2)
    visible = inside & (disagreement <= .01*energy+.5) & (np.abs(a.astype(np.float32)-warped_gray) <= 32)
    h, w = a.shape
    if prediction.shape != (h*2,w*2,12): raise ValueError('twelve 2x prediction channels required')
    flow2 = cv2.resize(backward, (w*2,h*2), interpolation=cv2.INTER_LINEAR)*2
    warped, inside2 = remap(prediction, flow2)
    visible2 = cv2.resize(visible.astype(np.uint8), (w*2,h*2), interpolation=cv2.INTER_NEAREST).astype(bool) & inside2
    return warped, visible2


@torch.inference_mode()
def prediction_channels(model, image, device):
    output = model(tensor(image.astype(np.float32)/255, device))
    if output.shape != (1,3,image.shape[0]*4,image.shape[1]*4): raise ValueError('full 4x model required')
    result = F.pixel_unshuffle(output,2)
    if not torch.isfinite(result).all(): raise ValueError('nonfinite prediction')
    return result[0].permute(1,2,0).cpu().numpy()


def tensor(image, device):
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float().to(device)


def features(current, base, prediction, history, masks, arm):
    high = base-F.avg_pool2d(F.pad(base,(1,1,1,1),mode='replicate'),3,1)
    patches = F.unfold(F.pad(torch.cat([current-base, high],dim=1),(1,1,1,1),mode='replicate'),3)[0].T
    learned = [prediction-base.repeat_interleave(4,dim=1)]
    if arm != 'spatial': learned.extend(previous-prediction for previous in history)
    learned = torch.cat(learned,dim=1)[0].permute(1,2,0).reshape(patches.shape[0],-1)
    patches = torch.cat([patches,learned],dim=1)
    if arm != 'spatial': patches = torch.cat([patches, *[m.reshape(-1,1) for m in masks]],dim=1)
    return torch.cat([patches,torch.ones_like(patches[:,:1])],dim=1)


def solve(gram, cross):
    """One fixed trace-scaled ridge value; solve in CPU float64."""
    a, b = gram.double().cpu(), cross.double().cpu()
    regularization = .001 * torch.trace(a) / a.shape[0]
    if not torch.isfinite(a).all() or regularization <= 0: raise ValueError('invalid fit statistics')
    return torch.linalg.solve(a+regularization*torch.eye(a.shape[0],dtype=a.dtype),b).float()


def metrics(output, reference):
    rgb = np.clip(np.rint(output*255),0,255).astype(np.uint8)
    y, r = [np.asarray(Image.fromarray(v).convert('L'),dtype=np.float64) for v in (rgb,reference)]
    # Same fixed interior for all arms; no flow-dependent metric masking.
    return {'mse_rgb':float(np.square(rgb[16:-16,16:-16].astype(np.float64)-reference[16:-16,16:-16]).mean()),
            'fine_correlation':correlation(bands(y)[0][16:-16,16:-16],bands(r)[0][16:-16,16:-16])}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','shipping_cache','out'):ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh output required')
    if digest(args.bank/'manifest.json')!=BANK:raise ValueError('fixed training bank required')
    if digest(args.shipping_cache/'manifest.json')!=SHIPPING_CACHE:raise ValueError('fixed shipping cache required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    if digest(args.checkpoint) != 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65': raise ValueError('frozen shipping checkpoint required')
    model,_,frames=load(args.checkpoint,args.device)
    if frames != 1: raise ValueError('single-frame model required')
    bank=json.loads((args.bank/'manifest.json').read_text());cache=json.loads((args.shipping_cache/'manifest.json').read_text())
    if cache['bank_sha256']!=BANK:raise ValueError('wrong shipping cache')
    sources={s['id']:s for s in bank['sources'] if s['split']=='train'}
    if len(sources)!=26:raise ValueError('fixed 26 training identities required')
    ordered=sorted(sources,key=lambda s:hashlib.sha256(('20260905-past-information:'+s).encode()).hexdigest())
    fit=set(ordered[::2]);check=set(ordered[1::2])
    selected={s:min((q for q in bank['sequences'] if q['source_id']==s and q['split']=='train'),key=lambda q:q['id']) for s in ordered}
    cached={q['id']:q for q in cache['sequences']}
    args.out.mkdir();started=time.monotonic()
    report={'complete':False,'purpose':__doc__,'bank_sha256':BANK,'shipping_cache_sha256':digest(args.shipping_cache/'manifest.json'),
        'source_sha256':digest(__file__),'metric_source_sha256':digest(Path(__file__).resolve().parents[1]/'eval_checkpoint.py'),
        'checkpoint_sha256':digest(args.checkpoint),'representation':'12 unshuffled full-4x prediction channels at 2x LR; current plus two past residuals',
        'architecture_sha256':digest(Path(__file__).resolve().parents[1]/'architectures/span_arch.py'),
        'loader_source_sha256':digest(Path(__file__).resolve().parents[1]/'train_span.py'),
        'fit_sources':sorted(fit),'check_sources':sorted(check),'source_families':{s:sources[s]['family'] for s in ordered},
        'selection':selected,'frames':TIMES,'fit_stride':8,'metric_margin':16,'ridge_trace_fraction':.001,
        'mechanism_limits':{'mse_improvement_min':.01,'fine_correlation_change_min':.002,'sources_with_both_improved_min':7},
        'torch':str(torch.__version__),'opencv':cv2.__version__,'device':args.device,'rows':[],
        'limitations':['All identities were previously used for neural training; this is a mechanism check, not release validation.',
            'Netflix collection families overlap across fit/check; source IDs do not.',
            'No future frames, HR-driven flow, reference-dependent visibility or quality-based hyperparameter selection.',
            'Linear residual fitting and CPU DIS are diagnostics, not a new architecture or measured Mac implementation.']}
    (args.out/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    stats={};weights={}
    for phase,identities in [('fit',fit),('check',check)]:
        for source in [s for s in ordered if s in identities]:
            seq=selected[source];row=cached[seq['id']];path=args.bank/seq['file'];cp=args.shipping_cache/row['file']
            if digest(path)!=seq['sha256'] or digest(cp)!=row['sha256']:raise ValueError('changed input sequence/cache')
            with np.load(path,allow_pickle=False) as pair:lr,hr=pair['lr'].copy(),pair['hr'].copy()
            with np.load(cp,allow_pickle=False) as pair:shipping=pair['teacher'].copy()
            if lr.dtype!=np.uint8 or hr.dtype!=np.uint8 or shipping.dtype!=np.uint8 or shipping.shape!=hr.shape or hr.shape[1:3]!=(lr.shape[1]*2,lr.shape[2]*2):raise ValueError('invalid pair')
            predictions={i:prediction_channels(model,lr[i],args.device) for i in sorted({j for t in TIMES for j in (t-2,t-1,t)})}
            for t in TIMES:
                h,w=hr.shape[1:3];current=np.asarray(Image.fromarray(lr[t]).resize((w,h),Image.Resampling.BICUBIC),dtype=np.float32)/255
                base=tensor(shipping[t].astype(np.float32)/255,args.device);cur=tensor(current,args.device)
                pred=tensor(predictions[t],args.device)
                aligned=[];unaligned=[];masks=[]
                for prev in (t-2,t-1):
                    warped,mask=align_prediction(lr[t],lr[prev],predictions[prev]);warped[~mask]=predictions[t][~mask]
                    aligned.append(tensor(warped,args.device));masks.append(torch.from_numpy(mask.astype(np.float32)).to(args.device))
                    unaligned.append(tensor(predictions[prev],args.device))
                target=tensor(hr[t].astype(np.float32)/255,args.device)
                if phase=='check':report['rows'].append({'source_id':source,'frame':t,'variant':'shipping','metrics':metrics(shipping[t].astype(np.float32)/255,hr[t])})
                for arm in ARMS:
                    history=aligned if arm=='aligned' else unaligned
                    visibility=masks if arm=='aligned' else [torch.ones_like(m) for m in masks]
                    x=features(cur,base,pred,history,visibility,arm)
                    if phase=='fit':
                        x=x.reshape(h,w,-1)[16:-16:8,16:-16:8].reshape(-1,x.shape[1])
                        y=(target-base)[0].permute(1,2,0)[16:-16:8,16:-16:8].reshape(-1,3)
                        a,b=x.T@x,x.T@y
                        if arm not in stats:stats[arm]=[a,b]
                        else:stats[arm][0]+=a;stats[arm][1]+=b
                    else:
                        correction=(x@weights[arm]).reshape(h,w,3).cpu().numpy()
                        predicted=np.clip(shipping[t].astype(np.float32)/255+correction,0,1)
                        report['rows'].append({'source_id':source,'frame':t,'variant':arm,'metrics':metrics(predicted,hr[t]),
                            'visible_fraction':float(torch.stack(masks).mean())})
                print(phase,source,t,flush=True)
            if phase=='check':(args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if phase=='fit':
            weights={arm:solve(*stats[arm]).to(args.device) for arm in ARMS}
            torch.save({k:v.cpu() for k,v in weights.items()},args.out/'diagnostic-linear-weights.pth')
    means={arm:{s:{m:float(np.mean([r['metrics'][m] for r in report['rows'] if r['variant']==arm and r['source_id']==s]))
        for m in ('mse_rgb','fine_correlation')} for s in check} for arm in ('shipping',*ARMS)}
    balanced={arm:{m:float(np.mean([v[m] for v in by_source.values()])) for m in ('mse_rgb','fine_correlation')} for arm,by_source in means.items()}
    comparisons={arm:{'mse_improvement':1-balanced['aligned']['mse_rgb']/balanced[arm]['mse_rgb'],
        'fine_correlation_change':balanced['aligned']['fine_correlation']-balanced[arm]['fine_correlation'],
        'sources_with_both_improved':sum(means['aligned'][s]['mse_rgb']<means[arm][s]['mse_rgb'] and means['aligned'][s]['fine_correlation']>means[arm][s]['fine_correlation'] for s in check)} for arm in ('shipping','spatial','unaligned')}
    report.update(complete=True,minutes=(time.monotonic()-started)/60,source_balanced=balanced,comparisons=comparisons,
        mechanism_pass=all(v['mse_improvement']>=.01 and v['fine_correlation_change']>=.002 and v['sources_with_both_improved']>=7 for v in comparisons.values()),
        promotion_authorized=False,weights_sha256=digest(args.out/'diagnostic-linear-weights.pth'))
    (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__=='__main__':main()
