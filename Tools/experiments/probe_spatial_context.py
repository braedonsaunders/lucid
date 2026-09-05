#!/usr/bin/env python3
"""Training-only comparison of global versus fixed local crop conditioning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from architectures.spatial_activation_control import SpatialActivationControl, CoarseSpatialActivationControl


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('samples','checkpoint','out'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--device', choices=['cpu','cuda','mps'], default='cuda')
    ap.add_argument('--coarse', action='store_true', help='Fixed stride-two grid with bilinear controls; distinct architecture')
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    manifest = json.loads((args.samples/'manifest.json').read_text())
    if manifest['bank_sha256'] != 'b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0' or manifest['split'] != 'training-only' or len(manifest['rows']) != 26:
        raise ValueError('fixed training-only samples required')
    if digest(args.checkpoint) != '24cc7552506808596ac14287d0a4ef652844cb9590658cbf336283aefa7884dd':
        raise ValueError('fixed constrained R4 mechanism checkpoint required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    global_model,_,_ = load(args.checkpoint, args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(state['controller_config'], window=9)
    local_model = (CoarseSpatialActivationControl if args.coarse else SpatialActivationControl)(global_model.anchor, **config).to(args.device).eval()
    local_model.load_state_dict(global_model.state_dict(), strict=True)
    rows = []
    with torch.inference_mode():
        for row in manifest['rows']:
            path = args.samples/row['file']
            if digest(path) != row['sha256']:
                raise ValueError('training sample changed')
            with np.load(path, allow_pickle=False) as sample:
                lr = sample['lr'].copy()
            h,w = lr.shape[:2]
            if min(h,w) < 224:
                raise ValueError('insufficient context')
            y,x = ((h-160)//4)*2,((w-160)//4)*2
            tensor = torch.from_numpy(lr).permute(2,0,1)[None].to(args.device).float()/255
            cropped = tensor[:,:,y:y+160,x:x+160].contiguous()
            result = dict(source_id=row['source_id'],sequence_id=row['sequence_id'],crop=[x,y,160,160])
            for label,model in [('global',global_model),('local9',local_model)]:
                full = model(tensor)[:,:,y*2:(y+160)*2,x*2:(x+160)*2]
                crop_output = model(cropped)
                margin = 96 if args.coarse else 88
                delta = (full[:,:,margin:-margin,margin:-margin]-crop_output[:,:,margin:-margin,margin:-margin]).abs()*255
                if not torch.isfinite(delta).all():
                    raise ValueError('nonfinite output')
                result[label] = dict(mean_rgb_levels=float(delta.mean()),max_rgb_levels=float(delta.max()))
            rows.append(result)
            print(row['source_id'], result['local9'], flush=True)
    args.out.mkdir()
    architecture = 'coarse_spatial_activation_control2x' if args.coarse else 'spatial_activation_control2x'
    torch.save(dict(state,model=local_model.cpu().state_dict(),architecture=architecture,
                    controller_config=config,spatial_probe_only=True),args.out/'nonzero-local-probe.pth')
    report = dict(complete=True,rows=rows,device=args.device,torch=str(torch.__version__),architecture=architecture,margin=margin,
                  checkpoint_sha256=digest(args.checkpoint),samples_sha256=digest(args.samples/'manifest.json'),
                  code_sha256=digest(__file__),architecture_sha256=digest(Path(__file__).resolve().parents[1]/'architectures/spatial_activation_control.py'),
                  probe_checkpoint_sha256=digest(args.out/'nonzero-local-probe.pth'),
                  passes=all(r['local9']['mean_rgb_levels']<=.001 and r['local9']['max_rgb_levels']<=.05 for r in rows),
                  scope='Untrained local reinterpretation of fixed global weights; mechanism and later native-cost probe only, no quality selection')
    report['source_balanced_mean_rgb_levels'] = {k:sum(r[k]['mean_rgb_levels'] for r in rows)/len(rows) for k in ('global','local9')}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
