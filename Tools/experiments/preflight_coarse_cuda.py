#!/usr/bin/env python3
"""Exercise coarse-control CUDA backward twice; no dataset training or promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from architectures.spatial_activation_control import CoarseSpatialActivationControl


def state_digest(state):
    return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()
                                  for k,v in sorted(state.items()))).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh report required')
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash!='24cc7552506808596ac14287d0a4ef652844cb9590658cbf336283aefa7884dd':
        raise ValueError('fixed R4 anchor required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    reference,_,_=load(args.checkpoint,'cuda')
    raw=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    rows=[]
    for repeat in range(2):
        torch.manual_seed(20260914);torch.cuda.manual_seed_all(20260914)
        model=CoarseSpatialActivationControl(reference.anchor,**dict(raw['controller_config'],window=9)).cuda().train()
        frozen=state_digest(model.anchor.state_dict())
        optimizer=torch.optim.AdamW(model.controller.parameters(),lr=.001)
        x=torch.rand(4,3,96,96,device='cuda')
        first=None
        for step in range(3):
            optimizer.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(x)
                if step==0:
                    if not torch.equal(output,model.anchor(x)):
                        raise ValueError('zero controller changes BF16 output')
                    first=state_digest({'output':output.float()})
            loss=(output.float()-.5).square().mean()
            loss.backward();optimizer.step()
        if frozen!=state_digest(model.anchor.state_dict()):raise ValueError('anchor changed')
        if model.controller[1].weight.grad.abs().sum()<=0:raise ValueError('missing control gradient')
        rows.append(dict(repeat=repeat,first_output=first,controller=state_digest(model.controller.state_dict()),anchor=frozen))
    if any(rows[0][k]!=rows[1][k] for k in ('first_output','controller','anchor')):
        raise ValueError('CUDA repeats differ')
    args.out.write_text(json.dumps(dict(complete=True,rows=rows,torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),checkpoint_sha256=checkpoint_hash,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Six synthetic optimizer steps test deterministic BF16 backward; not a dataset training endpoint'),indent=2)+'\n')


if __name__=='__main__':main()
