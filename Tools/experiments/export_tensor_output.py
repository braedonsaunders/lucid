#!/usr/bin/env python3
"""Export identical 4x reconstruction with image versus tensor output boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import coremltools as ct
import numpy as np
import torch

from profile_folded_shipping import freeze_convolutions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from convert_span import ImageRange, selective_fp16


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or min(args.width, args.height) <= 0 or args.width % 16 or args.height % 2:
        ap.error('fresh output and aligned positive geometry required')
    if sha(args.checkpoint) != 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65':
        raise ValueError('unchanged shipping weights required')
    torch.set_num_threads(4)
    model, _, frames = load(args.checkpoint, 'cpu')
    if frames != 1: raise ValueError('single-frame shipping model required')
    freeze_convolutions(model)
    wrapper = ImageRange(model).eval()
    example = torch.zeros(1, 3, args.height, args.width)
    with torch.inference_mode(): traced = torch.jit.trace(wrapper, example)
    args.out.mkdir(parents=True)
    report = dict(complete=False, purpose=__doc__, checkpoint_sha256=sha(args.checkpoint),
                  exporter_sha256=sha(__file__), torch=str(torch.__version__), coremltools=ct.__version__,
                  precision='identical shipping mixed precision; output storage type is the sole intervention',
                  input=[args.width,args.height], output=[args.width*4,args.height*4], models={})
    for label, dtype in [('image4x', None), ('tensor4x_fp32', np.float32), ('tensor4x_fp16', np.float16)]:
        output = ct.ImageType(name='output', color_layout=ct.colorlayout.RGB) if dtype is None else ct.TensorType(name='output', dtype=dtype)
        converted = ct.convert(traced, inputs=[ct.ImageType(name='input', shape=example.shape,
            color_layout=ct.colorlayout.RGB, scale=1/255)], outputs=[output],
            convert_to='mlprogram', compute_precision=selective_fp16(), minimum_deployment_target=ct.target.macOS15)
        converted.user_defined_metadata['lucid.checkpoint_sha256'] = sha(args.checkpoint)
        converted.user_defined_metadata['lucid.output_scale'] = '4'
        converted.user_defined_metadata['lucid.output_range'] = '0..255'
        path = args.out/(label+'.mlpackage'); converted.save(str(path))
        report['models'][label] = {str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    report['complete'] = True
    (args.out/'export.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
