#!/usr/bin/env python3
"""Screen the pinned official FiDeSR teacher on frozen decoded development RGB.

Uses author inference, documented settings and color correction. No training,
Mac speed claim, parameter selection, or substitution for missing weights.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

FRAMES = 'aac663ad088fede87e1ede68cb46501603fc0f0b066b5ae6c07726b6555d923f'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'assets', 'sources', 'frames', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    for name in ('repository', 'assets', 'sources', 'frames', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    if args.out.exists():
        ap.error('fresh output directory required')
    if digest(args.frames/'manifest.json') != FRAMES:
        raise ValueError('frozen full-frame development manifest required')
    sources = json.loads(args.sources.read_text())
    if sources['commit'] != '8042f139fc3f6f91ac450c41f69a14c45f16ac4b':
        raise ValueError('reviewed upstream revision required')
    for name, expected in sources['files'].items():
        if digest(args.repository/name) != expected:
            raise ValueError('upstream source changed: '+name)
    assets = json.loads((args.assets/'downloaded.json').read_text(encoding='utf-8-sig'))
    for row in assets:
        if digest(args.assets/row['destination']) != row['sha256']:
            raise ValueError('downloaded model/config changed')
    # Imports occur only after the pinned local sources/weights are checked.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.chdir(args.repository)
    sys.path.insert(0, str(args.repository))
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.transforms import functional as TF
    from fidesr import FiDeSR_eval
    from src.my_utils.wavelet_color_fix import adain_color_fix
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = dict(pretrained_model_path=str(args.assets/'base'),
        pretrained_path=str(args.assets/'teacher/fidesr.pkl'), mixed_precision='fp16',
        vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
        latent_tiled_size=96, latent_tiled_overlap=32,
        lf_scale=.2, lf_rc=.10, lf_order=2, lf_tau=.8, lf_sharp=10., lf_dmap_gamma=1.2,
        hf_scale=.2, hf_rc=.32, hf_order=2, hf_dmap_gamma=1.2, lf_hf_merge_ratio=.5,
        lrrb_in_ch=8, lrrb_mid_ch=64, lrrb_growth=32, lrrb_res_sc=.2)
    model = FiDeSR_eval(SimpleNamespace(**config))
    model.set_eval()
    manifest = json.loads((args.frames/'manifest.json').read_text())
    inputs = [r for r in manifest['frames'] if r['side'] == 'degraded']
    if len(inputs) != 48 or len({r['file'] for r in inputs}) != 48:
        raise ValueError('48 unique inputs required')
    args.out.mkdir(parents=True)
    receipt = dict(complete=False, input_manifest_sha256=FRAMES, rows=[], configuration=config,
        code_sha256=digest(__file__), sources=sources, assets=assets, smoke=args.smoke,
        torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        runtime_versions={name: importlib.metadata.version(name) for name in
            ('diffusers', 'transformers', 'peft', 'tokenizers', 'huggingface-hub', 'Pillow', 'torchvision')},
        parameters=sum(p.numel() for p in model.parameters()),
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32),
        seed='42 reset before each prediction; author CLI seed argument is otherwise unused',
        presentation='author RGB8 + default AdaIN, then PIL bicubic 4x to 2x',
        limitations=['external teacher screen, not native inference',
                     'development sources reused; upstream source overlap unknown'])

    def predict(source):
        # Frozen inputs already satisfy author's minimum size/multiple-of-eight rules.
        image = source.resize((2560, 1440)).resize((2560, 1440), Image.Resampling.LANCZOS)
        x = TF.to_tensor(image).unsqueeze(0).cuda()*2-1
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.inference_mode():
            seconds, output = model(x, prompt='')
        if output.shape != (1, 3, 1440, 2560) or not torch.isfinite(output).all():
            raise ValueError('nonfinite or incorrectly sized teacher output')
        result = TF.to_pil_image((output[0].cpu()*.5+.5).clamp(0, 1))
        result = adain_color_fix(target=result, source=image)
        return result.resize((1280, 720), Image.Resampling.BICUBIC), seconds

    for row in inputs[:1] if args.smoke else inputs:
        if digest(args.frames/row['file']) != row['sha256']:
            raise ValueError('frozen LR changed')
        with Image.open(args.frames/row['file']) as image:
            source = image.convert('RGB')
        if source.size != (640, 360):
            raise ValueError('fixed decoded geometry required')
        output, seconds = predict(source)
        if args.smoke:
            repeat, _ = predict(source)
            maximum = int(np.abs(np.array(output).astype(np.int16)-np.array(repeat)).max())
            receipt['repeat_max_rgb8_delta'] = maximum
            if maximum > 1:
                raise ValueError('seeded repeat changes more than one RGB8 level')
        path = args.out/Path(row['file']).name
        output.save(path)
        receipt['rows'].append(dict(row, output_file=path.name, output_sha256=digest(path),
                                    model_seconds=seconds))
        print(path.name, seconds, flush=True)
        (args.out/'outputs.json').write_text(json.dumps(receipt, indent=2)+'\n')
    receipt['complete'] = True
    (args.out/'outputs.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
