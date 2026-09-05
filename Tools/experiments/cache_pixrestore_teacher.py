#!/usr/bin/env python3
"""Cache pinned PixRestore targets for training-only 256px LR sequences."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from train_causal_detail import digest, load_bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repository', type=Path, required=True)
    ap.add_argument('--provenance', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--dino-repository', type=Path, required=True)
    ap.add_argument('--dino-checkpoint', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260913)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve existing teacher output; use a fresh directory')
    provenance = json.loads(args.provenance.read_text())
    for name, expected in provenance['code_sha256'].items():
        if digest(args.repository / name) != expected:
            raise ValueError(f'upstream code changed: {name}')
    if digest(args.checkpoint / 'ema_model.safetensors') != provenance['files'][0]['lfs']['oid']:
        raise ValueError('teacher weights differ from measured baseline')
    if digest(args.dino_checkpoint) != provenance['teacher_sha256']:
        raise ValueError('DINO weights changed')
    manifest, data = load_bank(args.bank)
    if manifest['lr_patch'] != 256:
        raise ValueError('teacher requires 512px output context')
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    sys.path.insert(0, str(args.repository.resolve()))
    import inference as upstream
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image
    config_data = upstream.read_config(args.checkpoint / 'config.json')
    if config_data != provenance['config']:
        # Baseline provenance records the unmodified downloaded configuration.
        raise ValueError('teacher configuration differs from measured baseline')
    config_data.update(cfg_scale=1, dinov2_repository=str(args.dino_repository),
                       dinov2_checkpoint=str(args.dino_checkpoint))
    state = upstream.read_state_dict(args.checkpoint / 'ema_model.safetensors')
    config = SimpleNamespace(**upstream.apply_checkpoint_architecture_flags(config_data, state))
    accelerator = upstream.Accelerator(mixed_precision='bf16')
    if accelerator.device.type != 'cuda' or accelerator.num_processes != 1:
        raise ValueError('single-owner CUDA run required')
    model = upstream.build_model(config)
    upstream.load_weights(model, args.checkpoint / 'ema_model.safetensors', state)
    del state
    model.to(accelerator.device).eval().requires_grad_(False)
    flow = upstream.build_flow(config, accelerator)
    encoder = upstream.load_dinov2(config.encoder_type, accelerator.device,
        repository=config.dinov2_repository, checkpoint=config.dinov2_checkpoint)
    args.out.mkdir(parents=True)
    rows, frame_index = [], 0
    for lr, _, identity in data['train']:
        outputs = []
        for pixels in lr:
            torch.manual_seed(args.seed + frame_index)
            torch.cuda.manual_seed_all(args.seed + frame_index)
            image = Image.fromarray(pixels).resize((512, 512), Image.Resampling.BICUBIC)
            lq = pil_to_tensor(image)[None].to(accelerator.device).float().div_(127.5).sub_(1)
            with torch.no_grad(), accelerator.autocast():
                features = upstream.extract_layers(encoder, lq, config.encoder_layers, config.encoder_input_size)
                prediction = flow.sample_multistep_fm(model, lq, venc_fea=features, n_steps=1, schedule='linear')
            image = prediction[0].float().cpu().add(1).mul(.5).clamp(0, 1)
            outputs.append(np.asarray(to_pil_image(image)).copy())
            frame_index += 1
        target = args.out / (identity + '.npz')
        np.savez_compressed(target, teacher=np.stack(outputs))
        rows.append({'id': identity, 'file': target.name, 'sha256': digest(target)})
        print(f'{len(rows)}/{len(data["train"])} {identity} frames={frame_index}', flush=True)
    receipt = {'bank_sha256': digest(args.bank / 'manifest.json'),
        'baseline_provenance_sha256': digest(args.provenance), 'code_sha256': digest(__file__),
        'teacher_sha256': provenance['files'][0]['lfs']['oid'], 'seed': args.seed,
        'inference': '1 step, CFG 1, BF16, PIL bicubic 2x, 512 square, no compilation',
        'split': 'train only', 'sequences': rows, 'frames': frame_index,
        'reference_used_for_prediction': False, 'torch': str(torch.__version__)}
    (args.out / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
