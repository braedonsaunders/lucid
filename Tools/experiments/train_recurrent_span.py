#!/usr/bin/env python3
"""Train the previous-output input convolution while preserving the SR backbone."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from train_causal_detail import batch, digest, load_bank
from train_presented_detail import reconstruction_objective, state_digest
from native_stages import preprocess_sequence
from recurrent_frame_feeding import recurrent_step
from architectures.recurrent_span import RecurrentSPAN
from eval_checkpoint import load
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def run_sequence(model, decoded, native_inputs=False):
    if native_inputs:
        with torch.no_grad():
            inputs = preprocess_sequence(decoded)
    else:
        inputs = decoded
    state, outputs = None, []
    for t in range(decoded.shape[1]):
        output, state, _ = recurrent_step(model, inputs[:, t], decoded[:, t], state,
                                          stream='sequence', index=t)
        outputs.append(output)
    return torch.stack(outputs, 1), inputs


@torch.inference_mode()
def validate(model, data, source_ids, native_inputs):
    model.eval()
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    for lr, hr, identity in data:
        source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
        output, inputs = run_sequence(model, source, native_inputs)
        base = torch.stack([model.sr(inputs[:, t]).clamp(0, 1) for t in range(inputs.shape[1])], 1)
        for variant, values in [('base', base), ('recurrent', output)]:
            # Longer validation history than the training unroll exposes drift.
            for t in sorted({0, inputs.shape[1] // 2, inputs.shape[1] - 1}):
                rows.append({'source_id': source_ids[identity], 'sequence_id': identity,
                    'variant': variant, 'frame': t,
                    'metrics': scorer.spatial(image(values[0, t, :, 8:-8, 8:-8]),
                                              image(reference[0, t, :, 8:-8, 8:-8]))})
            error = values[..., 8:-8, 8:-8] - reference[..., 8:-8, 8:-8]
            temporal.append({'id': identity, 'variant': variant,
                'residual_change_l1': float(torch.diff(error, dim=1).abs().mean()),
                'mean_rgb_mse': float(error.square().mean()),
                'last_rgb_mse': float(error[:, -1].square().mean())})
    return {'complete': True, 'scope': 'source-disjoint full-bank-sequence patch diagnostic; not native playback',
            'rows': rows, 'summary': summarize(rows), 'temporal': temporal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'init', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260914)
    ap.add_argument('--native-input-stages', action='store_true')
    args = ap.parse_args()
    if args.out.exists() or min(args.steps, args.batch) < 1 or args.frames < 3 or args.crop < 32 or args.crop % 2:
        ap.error('fresh output, positive steps/batch, frames >=3 and even crop >=32 required')
    if not torch.cuda.is_available():
        raise ValueError('authorized CUDA worker required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest, data = load_bank(args.bank)
    if manifest['frames'] < args.frames or any(min(lr.shape[1:3]) < args.crop for rows in data.values() for lr, _, _ in rows):
        ap.error('adequate bank sequence/crop geometry required')
    base, _, frames = load(args.init, 'cuda')
    if type(base) is not Unshuffled or frames != 1:
        ap.error('ordinary single-frame 2x SPAN initialization required')
    model = RecurrentSPAN(base).cuda().train()
    frozen_hash = state_digest(model.sr.state_dict())
    optimizer = torch.optim.AdamW(model.history.parameters(), lr=.0002, weight_decay=0)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    files = ['architectures/recurrent_span.py', 'architectures/subspace_adapter.py',
             'experiments/recurrent_frame_feeding.py', 'experiments/native_stages.py',
             'experiments/train_presented_detail.py', 'experiments/train_recurrent_span.py']
    experiment = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'bank_sha256': digest(args.bank / 'manifest.json'), 'checkpoint_sha256': digest(args.init),
        'frozen_backbone_sha256': frozen_hash, 'source_hashes': {p: digest(source_root / p) for p in files},
        'history_parameters': sum(p.numel() for p in model.history.parameters()),
        'recipe': 'final-frame HR reconstruction; backpropagation through quantized recurrent outputs; frozen current-frame graph; no GAN',
        'motion': 'decoded-LR native integer search plus half-pixel refinement around integer and zero seeds; detached correspondence',
        'history_storage': 'RGB8 straight-through quantization before output detail stages',
        'limitations': 'crop-local history; no native SR warp/cost parity; no browser timing or release admission',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        source, reference = batch(data['train'], rng, args.batch, args.frames, args.crop)
        if step == 1:
            experiment['first_decoded_sequence_sha256'] = state_digest({'source': source, 'reference': reference})
            (args.out / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
        source, reference = source.cuda(), reference[:, -1].cuda()
        optimizer.zero_grad(set_to_none=True)
        # Keep warping/quantization FP32. Only the neural graph needs autocast;
        # this initial causal probe deliberately uses FP32 throughout.
        output, _ = run_sequence(model, source, args.native_input_stages)
        a, b = output[:, -1, :, 8:-8, 8:-8], reference[..., 8:-8, 8:-8]
        loss = reconstruction_objective(a, b, b, 'reference')
        if not torch.isfinite(loss):
            raise ValueError('nonfinite recurrent objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.history.parameters(), 1)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr'] = .0002 * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'step': step, 'loss': float(loss.detach()),
                              'minutes': (time.monotonic() - started) / 60}), flush=True)
    if state_digest(model.sr.state_dict()) != frozen_hash:
        raise ValueError('frozen backbone changed')
    torch.save({'model': model.state_dict(), 'architecture': 'recurrent_span2x', 'scale': 2,
                'channels': model.sr.core.conv_1.out_channels, 'version': model.sr.version,
                'step': args.steps, 'experiment': experiment}, args.out / f'step{args.steps:06d}.pth')
    result = validate(model, data['validation'], {r['id']: r['source_id'] for r in manifest['sequences']}, args.native_input_stages)
    (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'complete.json').write_text(json.dumps({'complete': True, 'backbone_unchanged': True,
        'steps': args.steps, 'minutes': (time.monotonic() - started) / 60}) + '\n')


if __name__ == '__main__':
    main()
