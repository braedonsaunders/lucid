"""Fixed-weight source-TAA replay with or without stationary motion overrides."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.subspace_adapter import fuse_convolutions
from eval_checkpoint import load
import native_stages as stages
from replay_recurrent_motion import validation_pairs
from train_causal_detail import digest
from train_span import Unshuffled
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
from evaluate_sequences import Scorer, summarize


def policy_motion(current, previous, policy, search=stages.motion_blocks):
    if policy not in ('taa', 'search', 'gain_only'):
        raise ValueError('unknown motion policy')
    if policy == 'taa':
        return search(current, previous)
    field = search(current, previous, stationary_override=False)
    if policy == 'search':
        return field
    legacy = search(current, previous)
    # Where legacy kept motion, the match already passed its gain threshold.
    # Where it returned zero, legacy.error is the stationary photometric error.
    # Keep weak-match rejection but remove the independent low-error veto.
    stationary = (legacy[:, :2] == 0).all(1, keepdim=True)
    weak = legacy[:, 3:4] - field[:, 3:4] < 2 / 255
    return torch.where(stationary & weak, legacy, field)


@torch.inference_mode()
def evaluate(model, pairs, sources):
    scorer = Scorer(torch.device('cuda'))
    rows, temporal = [], []
    original = stages.motion_blocks
    image = lambda x: Image.fromarray(x.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    try:
        for policy in ('taa', 'gain_only', 'search'):
            # Local diagnostic override only. Production defaults and model bytes
            # stay unchanged; both policies retain the same TAA/deband pipeline.
            stages.motion_blocks = lambda a, b: policy_motion(a, b, policy, original)
            print('evaluating', policy, flush=True)
            for lr, hr, identity in pairs:
                source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
                reference = torch.from_numpy(hr.copy()).permute(0, 3, 1, 2)[None].float().cuda() / 255
                inputs = stages.preprocess_sequence(source)
                output = torch.stack([model(inputs[:, t]).clamp(0, 1) for t in range(16)], 1)
                for t in (0, 8, 15):
                    rows.append({'source_id': sources[identity], 'sequence_id': identity,
                                 'variant': policy, 'frame': t,
                                 'metrics': scorer.spatial(image(output[0, t, :, 8:-8, 8:-8]),
                                                           image(reference[0, t, :, 8:-8, 8:-8]))})
                error = output[..., 8:-8, 8:-8] - reference[..., 8:-8, 8:-8]
                temporal.append({'id': identity, 'source_id': sources[identity], 'variant': policy,
                                 'residual_change_l1': float(torch.diff(error, dim=1).abs().mean()),
                                 'mean_rgb_mse': float(error.square().mean()),
                                 'last_rgb_mse': float(error[:, -1].square().mean())})
            print('finished', policy, flush=True)
    finally:
        stages.motion_blocks = original
    return {'rows': rows, 'summary': summarize(rows), 'temporal': temporal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('bank', 'reds-bank', 'checkpoint', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('fresh output required')
    torch.set_num_threads(4)
    training = json.loads((args.bank / 'manifest.json').read_text())
    pairs, sources = validation_pairs(args.bank, training)
    reds, reds_sources = validation_pairs(args.reds_bank, training, reds=True)
    if sources.keys() & reds_sources.keys():
        raise ValueError('duplicate validation identities')
    pairs.extend(reds)
    sources.update(reds_sources)
    model, _, frames = load(args.checkpoint, 'cuda')
    if type(model) is not Unshuffled or frames != 1 or model.core.upsampler[1].upscale_factor != 4:
        raise ValueError('single-frame 2x checkpoint required')
    fuse_convolutions(model)
    model.eval()
    result = evaluate(model, pairs, sources)
    root = Path(__file__).resolve().parents[1]
    result.update(complete=True,
                  scope='Same weights,72 source-disjoint full16-frame validation patches,216 scored frames per policy; source-stage proxy,not native playback',
                  motion_policy='taa reproduces stationary overrides; gain_only removes the low-error veto while retaining weak-match rejection; search removes both; no subpixel refinement. gain_only uses two searches only as a diagnostic; this is not a native cost implementation.',
                  settings=asdict(stages.StageSettings()), selection=sources,
                  checkpoint_sha256=digest(args.checkpoint),
                  bank_sha256=digest(args.bank / 'manifest.json'),
                  reds_bank_sha256=digest(args.reds_bank / 'manifest.json'),
                  source_sha256={str(p.relative_to(root)): digest(p) for p in [
                      Path(__file__).resolve(), root / 'experiments/native_stages.py',
                      root / 'experiments/replay_recurrent_motion.py', root / 'eval_checkpoint.py',
                      root / 'architectures/subspace_adapter.py', root / 'frontier_eval/evaluate_sequences.py']},
                  torch=str(torch.__version__), device=torch.cuda.get_device_name())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('complete TAA motion replay', flush=True)


if __name__ == '__main__':
    main()
