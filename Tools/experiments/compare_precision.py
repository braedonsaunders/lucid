#!/usr/bin/env python3
"""Interleave existing Core ML precision variants after adversarial RGB checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_span import ImageRange
from eval_checkpoint import load
from profile_folded_shipping import freeze_convolutions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(width, height, video_frame):
    rng = np.random.default_rng(20260905)
    ramp = np.broadcast_to(np.linspace(0, 255, width, dtype=np.uint8)[None, :, None], (height, width, 3)).copy()
    checker = np.repeat((((np.indices((height, width)).sum(0) % 2) * 255).astype(np.uint8))[..., None], 3, axis=2)
    color = np.zeros((height, width, 3), np.uint8)
    for i in range(3):
        color[:, i * width // 3:(i + 1) * width // 3, i] = 255
    return {
        'black': Image.fromarray(np.zeros((height, width, 3), np.uint8)),
        'near_black': Image.fromarray(rng.integers(0, 5, (height, width, 3), dtype=np.uint8)),
        'white': Image.fromarray(np.full((height, width, 3), 255, np.uint8)),
        'ramp': Image.fromarray(ramp), 'checker': Image.fromarray(checker),
        'saturated': Image.fromarray(color),
        'noise': Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)),
        'video': Image.open(video_frame).convert('RGB').resize((width, height), Image.Resampling.BICUBIC),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shipping', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--video-frame', type=Path, required=True)
    parser.add_argument('--packages', type=Path, default=Path('.build'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=60)
    args = parser.parse_args()
    if args.out.exists() or args.samples < 40:
        parser.error('fresh output and at least 40 samples required')
    policies = ['mul-fp32', 'shuffle-fp32', 'fp16']
    checkpoints = {'shipping4x': args.shipping, 'direct2x_trained': args.candidate}
    profiles = {p: json.loads((args.packages/f'precision-{p}'/'profile.json').read_text()) for p in policies}
    for p, profile in profiles.items():
        if (profile.get('complete') is not True or profile['precision_policy'] != p or
                profile['checkpoint_sha256'] != sha(args.shipping) or
                profile['direct_checkpoint_sha256'] != sha(args.candidate)):
            raise ValueError('package provenance mismatch')
    models = {}
    for label, checkpoint in checkpoints.items():
        model, _, _ = load(checkpoint, 'cpu')
        freeze_convolutions(model)
        models[label] = ImageRange(model).eval()
    report = {'purpose': 'precision correctness and paired native graph cost; not playback or model promotion',
              'script_sha256': sha(__file__), 'video_frame_sha256': sha(args.video_frame),
              'profile_sha256': {p: sha(args.packages/f'precision-{p}'/'profile.json') for p in policies},
              'policies': policies, 'samples': args.samples, 'rows': [], 'complete': False}
    def save():
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    for size in ['640x360', '1280x720']:
        width, height = map(int, size.split('x'))
        images = inputs(width, height, args.video_frame)
        natives = {}
        correctness = {}
        for label, model in models.items():
            natives[label] = {p: ct.models.MLModel(str(args.packages/f'precision-{p}'/f'{label}_{size}.mlpackage'),
                              compute_units=ct.ComputeUnit.CPU_AND_GPU) for p in policies}
            correctness[label] = {p: {} for p in policies}
            for name, image in images.items():
                tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float()/255
                with torch.inference_mode():
                    expected = model(tensor)[0].permute(1, 2, 0).numpy()
                for p, native in natives[label].items():
                    actual = np.asarray(native.predict({'input': image})['output'].convert('RGB')).astype(np.float32)
                    if actual.shape != expected.shape:
                        raise ValueError('output geometry mismatch')
                    delta = np.abs(actual-expected)
                    correctness[label][p][name] = {'max_rgb': float(delta.max()), 'mean_rgb': float(delta.mean()),
                                                  'pass': bool(delta.max() <= 3 and delta.mean() <= .6)}
                print(size, label, name, 'checked', flush=True)
        accepted = [(label, p) for label in models for p in policies
                    if all(row['pass'] for row in correctness[label][p].values())]
        times = {label: {p: [] for p in policies if (label, p) in accepted} for label in models}
        order_rng = random.Random(20260905)
        load_samples = []
        for i in range(args.samples+10):
            order = list(accepted)
            order_rng.shuffle(order)
            image = list(images.values())[i % len(images)]
            for label, p in order:
                start = time.perf_counter()
                output = natives[label][p].predict({'input': image})['output']
                elapsed = (time.perf_counter()-start)*1000
                if i >= 10:
                    times[label][p].append(elapsed)
            if i % 10 == 0:
                load_samples.append({'iteration': i, 'load_average': list(os.getloadavg())})
        summary = {label: {p: {'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95))}
                            for p, values in variants.items()} for label, variants in times.items()}
        report['rows'].append({'input_size': size, 'correctness': correctness, 'timings': summary,
                               'samples_ms': times, 'system_load': load_samples})
        save()
        print(size, summary, flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
