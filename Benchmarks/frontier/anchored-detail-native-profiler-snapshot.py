#!/usr/bin/env python3
"""Compare shipping 4x and direct-area 2x Core ML graphs without changing assets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import coremltools as ct
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load
from architectures.span_arch import Conv3XC
from convert_span import ImageRange, selective_fp16
from fold_shipping_head import fold_head


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_convolutions(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Conv3XC):
            child.update_params()
            replacement = copy.deepcopy(child.eval_conv)
            if child.has_relu:
                replacement = torch.nn.Sequential(replacement, torch.nn.LeakyReLU(0.05))
            setattr(module, name, replacement)
        else:
            freeze_convolutions(child)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--expected-checkpoint-sha256', help='Reject a substituted baseline before conversion')
    ap.add_argument('--direct-checkpoint', type=Path,
                    help='Optional trained 2x checkpoint; otherwise use the exact area-folded initialization')
    ap.add_argument('--anchored-probe', action='store_true',
                    help='Also measure a nonzero untrained frozen-base detail branch; no quality claim')
    ap.add_argument('--sizes', nargs='+', default=['640x360', '1280x720'])
    ap.add_argument('--samples', type=int, default=40)
    ap.add_argument('--compute-units', nargs='+', choices=['CPU_AND_GPU', 'ALL'], default=['CPU_AND_GPU'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.samples < 20 or args.out.exists():
        ap.error('fresh output directory and at least 20 samples required')
    if args.expected_checkpoint_sha256 and digest(args.checkpoint) != args.expected_checkpoint_sha256:
        ap.error('baseline checkpoint differs from the declared SHA256')
    args.out.mkdir(parents=True)
    model, step, frames = load(args.checkpoint, 'cpu')
    if frames != 1 or model.core.upsampler[1].upscale_factor != 8:
        raise ValueError('shipping comparator must be single-frame 4x')
    if args.direct_checkpoint:
        folded, _, direct_frames = load(args.direct_checkpoint, 'cpu')
        if direct_frames != 1 or folded.core.upsampler[1].upscale_factor != 4:
            raise ValueError('direct checkpoint must be a single-frame 2x reconstruction model')
    else:
        folded = fold_head(model)
    freeze_convolutions(model)
    freeze_convolutions(folded)
    direct_label = 'direct2x_trained' if args.direct_checkpoint else 'direct2x_area'
    models = {'shipping4x': model, direct_label: folded}
    if args.anchored_probe:
        from architectures.anchored_detail import AnchoredDetail
        torch.manual_seed(20260905)
        probe = AnchoredDetail(copy.deepcopy(folded), channels=32, blocks=4).eval()
        # Keep the branch nonzero so compilation cannot erase a zero-output head.
        torch.nn.init.normal_(probe.head.weight, std=.001)
        torch.nn.init.normal_(probe.head.bias, std=.001)
        models['anchored_detail_untrained'] = probe
    report = {'purpose': 'native graph comparison at different output scales; no playback claim',
        'checkpoint_sha256': digest(args.checkpoint), 'profiler_sha256': digest(__file__),
        'direct_checkpoint_sha256': digest(args.direct_checkpoint) if args.direct_checkpoint else None,
        'fold_sha256': digest(Path(__file__).with_name('fold_shipping_head.py')),
        'torch': str(torch.__version__), 'coremltools': str(ct.__version__),
        'platform': platform.platform(), 'samples': args.samples,
        'excludes': 'capture, network, presentation; 4x output downsampling is not timed',
        'quality': 'trained checkpoint requires independent quality evaluation' if args.direct_checkpoint else
                   'area average before clamp differs from shipping RGB8 bicubic presentation; evaluate separately',
        'rows': [], 'complete': False}
    if args.anchored_probe:
        report['anchored_probe'] = {
            'code_sha256': digest(Path(__file__).resolve().parents[1] / 'architectures/anchored_detail.py'),
            'seed': 20260905, 'channels': 32, 'blocks': 4,
            'added_parameters': sum(p.numel() for p in probe.parameters() if p.requires_grad),
            'status': 'random nonzero detail branch; conversion and cost only, not trained quality'}
    def save():
        (args.out / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    for size in args.sizes:
        width, height = map(int, size.split('x'))
        if min(width, height) <= 0 or width % 2 or height % 2:
            raise ValueError('positive even dimensions required')
        rng = np.random.default_rng(20260904)
        images = [Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)) for _ in range(4)]
        example = torch.from_numpy(np.asarray(images[0]).copy()).permute(2, 0, 1)[None].float() / 255
        paths, checks = {}, {}
        for label, graph in models.items():
            wrapper = ImageRange(graph).eval()
            with torch.inference_mode():
                traced = torch.jit.trace(wrapper, example)
                expected = wrapper(example)[0].permute(1, 2, 0).numpy()
            converted = ct.convert(traced, inputs=[ct.ImageType(name='input', shape=example.shape,
                color_layout=ct.colorlayout.RGB, scale=1/255)],
                outputs=[ct.ImageType(name='output', color_layout=ct.colorlayout.RGB)],
                convert_to='mlprogram', compute_precision=selective_fp16(),
                minimum_deployment_target=ct.target.macOS15)
            converted.user_defined_metadata['lucid.checkpoint_sha256'] = digest(
                args.direct_checkpoint if label != 'shipping4x' and args.direct_checkpoint else args.checkpoint)
            converted.user_defined_metadata['lucid.output_scale'] = '4' if label == 'shipping4x' else '2'
            converted.user_defined_metadata['lucid.transformation'] = (
                'untrained frozen-base detail probe' if label == 'anchored_detail_untrained' else
                'area phase folding' if label == 'direct2x_area' else 'trained weights')
            path = args.out / f'{label}_{size}.mlpackage'
            converted.save(str(path))
            paths[label] = path
            checks[label] = expected
        for units in args.compute_units:
            native = {label: ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
                      for label, path in paths.items()}
            errors, samples, output_hashes = {}, {k: [] for k in models}, {}
            for label, graph in native.items():
                output = np.asarray(graph.predict({'input': images[0]})['output'].convert('RGB')).astype(np.float32)
                expected = checks[label]
                if output.shape != expected.shape:
                    raise ValueError('native output dimensions changed')
                delta = np.abs(output - expected)
                errors[label] = {'max_rgb': float(delta.max()), 'mean_rgb': float(delta.mean()),
                                 'output_shape': list(output.shape)}
                if delta.max() > 3 or delta.mean() > 0.6:
                    raise ValueError(f'native conversion mismatch: {label} {units} {errors[label]}')
                output_hashes[label] = hashlib.sha256(output.tobytes()).hexdigest()
            for iteration in range(args.samples + 10):
                labels = list(native)
                if iteration % 2:
                    labels.reverse()
                for label in labels:
                    started = time.perf_counter()
                    output = native[label].predict({'input': images[iteration % len(images)]})['output']
                    elapsed = (time.perf_counter() - started) * 1000
                    if iteration >= 10:
                        samples[label].append(elapsed)
            report['rows'].append({'input_size': size, 'compute_units': units,
                'correctness': errors, 'output_hashes': output_hashes,
                'timings': {label: {'mean_ms': float(np.mean(values)), 'p95_ms': float(np.percentile(values, 95)),
                    'samples_ms': values} for label, values in samples.items()}})
            save()
            print(size, units, {k: np.mean(v) for k, v in samples.items()}, flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
