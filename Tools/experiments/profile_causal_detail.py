#!/usr/bin/env python3
"""Check recurrent Core ML conversion and measure graph latency, not playback."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail_v2 import make_model, SeparableLanczos2x


class ImageOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, state, valid):
        image, next_state = self.model(frame, state, valid)
        return image * 255, next_state


class StatefulImageOutput(torch.nn.Module):
    def __init__(self, model, initial):
        super().__init__()
        self.model = model
        self.register_buffer('history', initial.half().clone())

    def forward(self, frame, valid):
        image, next_state = self.model(frame, self.history.float(), valid)
        self.history[:, :, :, :] = next_state
        return image * 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', default=['640x360', '1280x720'])
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--architecture', choices=('causal_detail_v1', 'causal_detail_v2'), default='causal_detail_v1')
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--separable-floor', action='store_true', help='Equivalent two-pass fixed floor, v2 only')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--stateful', action='store_true', help='Keep feature history inside Core ML MLState')
    parser.add_argument('--compute-units', nargs='+', choices=('CPU_AND_GPU', 'ALL'), default=['CPU_AND_GPU', 'ALL'])
    parser.add_argument('--checkpoint', type=Path,
                        help='Trusted local causal training checkpoint; otherwise random graph probe')
    parser.add_argument('--out', type=Path, default=Path('.build/causal-detail'))
    args = parser.parse_args()
    if args.samples < 10:
        parser.error('at least ten timing samples required')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260904)
    no_history = False
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        args.architecture = checkpoint['architecture']
        args.channels, args.blocks, args.scale = (checkpoint[k] for k in ('channels', 'blocks', 'scale'))
        no_history = checkpoint['no_history']
    model = make_model(args.architecture, args.channels, args.blocks, args.scale).eval()
    if args.checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        # A zero head can be constant-folded away. Keep the entire random graph.
        torch.nn.init.normal_(model.head.weight, std=0.002)
    if args.separable_floor:
        if args.architecture != 'causal_detail_v2':
            parser.error('separable floor requires v2')
        floor = SeparableLanczos2x()
        if not torch.equal(floor.kernels, model.floor.kernels):
            raise ValueError('checkpoint fixed kernel differs; cannot substitute separable floor')
        model.floor = floor
    wrapper = ImageOutput(model).eval()
    report = {'untrained': not bool(args.checkpoint), 'purpose': 'Core ML graph correctness and latency feasibility only',
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        'checkpoint_step': checkpoint['step'] if args.checkpoint else None, 'no_history': no_history,
        'channels': args.channels, 'blocks': args.blocks, 'scale': args.scale,
        'architecture': args.architecture,
        'separable_floor': args.separable_floor,
        'stateful': args.stateful,
        'parameters': sum(p.numel() for p in model.parameters()),
        'platform': platform.platform(),
        'profiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coremltools': str(ct.__version__), 'torch': str(torch.__version__),
        'architecture_sha256': hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('architectures/causal_detail.py').read_bytes()).hexdigest(),
        'architecture_v2_sha256': hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('architectures/causal_detail_v2.py').read_bytes()).hexdigest(),
        'includes': 'synchronous Python Core ML prediction with PIL image; internal MLState' if args.stateful else 'synchronous Python Core ML prediction with PIL image and explicit feature-state copies',
        'excludes': 'video decode, browser transport, app rendering; conversion probes do not measure reconstruction quality',
        'complete': False, 'rows': []}
    for size in args.sizes:
        width, height = map(int, size.split('x'))
        pixels = np.random.default_rng(71).integers(0, 256, (height, width, 3), dtype=np.uint8)
        frame = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float() / 255
        initial = model.initial_state(frame)
        valid = torch.ones(1, 1, 1, 1)
        with torch.inference_mode():
            traced = torch.jit.trace(StatefulImageOutput(model, initial).eval(), (frame, valid), check_trace=False) if args.stateful else torch.jit.trace(wrapper, (frame, initial, valid))
        path = args.out / f'causal_ch{args.channels}_x{args.scale}_{size}.mlpackage'
        inputs = [ct.ImageType(name='input', shape=frame.shape, scale=1/255, color_layout=ct.colorlayout.RGB)]
        if not args.stateful:
            inputs.append(ct.TensorType(name='history_features', shape=initial.shape))
        inputs.append(ct.TensorType(name='valid', shape=valid.shape))
        options = {'states': [ct.StateType(wrapped_type=ct.TensorType(shape=initial.shape), name='history')]} if args.stateful else {}
        converted = ct.convert(traced, inputs=inputs,
            outputs=[ct.ImageType(name='output', color_layout=ct.colorlayout.RGB)] + ([] if args.stateful else [ct.TensorType(name='next_state')]),
            convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15, **options)
        converted.save(str(path))
        for units in [getattr(ct.ComputeUnit, name) for name in args.compute_units]:
            native = ct.models.MLModel(str(path), compute_units=units)
            # Roll each implementation's own state, rather than feeding the
            # native graph perfect Torch history. Include a mid-sequence reset.
            state_reference = initial
            state_native = native.make_state() if args.stateful else initial.numpy()
            parity = []
            with torch.inference_mode():
                for index in range(6):
                    moved = np.roll(pixels, (index, -index), axis=(0, 1)).copy()
                    current = torch.from_numpy(moved).permute(2, 0, 1)[None].float()/255
                    use_history = float(index not in (0, 4) and not no_history)
                    reset = torch.full_like(valid, use_history)
                    reference, state_reference = wrapper(current, state_reference, reset)
                    inputs = {'input': Image.fromarray(moved), 'valid': reset.numpy()}
                    if args.stateful:
                        result = native.predict(inputs, state=state_native)
                        actual_state = state_native.read_state('history')
                    else:
                        inputs['history_features'] = state_native
                        result = native.predict(inputs)
                        state_native = result['next_state']
                        actual_state = state_native
                    expected = reference[0].permute(1, 2, 0).numpy().clip(0, 255)
                    actual = np.asarray(result['output'].convert('RGB'), dtype=np.float32)
                    parity.append({'frame': index, 'history_valid': bool(use_history),
                        'max_pixel_error_255': float(np.max(np.abs(actual-expected))),
                        'max_state_error': float(np.max(np.abs(actual_state-state_reference.numpy())))})
            max_error = max(p['max_pixel_error_255'] for p in parity)
            state_error = max(p['max_state_error'] for p in parity)
            if max_error > 2 or state_error > 0.02:
                raise RuntimeError(f'conversion mismatch: image={max_error}, state={state_error}')
            inputs = {'input': Image.fromarray(pixels),
                      'valid': np.full((1, 1, 1, 1), float(not no_history), dtype=np.float32)}
            if not args.stateful:
                inputs['history_features'] = initial.numpy()
            else:
                state_native = native.make_state()
            samples = []
            for i in range(5 + args.samples):
                started = time.perf_counter()
                result = native.predict(inputs, state=state_native) if args.stateful else native.predict(inputs)
                if not args.stateful:
                    inputs['history_features'] = result['next_state']
                elapsed = (time.perf_counter()-started)*1000
                if i >= 5:
                    samples.append(elapsed)
            row = {'input': size, 'output': f'{width*args.scale}x{height*args.scale}',
                'compute_units': units.name, 'mean_ms': float(np.mean(samples)),
                'p95_ms': float(np.percentile(samples, 95)), 'samples_ms': samples,
                'conversion_max_pixel_error_255': max_error,
                'conversion_max_state_error': state_error,
                'conversion_sequence': parity,
                'state_bytes_fp32': initial.numel()*4}
            report['rows'].append(row)
            (args.out/'profile.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    report['complete'] = True
    (args.out/'profile.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
