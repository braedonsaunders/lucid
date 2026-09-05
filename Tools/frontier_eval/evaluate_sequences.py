#!/usr/bin/env python3
"""Paired, source-balanced video screening with spatial and temporal guardrails.

Short sequences are development screening, not a release promotion benchmark.
Metrics never resize a model output to hide an incorrect reconstruction scale.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load, bands, correlation, detail_ratio


def load_efrlfn(repository, weights, device):
    """Load reviewed upstream modules without replacing Python's stdlib code.

Upstream uses the package name `code`. Restore all temporary aliases before
returning, including on error. No pickle object execution is needed for weights.
    """
    module_names = ['code', 'code.utils', 'code.blocks', 'code.model']
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        for name, relative in zip(module_names, ['__init__.py', 'utils.py', 'blocks.py', 'model.py']):
            path = Path(repository) / 'code' / relative
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            if '.' in name:
                setattr(sys.modules['code'], name.split('.')[1], module)
            spec.loader.exec_module(module)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model = sys.modules['code.model'].EfRLFN(upscale=4)
        model.load_state_dict(state, strict=True)
        return model.eval().to(device), 'published', 1
    finally:
        for name in module_names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def decode(path, count):
    with tempfile.TemporaryDirectory(prefix='lucid-sequence-') as directory:
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
            '-frames:v', str(count), '-fps_mode', 'passthrough',
            str(Path(directory) / '%04d.png')], check=True, timeout=120)
        result = []
        for file in sorted(Path(directory).glob('*.png')):
            with Image.open(file) as image:
                result.append(image.convert('RGB').copy())
    if len(result) != count:
        raise ValueError(f'{path}: expected {count} frames, decoded {len(result)}')
    return result


def temporal_metrics(outputs, references):
    """Reference-static flicker and temporal error over the entire frame.

Residual change subtracts the reference's own change, but is not optical-flow
compensated. Report it honestly alongside detail fidelity; neither temporal
number alone is evidence of good reconstruction.
    """
    out = [np.asarray(i.convert('L'), dtype=np.float32) for i in outputs]
    ref = [np.asarray(i.convert('L'), dtype=np.float32) for i in references]
    flicker, residual, coverage = [], [], []
    for t in range(1, len(out)):
        delta_o, delta_r = out[t] - out[t-1], ref[t] - ref[t-1]
        mask = np.abs(delta_r) < 1.2
        coverage.append(float(mask.mean()))
        if mask.sum() >= 1000:
            flicker.append(float(np.abs(delta_o)[mask].mean()))
        residual.append(float(np.abs(delta_o - delta_r).mean()))
    return {'static_flicker': float(np.mean(flicker)) if flicker else None,
            'static_coverage': float(np.mean(coverage)),
            'temporal_residual_l1': float(np.mean(residual))}


class Scorer:
    def __init__(self, device):
        import lpips
        import piq
        self.device = device
        self.lpips = lpips.LPIPS(net='alex', verbose=False).eval().to(device)
        self.dists = piq.DISTS().eval().to(device)

    @torch.inference_mode()
    def spatial(self, output, reference):
        if output.size != reference.size:
            raise ValueError(f'output/reference size mismatch: {output.size}, {reference.size}')
        def tensor(image):
            return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        a, b = tensor(output), tensor(reference)
        y, r = (np.asarray(i.convert('L'), dtype=np.float64) for i in (output, reference))
        return {'lpips': float(self.lpips(a * 2 - 1, b * 2 - 1)),
                'dists': float(self.dists(a, b)),
                'psnr_y': float(10 * np.log10(255**2 / max(np.mean((y-r)**2), 1e-9))),
                'detail_energy': detail_ratio(output, reference),
                'fine_correlation': correlation(bands(y)[0], bands(r)[0])}


def average(rows):
    return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            if any(r[key] is not None for r in rows) else None for key in rows[0]}


def summarize(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['variant']][row['source_id']].append(row['metrics'])
    result = {}
    for label, sources in grouped.items():
        per_source = {s: average(values) for s, values in sources.items()}
        result[label] = {'sources': per_source, 'source_balanced': average(list(per_source.values()))}
    return result


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--checkpoint', nargs=2, action='append', default=[], metavar=('LABEL', 'PATH'))
    ap.add_argument('--causal', nargs=2, action='append', default=[], metavar=('LABEL', 'PATH'))
    ap.add_argument('--efrlfn', nargs=3, action='append', default=[], metavar=('LABEL', 'REPOSITORY', 'WEIGHTS'))
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--spatial-stride', type=int, default=4)
    args = ap.parse_args()
    labels = [label for label, _ in args.checkpoint + args.causal] + [label for label, _, _ in args.efrlfn]
    if not labels or len(set(labels)) != len(labels) or 'lanczos' in labels or args.spatial_stride < 1:
        ap.error('unique checkpoint labels excluding lanczos, and a positive stride, required')
    manifest = json.loads(args.manifest.read_text())
    if not manifest or len({r['id'] for r in manifest}) != len(manifest):
        ap.error('manifest must contain unique sequences')
    for seq in manifest:
        for side in ('reference', 'degraded'):
            if digest(seq[side]) != seq[f'{side}_sha256']:
                raise ValueError(f"changed media: {seq['id']}/{side}")
    device = torch.device(args.device)
    scorer = Scorer(device)
    models = {label: load(path, device) for label, path in args.checkpoint}
    models.update({label: load_efrlfn(repo, weights, device) for label, repo, weights in args.efrlfn})
    for label, path in args.causal:
        from architectures.causal_detail_v2 import make_model
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model(state['architecture'], state['channels'], state['blocks'], state['scale']).eval().to(device)
        model.load_state_dict(state['model'], strict=True)
        model.no_history = state['no_history']
        models[label] = model, state['step'], 0
    report = {'schema': 1, 'purpose': 'short development screening; no release promotion claim',
        'manifest_sha256': digest(args.manifest), 'sequences': manifest,
        'checkpoint_sha256': {label: digest(path) for label, path in args.checkpoint + args.causal},
        'external_models': {label: {'weights_sha256': digest(weights),
            'code_sha256': {str(p.relative_to(repo)): digest(p) for p in sorted(Path(repo).glob('code/*.py'))},
            'provenance': 'https://github.com/EvgeneyBogatyrev/EfRLFN',
            'training_overlap': 'not verified; comparison is a development baseline, not an independent generalization claim'}
            for label, repo, weights in args.efrlfn},
        'torch': str(torch.__version__), 'device': str(device),
        'spatial_stride': args.spatial_stride,
        'limitations': ['16-frame excerpts do not establish long-duration stability',
          'temporal residual is not flow-compensated',
          'raw sources were excluded by identity from documented training inputs; legacy bank provenance is incomplete',
          'spatial metrics use a deterministic frame subset; temporal metrics use every consecutive frame'],
        'rows': [], 'complete': False}
    for seq in manifest:
        sources = decode(seq['degraded'], seq['frames'])
        references = decode(seq['reference'], seq['frames'])
        for label in ['lanczos', *models]:
            if label == 'lanczos':
                outputs = [s.resize(r.size, Image.Resampling.LANCZOS) for s, r in zip(sources, references)]
            else:
                model, step, history = models[label]
                tensors = [torch.from_numpy(np.asarray(s, dtype=np.float32) / 255).permute(2, 0, 1) for s in sources]
                outputs = []
                with torch.inference_mode():
                    recurrent_state = None
                    for i in range(len(sources)):
                        if history == 0:
                            x = tensors[i][None].to(device)
                            if recurrent_state is None:
                                recurrent_state = model.initial_state(x)
                            valid = x.new_full((1, 1, 1, 1), float(i > 0 and not model.no_history))
                            y, recurrent_state = model(x, recurrent_state, valid)
                        else:
                            x = torch.cat([tensors[max(0, i-k)] for k in range(history-1, -1, -1)], dim=0)[None].to(device)
                            y = model(x)
                        y = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                        outputs.append(Image.fromarray((y * 255).round().astype(np.uint8)))
            if any(o.size != r.size for o, r in zip(outputs, references)):
                raise ValueError('wrong model output scale')
            frames = [{'index': i, **scorer.spatial(outputs[i], references[i])}
                      for i in range(0, len(outputs), args.spatial_stride)]
            metrics = average([{k: v for k, v in f.items() if k != 'index'} for f in frames])
            metrics.update(temporal_metrics(outputs, references))
            report['rows'].append({'sequence_id': seq['id'], 'source_id': seq['source_id'],
                                  'variant': label, 'metrics': metrics, 'spatial_frames': frames})
            report['summary'] = summarize(report['rows'])
            save(args.report, report)
            print(seq['id'], label, json.dumps(metrics), flush=True)
    report['complete'] = True
    save(args.report, report)


if __name__ == '__main__':
    main()
