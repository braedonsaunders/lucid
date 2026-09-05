#!/usr/bin/env python3
"""Training-source-only probe of sparse activation control; no weights are trained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_checkpoint import load, bands, correlation
from architectures.subspace_adapter import fuse_convolutions
from fold_shipping_head import fold_head
from build_causal_bank import validate_sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(bank, manifest):
    """One fixed first frame per training source, with verified sequence bytes."""
    sources = {s['id']: s for s in manifest['sources']}
    chosen = {}
    for sequence in sorted(manifest['sequences'], key=lambda s: s['id']):
        source = sources[sequence['source_id']]
        if any(sequence[k] != source[k] for k in ('split', 'family')) or sequence['source_sha256'] != source['sha256']:
            raise ValueError('source/sequence provenance disagrees')
        if source['split'] != 'train' or source['id'] in chosen:
            continue
        path = bank / sequence['file']
        if digest(path) != sequence['sha256']:
            raise ValueError('changed sequence bytes')
        with np.load(path, allow_pickle=False) as data:
            lr, hr = data['lr'][0].copy(), data['hr'][0].copy()
        if lr.dtype != np.uint8 or hr.dtype != np.uint8 or hr.shape[:2] != (lr.shape[0]*2, lr.shape[1]*2):
            raise ValueError('2x RGB8 pair required')
        chosen[source['id']] = (lr, hr, sequence)
    if set(chosen) != {s['id'] for s in sources.values() if s['split'] == 'train'}:
        raise ValueError('missing training sources')
    return chosen


def fine(output, reference):
    def gray(a):
        return np.asarray(Image.fromarray(a).convert('L'), dtype=np.float64)
    return correlation(bands(gray(output))[0], bands(gray(reference))[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--init', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('fresh output required')
    bank_hash = digest(args.bank/'manifest.json')
    shipping_hash = digest(args.init)
    if bank_hash != 'b11ccdaba5ed388ec615da4270058b6441369fb6325a6e5e6680472e98d6d2f0' or shipping_hash != 'fde6c7c9866f55a24f8b2923420344758e7c2684930ba239c974b4682ceb6e65':
        raise ValueError('frozen shipping and training bank required')
    torch.set_num_threads(4)
    torch.manual_seed(20260905)
    manifest = json.loads((args.bank/'manifest.json').read_text())
    validate_sources(manifest['sources'])
    data = samples(args.bank, manifest)
    model, _, frames = load(args.init, args.device)
    if frames != 1:
        raise ValueError('single-frame shipping required')
    model = fold_head(model)
    fuse_convolutions(model)
    model.eval()
    blocks = {name: module for name, module in model.named_modules()
              if name.startswith('core.block_') and name.count('.') == 1}
    if len(blocks) != 6:
        raise ValueError('expected six shipping blocks')
    stats = {name: [] for name in blocks}
    handles = []
    def observe(name):
        def hook(module, inputs, output):
            x = output[0].float()
            stats[name].append({'absolute': x.abs().mean((0, 2, 3)).cpu(),
                                'energy': x.square().mean((0, 2, 3)).cpu()})
        return hook
    def tensor(lr):
        return torch.from_numpy(lr).permute(2, 0, 1)[None].to(args.device).float()/255
    with torch.inference_mode():
        try:
            handles = [module.register_forward_hook(observe(name)) for name, module in blocks.items()]
            for lr, _, _ in data.values():
                model(tensor(lr))
        finally:
            for handle in handles:
                handle.remove()
        selections, concentration = {}, {}
        rng = np.random.default_rng(20260905)
        for name, values in stats.items():
            magnitude = torch.stack([v['absolute'] for v in values]).mean(0)
            energy = torch.stack([v['energy'] for v in values]).mean(0)
            order = magnitude.argsort(descending=True)
            selections[name] = {'top': order[:4].tolist(), 'bottom': order[-4:].tolist(),
                                'random': rng.choice(len(order), size=4, replace=False).tolist()}
            concentration[name] = {'channels': len(order), 'mean_absolute': magnitude.tolist(),
                'mean_squared': energy.tolist(), 'top_absolute_ranked_energy_fraction': {
                    str(k): float(energy[order[:k]].sum()/energy.sum().clamp_min(1e-30)) for k in (1, 4, 8)}}
        rows = []
        def intervene(indices):
            def hook(module, inputs, output):
                main = output[0].clone()
                main[:, indices] *= .95
                return (main, *output[1:])
            return hook
        for source, (lr, hr, sequence) in data.items():
            x = tensor(lr)
            baseline = model(x)
            for strategy in ('none', 'top', 'bottom', 'random'):
                handles = []
                try:
                    if strategy != 'none':
                        handles = [module.register_forward_hook(intervene(selections[name][strategy]))
                                   for name, module in blocks.items()]
                    output = model(x)
                    if strategy == 'none' and not torch.equal(output, baseline):
                        raise ValueError('unmodified repeat is not identical')
                    rgb = output[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                    rows.append({'source_id': source, 'sequence_id': sequence['id'], 'frame': 0,
                        'strategy': strategy, 'output_change_l1': float((output-baseline).abs().mean()),
                        'reference_mse_rgb8': float(np.square(rgb.astype(np.float64)-hr).mean()),
                        'fine_correlation': fine(rgb, hr)})
                finally:
                    for handle in handles:
                        handle.remove()
            print(source, flush=True)
    result = {'purpose': 'training-only mechanism diagnostic; no validation or quality-promotion claim',
        'source_paper': 'https://arxiv.org/html/2609.03813v1',
        'difference_from_paper': 'convolutional block outputs, offline one-frame-per-source ranking, fixed 5% attenuation; no VAE, DiT, learned controller or SPARK reproduction',
        'bank_sha256': bank_hash, 'shipping_sha256': shipping_hash,
        'code_sha256': digest(__file__), 'device': args.device, 'torch': torch.__version__,
        'selection': 'first sorted sequence, frame0 for every train source; no validation sources',
        'sources': {source: {'sequence_id': seq['id'], 'sha256': seq['sha256']}
                    for source, (_, _, seq) in data.items()},
        'intervention': 'all six main block outputs simultaneously; gain .95 on four selected channels; other tuple outputs unchanged',
        'concentration': concentration, 'selections': selections, 'rows': rows,
        'complete': True, 'weights_modified': False}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
