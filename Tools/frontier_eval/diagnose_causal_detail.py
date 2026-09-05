#!/usr/bin/env python3
"""Separate spatial-floor and history effects; post-hoc variants are not trained models."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail import CausalDetail
from evaluate_sequences import Scorer, average, decode, digest, save, summarize, temporal_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['architecture'] != 'causal_detail_v1' or checkpoint['no_history']:
        raise ValueError('diagnostic requires a trusted recurrent v1 checkpoint')
    model = CausalDetail(checkpoint['channels'], checkpoint['blocks'], checkpoint['scale']).eval().to(args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    manifest = json.loads(args.manifest.read_text())
    for sequence in manifest:
        for side in ('reference', 'degraded'):
            if digest(sequence[side]) != sequence[f'{side}_sha256']:
                raise ValueError('sequence media changed')
    scorer = Scorer(torch.device(args.device))
    report = {'purpose': __doc__, 'checkpoint_sha256': digest(args.checkpoint),
        'manifest_sha256': digest(args.manifest), 'sequences': manifest,
        'limitations': ['Post-hoc interpolation replacement was not trained and is not a promotion candidate.',
            'Forced reset isolates inference history, not the effect of training with history.',
            'Short development sequences; spatial frames 0/4/8/12, all frames for unwarped temporal metrics.'],
        'rows': [], 'complete': False}
    for sequence in manifest:
        sources = decode(sequence['degraded'], sequence['frames'])
        references = decode(sequence['reference'], sequence['frames'])
        variants = {k: [] for k in ('bilinear', 'lanczos', 'causal', 'forced-reset', 'lanczos-plus-residual')}
        state = None
        for index, (source, reference) in enumerate(zip(sources, references)):
            x = torch.from_numpy(np.asarray(source, dtype=np.float32)/255).permute(2, 0, 1)[None].to(args.device)
            if state is None:
                state = model.initial_state(x)
            output, state = model(x, state, x.new_full((1, 1, 1, 1), float(index > 0)))
            reset, _ = model(x, state, x.new_zeros(1, 1, 1, 1))
            base = F.interpolate(x, scale_factor=checkpoint['scale'], mode='bilinear', align_corners=False)
            if tuple(output.shape[-2:]) != (reference.height, reference.width):
                raise ValueError('wrong output scale')
            lanczos = source.resize(reference.size, Image.Resampling.LANCZOS)
            def array(tensor):
                return tensor[0].permute(1, 2, 0).cpu().numpy()*255
            def image(values):
                return Image.fromarray(values.clip(0, 255).round().astype(np.uint8))
            variants['bilinear'].append(image(array(base)))
            variants['lanczos'].append(lanczos)
            variants['causal'].append(image(array(output)))
            variants['forced-reset'].append(image(array(reset)))
            # Keep the learned residual exactly; change only interpolation.
            variants['lanczos-plus-residual'].append(image(np.asarray(lanczos, dtype=np.float32)+array(output-base)))
        for label, outputs in variants.items():
            metrics = average([scorer.spatial(outputs[i], references[i]) for i in range(0, len(outputs), 4)])
            metrics.update(temporal_metrics(outputs, references))
            report['rows'].append({'sequence_id': sequence['id'], 'source_id': sequence['source_id'],
                'variant': label, 'metrics': metrics})
            report['summary'] = summarize(report['rows'])
            save(args.report, report)
            print(sequence['id'], label, json.dumps(metrics), flush=True)
    report['complete'] = True
    save(args.report, report)


if __name__ == '__main__':
    main()
