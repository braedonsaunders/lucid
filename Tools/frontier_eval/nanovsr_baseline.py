#!/usr/bin/env python3
"""Score reviewed NanoVSR weights with explicit bidirectional and causal contexts."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_sequences import Scorer, average, decode, digest, load, pixel_digest, present_4x_at_2x, save, summarize


class CausalEndpoint(nn.Module):
    """Exact last output of the official model on an ever-growing input prefix.

    The backward branch sees only the current frame. No fine-tuning or claim
    that boundary-only usage matches the training distribution is implied.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, history):
        m = self.model
        feature = m.feat_extract(image)
        state = m.forward_net(feature + history)
        backward = m.backward_net(feature)
        fused = m.fusion(torch.cat((state, backward), dim=1))
        output = m.conv_last(m.upsample2(m.upsample1(fused)))
        return output + F.interpolate(image, scale_factor=4, mode='bilinear', align_corners=False), state


def load_reviewed(source, weights, source_sha256, weights_sha256, device):
    if digest(source) != source_sha256 or digest(weights) != weights_sha256:
        raise ValueError('reviewed upstream source/weights changed')
    spec = importlib.util.spec_from_file_location('lucid_reviewed_nanovsr', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model = module.NanoVSR(num_feat=48, num_blocks=12).eval()
    model.load_state_dict(state, strict=True)
    model.switch_to_deploy()
    return model.eval().to(device)


@torch.inference_mode()
def verify_endpoint(model, device):
    generator = torch.Generator().manual_seed(20260905)
    sequence = torch.rand(1, 4, 3, 12, 16, generator=generator).to(device)
    causal = CausalEndpoint(model)
    state = sequence.new_zeros(1, model.num_feat, 12, 16)
    errors = []
    for index in range(4):
        value, state = causal(sequence[:, index], state)
        expected = model(sequence[:, :index+1])[:, -1]
        errors.append(float((value-expected).abs().max()))
    if max(errors) > 1e-4:
        raise ValueError('causal endpoint differs from official prefix output')
    return {'seed': 20260905, 'prefix_lengths': [1,2,3,4], 'max_abs_error_per_prefix': errors,
            'tolerance': 1e-4, 'scope': 'FP32 causal wrapper equivalence on random 12x16 input, not image-quality validation'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--source-sha256', required=True)
    ap.add_argument('--weights-sha256', required=True)
    ap.add_argument('--shipping', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.out.exists(): ap.error('fresh comparison output required')
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    if not manifest or len({row['id'] for row in manifest}) != len(manifest):
        raise ValueError('unique nonempty sequence manifest required')
    for row in manifest:
        for side in ('reference', 'degraded'):
            if digest(row[side]) != row[side+'_sha256']: raise ValueError('changed input media')
    model = load_reviewed(args.source, args.weights, args.source_sha256, args.weights_sha256, args.device)
    causal = CausalEndpoint(model)
    shipping, _, count = load(args.shipping, args.device)
    if count != 1: raise ValueError('single-frame shipping weights required')
    scorer = Scorer(args.device)
    report = {'purpose': '2026 pretrained baseline; not release promotion or streaming latency evidence',
              'upstream': 'https://github.com/filippawlicki/nanovsr/tree/b48e2bad89d01e22226eb8613bee25cb45623fae',
              'source_sha256': digest(args.source), 'weights_sha256': digest(args.weights),
              'shipping_sha256': digest(args.shipping), 'manifest_sha256': digest(args.manifest),
              'evaluator_sha256': digest(__file__), 'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
              'torch': str(torch.__version__), 'device': args.device, 'precision': 'FP32, no autocast',
              'adapter': 'all 4x outputs clamp/round RGB8 then PIL bicubic to declared 2x reference size',
              'context': {'nanovsr_bidirectional': 'entire 16-frame excerpt, future frames included',
                          'nanovsr_causal': 'past/current only, fresh zero forward state at each excerpt',
                          'shipping': 'single frame'},
              'limitations': ['development sources, upstream pretraining overlap unverified',
                              'CUDA excerpt timing is not Mac inference or capture-to-display latency',
                              'no temporal filtering work or temporal promotion decision in this screen'],
              'decoded_rgb_sha256': {}, 'rows': [], 'complete': False}
    report['causal_endpoint_verification'] = verify_endpoint(model, args.device)
    with torch.inference_mode():
        for row in manifest:
            if row['frames'] != 16: raise ValueError('frozen 16-frame excerpts required')
            sources, references = [decode(row[key], row['frames']) for key in ('degraded', 'reference')]
            if any(r.size != (s.width*2, s.height*2) for s, r in zip(sources, references)):
                raise ValueError('explicit 2x presentation comparison required')
            report['decoded_rgb_sha256'][row['id']] = {key: [pixel_digest(im) for im in values]
                    for key, values in [('degraded', sources), ('reference', references)]}
            x = torch.from_numpy(np.stack(sources).copy()).permute(0,3,1,2).to(args.device).float()/255
            for variant in ('shipping', 'nanovsr_bidirectional', 'nanovsr_causal'):
                if args.device == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                if variant == 'nanovsr_bidirectional': output = model(x[None])[0]
                else:
                    outputs = []
                    state = x.new_zeros(1, model.num_feat, x.shape[-2], x.shape[-1])
                    for frame in x:
                        if variant == 'shipping': y = shipping(frame[None])
                        else: y, state = causal(frame[None], state)
                        outputs.append(y[0])
                    output = torch.stack(outputs)
                if args.device == 'cuda': torch.cuda.synchronize()
                milliseconds = (time.perf_counter()-start)*1000/len(sources)
                pixels = output.clamp(0,1).permute(0,2,3,1).cpu().numpy()
                del output
                frames = []
                for i in range(0, len(sources), 4):
                    image = Image.fromarray(np.rint(pixels[i]*255).astype(np.uint8))
                    image = present_4x_at_2x(image, sources[i].size)
                    frames.append({'index': i, **scorer.spatial(image, references[i])})
                report['rows'].append({'source_id': row['source_id'], 'sequence_id': row['id'], 'variant': variant,
                    'metrics': average([{k:v for k,v in f.items() if k != 'index'} for f in frames]),
                    'spatial_frames': frames, 'excerpt_prediction_ms_per_frame': milliseconds})
                report['summary'] = summarize(report['rows'])
                save(args.out, report)
                print(row['id'], variant, milliseconds, flush=True)
    report['complete'] = True
    save(args.out, report)


if __name__ == '__main__': main()
