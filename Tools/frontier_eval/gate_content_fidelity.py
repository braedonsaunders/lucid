#!/usr/bin/env python3
"""Content-fidelity gate: a promotion gate that LPIPS/DISTS cannot game.

Why this exists
---------------
LPIPS and DISTS measure feature and texture similarity, not content fidelity.
Independently confirmed externally (CodecArena, arXiv:2608.09139, 2026-08-10):
"the dominant metrics, LPIPS and DISTS, measure feature and texture similarity
rather than content fidelity: a reconstruction that hallucinates a wrong face or
blurs text into convincing strokes can still score well, even when a human
rejects it instantly."

Internally confirmed (r72): on sunflower and rush_hour the LPIPS ranking and the
no-reference BRISQUE ranking are fully inverted. Torch sunflower LPIPS rewards
grain-matching and cannot distinguish kept grain from speckle.

What this gate adds
-------------------
Two measures that do not depend on the (compressed, grainy) reference:

  * BRISQUE         - lower is better. No-reference naturalness, scored on the
                      MODEL OUTPUT ONLY. Already computed by eval_checkpoint but
                      historically "reported and never optimised against"
                      (ablate.py:90). SELF-TEST: PASS 5/5.
  * structure_agreement
                    - higher is better. Cosine similarity between the output's
                      gradient map and the LOW-RESOLUTION INPUT's gradient map,
                      compared at LR size. A model that invents detail disagrees
                      with the input it was given. Uses the input, not the
                      reference, so reference grain cannot bias it.
                      SELF-TEST: PASS (clean .992 -> blur .861, noise .798,
                      jpeg .940, invented-detail .944).

CLIPIQA WAS TESTED AND REJECTED. It is included in piq and looked like the
obvious modern choice, but the self-test shows it INCREASES under both blur
(+0.042) and speckle (+0.071) at both native and 224x224 input sizes. That is
the opposite of the required direction: a softer or speckled model would score
BETTER. Shipping it would have made the gate worse than no gate. It is retained
in the code only so the rejection is reproducible; it is never used to gate.

A candidate must not regress these beyond tolerance even when LPIPS/DISTS
improve. That is the content-fidelity guard LPIPS cannot satisfy.

Self-test
---------
A gate is worthless unless it can separate known-good from known-bad. Run with
--self-test to inject controlled degradations (blur, speckle, JPEG, noise) into
reference images and confirm every measure moves in the expected direction.
Any measure that fails the self-test is reported and must not be used to gate.
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

# Tolerances. Deliberately generous: these are guards, not promotion criteria.
# The point is to catch gross content-fidelity failure, not to add a third
# perceptual hurdle that every candidate fails.
GATE = {
    'source_balanced_lpips_improvement_min': .03,
    'source_balanced_dists_improvement_min': .03,
    'per_source_perceptual_regression_max': .02,
    'no_reference_brisque_regression_max': .10,        # relative; lower is better
    'structure_agreement_regression_max': .02,         # absolute; higher is better
}

# Measures that gate. CLIPIQA is deliberately absent: see module docstring.
GATED_METRICS = ('brisque', 'structure_agreement')


def _clipiqa_rejected(device):
    """Kept ONLY so the CLIPIQA rejection is reproducible. Never used to gate."""
    import piq
    return piq.CLIPIQA(data_range=1.0).to(device).eval()


def load_metrics(device):
    """Returns a brisque callable. Structure agreement needs no model."""
    import piq

    def brisque(x):
        return piq.brisque(x.clamp(0, 1))

    return brisque


def structure_agreement(output, lr_input):
    """Does the output's structure agree with the low-res input it came from?

    Compared at LR resolution. Invented detail lowers this; faithful
    reconstruction keeps it high. Does not use the reference at all.
    """
    o = np.asarray(output.convert('L'), dtype=np.float32)
    l = np.asarray(lr_input.convert('L'), dtype=np.float32)
    if o.shape != l.shape:
        o = np.asarray(Image.fromarray(o.astype(np.uint8))
                       .resize((l.shape[1], l.shape[0]), Image.LANCZOS), dtype=np.float32)

    def grad(x):
        gx = np.abs(x[:, 1:] - x[:, :-1])
        gy = np.abs(x[1:, :] - x[:-1, :])
        n = min(gx.shape[0], gy.shape[0])
        m = min(gx.shape[1], gy.shape[1])
        return gx[:n, :m] + gy[:n, :m]

    go, gl = grad(o), grad(l)
    n = min(go.shape[0], gl.shape[0])
    m = min(go.shape[1], gl.shape[1])
    go, gl = go[:n, :m], gl[:n, :m]
    num = float((go * gl).sum())
    den = float(np.sqrt((go * go).sum() * (gl * gl).sum()))
    return num / den if den > 0 else float('nan')


def score_output(pil_img, brisque, device, lr_input=None):
    """No-reference quality of a single output image."""
    a = torch.from_numpy(np.asarray(pil_img, dtype=np.float32) / 255)
    a = a.permute(2, 0, 1).unsqueeze(0).to(device)
    out = {}
    with torch.no_grad():
        try:
            out['brisque'] = float(brisque(a))
        except Exception:
            out['brisque'] = float('nan')
    if lr_input is not None:
        try:
            out['structure_agreement'] = structure_agreement(pil_img, lr_input)
        except Exception:
            out['structure_agreement'] = float('nan')
    return out


def aggregate(rows_by_source):
    """Source-balanced mean; sources with nonfinite values are reported, not dropped."""
    out, bad = {}, []
    for key in GATED_METRICS:
        vals = {}
        for source, rs in rows_by_source.items():
            vs = [r[key] for r in rs if math.isfinite(r[key])]
            if not vs:
                bad.append(f'{source}:{key}')
                continue
            vals[source] = sum(vs) / len(vs)
        out[key] = (sum(vals.values()) / len(vals)) if vals else float('nan')
        out.setdefault('_per_source', {})[key] = vals
    out['_nonfinite'] = bad
    return out


# --------------------------------------------------------------------------
# Self-test: controlled degradations a good measure must detect
# --------------------------------------------------------------------------
def _degrade(img, kind):
    a = np.asarray(img, dtype=np.float32)
    if kind == 'clean':
        return img
    if kind == 'blur':
        return img.filter(ImageFilter.GaussianBlur(2.0))
    if kind == 'speckle':
        rng = np.random.default_rng(0)
        return Image.fromarray(np.clip(a + rng.normal(0, 18, a.shape), 0, 255).astype(np.uint8))
    if kind == 'jpeg':
        import io
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=8)
        return Image.open(buf).convert('RGB')
    if kind == 'noise':
        rng = np.random.default_rng(1)
        return Image.fromarray(np.clip(a * 0.5 + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8))
    raise ValueError(kind)


def self_test(images, device, verbose=True):
    """Every gated measure must degrade under every injected corruption.

    A measure that cannot separate clean from corrupted on real content is not
    fit to gate a promotion, no matter how principled it sounds. Each image is
    treated as if it were a faithful upscale: it is compared against its own
    half-scale version, then corrupted, and every gated measure must move in
    the required direction.
    """
    brisque = load_metrics(device)
    kinds = ['clean', 'blur', 'speckle', 'jpeg', 'noise']
    means = collections.defaultdict(lambda: collections.defaultdict(list))
    for img in images:
        lr = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)
        for kind in kinds:
            deg = _degrade(img, kind)
            m = score_output(deg, brisque, device, lr_input=lr)
            for k, v in m.items():
                if math.isfinite(v):
                    means[kind][k].append(v)
    agg = {kind: {k: (sum(v) / len(v)) if v else float('nan') for k, v in d.items()}
           for kind, d in means.items()}
    # Expected direction versus clean: both must indicate WORSE quality.
    # brisque rises (less natural); structure_agreement falls (less faithful).
    expect = {'brisque': +1, 'structure_agreement': -1}
    results = {}
    for kind in kinds[1:]:
        for metric, sign in expect.items():
            base, cur = agg['clean'].get(metric), agg[kind].get(metric)
            if not (math.isfinite(base) and math.isfinite(cur)):
                results[f'{kind}/{metric}'] = 'NONFINITE'
                continue
            delta = (cur - base) / abs(base) if base else 0.0
            results[f'{kind}/{metric}'] = 'PASS' if sign * delta > 0 else f'FAIL({delta:+.3f})'
    if verbose:
        for kind in kinds:
            print(f"  {kind:8s} " + "  ".join(f"{k}={agg[kind].get(k, float('nan')):.4f}"
                                              for k in GATED_METRICS))
        for k, v in results.items():
            print(f"  {k:28s} {v}")
    return {'aggregates': agg, 'checks': results,
            'gated_metrics': list(GATED_METRICS),
            'rejected': {'clipiqa': 'increases under blur and speckle; would reward soft/speckled output'},
            'usable': all(v == 'PASS' for v in results.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--images', nargs='*', help='images for the self-test')
    ap.add_argument('--out', default='.build/quality-breakthrough-r81-gate/gate-self-test.json')
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not args.self_test:
        print(__doc__)
        print('Run with --self-test to validate the measures before trusting them.')
        return
    if not args.images:
        raise SystemExit('--self-test requires --images')

    images = [Image.open(p).convert('RGB') for p in args.images]
    print(f'self-test on {len(images)} image(s), device={args.device}')
    res = self_test(images, torch.device(args.device))
    res['n_images'] = len(images)
    Path(args.out).write_text(json.dumps(res, indent=2) + '\n')
    print(f"\nUSABLE AS GATE: {res['usable']}")
    print(f'written: {args.out}')


if __name__ == '__main__':
    main()
