#!/usr/bin/env python3
"""CSF-weighted, display-resolution perceptual metric and loss weighting.

Why
---
Measured in r83: 74% of the model's error power sits above 0.10 cyc/px, where the
LR input carries only ~2.5% of its energy. Error per unit reference energy rises
monotonically across bands (0.01 -> 0.95, ~95x). We are grading the model hardest
where the input has least information, which creates gradient pressure to invent
high-frequency content — i.e. hallucination.

In r84 a rejected checkpoint's two fatal regressions collapsed when unreachable
bands were excluded (sunflower +18.12% -> +0.66%, rush_hour +4.53% -> +0.38%).
So the verdict on that model depends on which frequencies are graded.

What this provides
------------------
1. `csf_weights(shape, viewing_distance_px)` — a contrast-sensitivity weighting
   in the radial frequency domain, plus an optional reachability rolloff.
2. `csf_filter(pil_img, ...)` — apply the weighting to an image (for scoring).
3. `csf_l1(a, b, ...)` — a frequency-weighted L1 usable as a TRAINING loss.

Design notes / honesty
----------------------
* The CSF is a Mannos-Sakrison-style parabola in log frequency. It is a standard
  approximation, not a calibrated model of any particular display.
* `reachable_cutoff` is an EXPERIMENTAL switch, not a principled constant. With
  it enabled the metric deliberately stops penalising bands the LR cannot
  support. That is a hypothesis under test, and it is reported as such.
* Self-test (`--self-test`) must pass before this is used to rank anything:
  blur/speckle/JPEG/noise must all degrade the score monotonically.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def csf_weights(shape, cycles_per_degree_peak=4.0, degrees_per_pixel=1.0 / 60.0,
                reachable_cutoff=None, rolloff=8.0):
    """Radial contrast-sensitivity weight map with shape (H, W//2+1).

    Peak sensitivity near 4 cycles/degree. `degrees_per_pixel` converts pixel
    frequency to visual angle; default assumes roughly 60 px per degree (a
    typical desktop viewing distance), so override it for your display.
    """
    h, w = shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    r_px = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)      # cycles/pixel
    r_deg = r_px / max(degrees_per_pixel, 1e-9)              # cycles/degree
    x = r_deg / max(cycles_per_degree_peak, 1e-9)
    # Mannos-Sakrison style: rises to the peak then falls off.
    csf = (x ** 2) * np.exp(-x ** 2)
    csf = csf / max(csf.max(), 1e-12)
    if reachable_cutoff is not None:
        # Smoothly stop grading bands the LR input cannot support.
        csf = csf / (1.0 + np.exp((r_px - reachable_cutoff) * rolloff))
    return csf, r_px


def _fft(a):
    return np.fft.rfft2(a - a.mean())


def csf_l1(a, b, cutoff=None, degrees_per_pixel=1.0 / 60.0):
    """Frequency-weighted L1 between two float arrays (H, W) or (H, W, C).

    Usable directly as a training loss. Lower is better.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]; b = b[..., None]
    total, norm = 0.0, 0.0
    w, _ = csf_weights(a.shape[:2], degrees_per_pixel=degrees_per_pixel,
                       reachable_cutoff=cutoff)
    for c in range(a.shape[2]):
        d = np.abs(_fft(a[:, :, c]) - _fft(b[:, :, c]))
        total += float((d * w).sum()); norm += float(w.sum())
    return total / max(norm, 1e-12)


def csf_filter(pil_img, cutoff=None, degrees_per_pixel=1.0 / 60.0):
    """Apply the CSF weighting to an image, for CSF-weighted metric scoring."""
    a = np.asarray(pil_img.convert('RGB'), dtype=np.float32)
    out = np.zeros_like(a)
    w, _ = csf_weights(a.shape[:2], degrees_per_pixel=degrees_per_pixel,
                       reachable_cutoff=cutoff)
    for c in range(3):
        out[:, :, c] = np.fft.irfft2(_fft(a[:, :, c]) * w, a.shape[:2])
    lo, hi = out.min(), out.max()
    if hi - lo > 1e-6:
        out = (out - lo) / (hi - lo) * 255.0
    return __import__('PIL.Image', fromlist=['Image']).fromarray(
        np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- self-test
def self_test(images):
    """A metric that cannot separate clean from corrupted must not rank models."""
    from PIL import Image, ImageFilter
    import io

    def degrade(im, kind):
        a = np.asarray(im, dtype=np.float32)
        if kind == 'clean':
            return im
        if kind == 'blur':
            return im.filter(ImageFilter.GaussianBlur(2.0))
        if kind == 'speckle':
            rng = np.random.default_rng(0)
            return Image.fromarray(np.clip(a + rng.normal(0, 18, a.shape), 0, 255).astype(np.uint8))
        if kind == 'jpeg':
            buf = io.BytesIO(); im.save(buf, 'JPEG', quality=8)
            return Image.open(buf).convert('RGB')
        if kind == 'noise':
            rng = np.random.default_rng(1)
            return Image.fromarray(np.clip(a * 0.5 + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8))
        raise ValueError(kind)

    kinds = ['clean', 'blur', 'speckle', 'jpeg', 'noise']
    means = {k: [] for k in kinds}
    for im in images:
        ref = np.asarray(im.convert('L'), dtype=np.float32)
        for k in kinds:
            d = np.asarray(degrade(im, k).convert('L'), dtype=np.float32)
            means[k].append(csf_l1(d, ref))
    agg = {k: (sum(v) / len(v)) for k, v in means.items()}
    base = agg['clean']
    checks = {k: ('PASS' if agg[k] > base else f'FAIL({agg[k]:.4f} vs {base:.4f})')
              for k in kinds[1:]}
    for k in kinds:
        print(f'  {k:8s} csf_l1={agg[k]:.5f}')
    for k, v in checks.items():
        print(f'  {k:16s} {v}')
    return {'scores': agg, 'checks': checks, 'usable': all(v == 'PASS' for v in checks.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--images', nargs='*')
    ap.add_argument('--out', default='.build/quality-breakthrough-r85-csf/csf-self-test.json')
    a = ap.parse_args()
    if not a.self_test:
        print(__doc__); return
    if not a.images:
        raise SystemExit('--self-test needs --images')
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in a.images]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = self_test(ims)
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(f"\nUSABLE: {res['usable']}")


if __name__ == '__main__':
    main()
