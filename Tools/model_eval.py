#!/usr/bin/env python3
"""Fair fight: is a candidate engine actually better than Apple's scaler?

Reads one bench frame directory (NNN-input / -cleaned / -lowlatency /
-detail / -reference PNGs at matched frame indices) and scores every
engine present, plus bicubic/Lanczos upscales of the input as anchors,
plus any .mlpackage passed with --model (run via coremltools on
CPU_AND_NE).

Per engine: PSNR and per-band CORRELATION with the reference at 1px /
2px / 5px blur radii. Correlation is the metric that separates
recovered detail from invented detail: band energy alone is
scale-invariant and cannot tell a near-black conversion failure from a
good upscale, so every model output is level-checked (mean ratio vs
the source must sit inside 0.8-1.2) and refused otherwise.

No composite score. Bands are shown separately and the verdict is a
per-band win count against the Lanczos anchor.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

RADII = (1.0, 2.0, 5.0)
LEVEL_LO, LEVEL_HI = 0.8, 1.2


def luma(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def to_ref_size(arr_img, ref_shape):
    """Resize a L-mode array to the reference (h, w) with Lanczos."""
    if arr_img.shape == ref_shape:
        return arr_img
    im = Image.fromarray(np.clip(arr_img, 0, 255).astype(np.uint8))
    return np.asarray(
        im.resize((ref_shape[1], ref_shape[0]), Image.LANCZOS),
        dtype=np.float64,
    )


def highpass(a, radius):
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    b = np.asarray(im.filter(ImageFilter.GaussianBlur(radius)),
                   dtype=np.float64)
    return a - b


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0


def psnr(a, r):
    mse = ((a - r) ** 2).mean()
    return float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else float("inf")


def run_model(path, inputs):
    import coremltools as ct

    mlmodel = ct.models.MLModel(path,
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
    # Image-in model: predict takes a PIL image keyed by input name.
    spec = mlmodel.get_spec()
    in_name = spec.description.input[0].name
    outs = []
    for p in inputs:
        pred = mlmodel.predict({in_name: Image.open(p).convert("RGB")})
        out = pred[spec.description.output[0].name]
        if not isinstance(out, Image.Image):
            out = Image.fromarray(np.asarray(out).astype(np.uint8))
        outs.append(np.asarray(out.convert("L"), dtype=np.float64))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("framedir", help="bench output dir with NNN-*.png frames")
    ap.add_argument("--model", action="append", default=[],
                    help=".mlpackage to score; repeatable")
    args = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(args.framedir, "*-reference.png")))
    if not refs:
        sys.exit(f"no *-reference.png in {args.framedir}")
    inputs = [r.replace("-reference", "-input") for r in refs]
    if not all(os.path.exists(p) for p in inputs):
        sys.exit("matched *-input.png missing for some frames")

    engines = {}  # name -> list of L arrays (native size)
    have = lambda suf: all(
        os.path.exists(r.replace("-reference", suf)) for r in refs)
    inp = Image.open(inputs[0]).convert("RGB")
    anchors = {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}
    ref0 = np.asarray(Image.open(refs[0]).convert("L"), dtype=np.float64)
    for name, filt in anchors.items():
        engines[f"anchor-{name}"] = [
            np.asarray(Image.open(p).convert("RGB")
                       .resize((ref0.shape[1], ref0.shape[0]), filt)
                       .convert("L"), dtype=np.float64)
            for p in inputs
        ]
    for suf in ("-cleaned", "-lowlatency", "-detail"):
        if have(suf):
            engines["pipeline" + suf] = [
                luma(r.replace("-reference", suf)) for r in refs]
    refused = {}
    for mpath in args.model:
        name = os.path.splitext(os.path.basename(mpath))[0]
        try:
            outs = run_model(mpath, inputs)
        except Exception as e:  # noqa: BLE001 - report and continue
            refused[name] = f"run failed: {e}"
            continue
        ratio = float(np.mean([o.mean() for o in outs]) /
                      np.mean([np.asarray(Image.open(p).convert("L"),
                                          dtype=np.float64).mean()
                               for p in inputs]))
        if not (LEVEL_LO <= ratio <= LEVEL_HI):
            refused[name] = (f"level check failed: mean ratio {ratio:.3f} "
                             f"outside {LEVEL_LO}-{LEVEL_HI}; not scored")
            continue
        engines["model-" + name] = outs

    # Score everything at reference size.
    rows = {}
    for name, arrs in engines.items():
        ps, cs = [], [[] for _ in RADII]
        for arr, r in zip(arrs, refs):
            R = luma(r)
            A = to_ref_size(arr, R.shape)
            ps.append(psnr(A, R))
            for i, rad in enumerate(RADII):
                cs[i].append(correlation(highpass(A, rad),
                                         highpass(R, rad)))
        rows[name] = (float(np.mean(ps)),
                      [float(np.mean(c)) for c in cs])

    lanc = rows.get("anchor-lanczos")
    print(f"frames: {len(refs)} from {args.framedir}")
    hdr = f"{'engine':36}{'PSNR':>7}" + "".join(
        f"{f'corr@{r:g}px':>12}" for r in RADII) + "  verdict"
    print(hdr)
    for name, (p, cl) in sorted(rows.items()):
        if lanc is not None and name != "anchor-lanczos":
            wins = sum(c > l for c, l in zip(cl, lanc[1]))
            verdict = (f"beats anchor ({wins}/3 bands)"
                       if wins >= 2 else "does not beat anchor")
        else:
            verdict = "anchor" if name == "anchor-lanczos" else ""
        print(f"{name:36}{p:7.2f}" + "".join(f"{c:12.3f}"
              for c in cl) + f"  {verdict}")
    for name, why in refused.items():
        print(f"model-{name}: REFUSED - {why}")


if __name__ == "__main__":
    main()
