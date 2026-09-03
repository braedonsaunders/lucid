#!/usr/bin/env python3
"""Finds the gain scaling that makes a setting mean the same thing at any
upscale factor. Holds the settings fixed, varies the kernel radius, and for each
radius searches the multiplier on the sharpening gains that reproduces the band
energy radius 4 produces. The result is the normalisation curve."""
import glob, json, os, shutil, subprocess
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/DerivedData/Build/Products/Debug/Lucid.app/Contents/MacOS/Lucid")
CLIP, REF = ".build/bench/input-270p-300k.mp4", ".build/bench/reference-1080p.mp4"
BASE = json.load(open(os.path.join(ROOT, "Tools/tuning.json")))

def luma(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)

def bands(a):
    im = Image.fromarray(a.astype(np.uint8))
    b = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64) for r in (1.0, 2.5, 6.0)]
    return np.array([np.abs(a - b[0]).mean(), np.abs(b[0] - b[1]).mean(), np.abs(b[1] - b[2]).mean()])

def run(radius, multiplier):
    tuning = dict(BASE)
    tuning["sharpness"] = BASE["sharpness"] * multiplier
    tuning["fine"] = BASE["fine"] * multiplier
    path = os.path.join(ROOT, f".build/bench/fit-{radius}-{multiplier:.3f}.json")
    with open(path, "w") as fh: json.dump(tuning, fh)
    out = os.path.join(ROOT, f".build/bench/fit-{radius}-{multiplier:.3f}")
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([APP, "--bench", CLIP, REF, out, "3", "3"], cwd=ROOT,
                   env=dict(os.environ, LUCID_TUNING=path, LUCID_RADIUS=str(radius)),
                   capture_output=True, timeout=300)
    files = sorted(glob.glob(f"{out}/*-detail.png"))
    value = np.mean([bands(luma(f)) for f in files], axis=0) if files else None
    shutil.rmtree(out, ignore_errors=True); os.remove(path)
    return value

target = run(4, 1.0)
print(f"target (radius 4, gain x1.00): fine {target[0]:.3f}  mid {target[1]:.3f}  coarse {target[2]:.3f}\n")
print(f"{'radius':>7}{'gain':>7}{'fine':>8}{'mid':>8}{'coarse':>8}{'distance':>10}")
best = {}
for radius in (2, 4, 8):
    scored = []
    for m in (0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
        v = run(radius, m)
        if v is None: continue
        # Match the mid and coarse bands: those are the ones that ran away.
        d = abs(v[1] - target[1]) / target[1] + abs(v[2] - target[2]) / target[2]
        scored.append((d, m, v))
        print(f"{radius:>7}{m:>7.2f}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{d:10.4f}")
    if scored:
        d, m, v = min(scored)
        best[radius] = m
        print(f"        -> best gain for radius {radius}: x{m:.2f}\n")
print("normalisation:", {r: round(m, 3) for r, m in best.items()})
