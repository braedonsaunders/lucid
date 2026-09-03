#!/usr/bin/env python3
"""Compares fine and mid frequency-band energy against the reference.

Total gradient rewards any high-frequency energy, including the coarse
structure that makes an upscale look blobby. Splitting the bands shows whether
an engine matches the reference's balance or just adds energy at the wrong
scale.
"""
import sys, glob, os
import numpy as np
from PIL import Image, ImageFilter

def gray(p):
    return np.asarray(Image.open(p).convert('L'), dtype=np.float64)

def blur(a, r):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)

def bands(a):
    b1, b2 = blur(a, 1.0), blur(a, 2.5)
    return float(np.mean(np.abs(a - b1))), float(np.mean(np.abs(b1 - b2)))

targets = sys.argv[1:]
print(f"{'variant':34}{'fine':>8}{'mid':>8}{'fine/ref':>10}{'mid/ref':>9}{'balance':>9}")
rows = []
for d in targets:
    refs = sorted(glob.glob(f'{d}/*-reference.png'))
    if not refs: continue
    n = os.path.basename(refs[0]).split('-')[0]
    rf, rm = bands(gray(refs[0]))
    for variant, label in (('input','browser scaling'), ('lowlatency','Apple scaler'), ('detail', os.path.basename(d)), ('efrlfn','EfRLFN'), ('neural','EfRLFN + detail'), ('temporal','Apple temporal')):
        path = f'{d}/{n}-{variant}.png'
        if not os.path.exists(path): continue
        if variant == 'input':
            im = Image.open(path).convert('L').resize(Image.open(refs[0]).size, Image.BICUBIC)
            a = np.asarray(im, dtype=np.float64)
        else:
            a = gray(path)
        f, m = bands(a)
        key = label
        if key in [r[0] for r in rows] and variant not in ('detail','efrlfn','neural'): continue
        rows.append((key, f, m, f/rf, m/rm, (f/rf)/(m/rm)))
    rows.append(('reference (truth)', rf, rm, 1.0, 1.0, 1.0))
seen = set()
for k, f, m, fr, mr, bal in rows:
    if k in seen and k != 'reference (truth)': continue
    if k == 'reference (truth)' and k in seen: continue
    seen.add(k)
    print(f"{k:34}{f:8.3f}{m:8.3f}{fr:10.3f}{mr:9.3f}{bal:9.3f}")
