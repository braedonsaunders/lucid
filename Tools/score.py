#!/usr/bin/env python3
"""Scores each engine's frames against the ground-truth reference frames."""
import sys, glob, os
import numpy as np
from PIL import Image

def load(p): return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)
def gray(a): return np.asarray(Image.fromarray(a.astype(np.uint8)).convert('L'), dtype=np.float64)
def psnr(a,b):
    m = np.mean((a-b)**2); return 99.0 if m == 0 else 10*np.log10(255*255/m)
def ssim(a,b):
    a, b = gray(a), gray(b)
    mu_a, mu_b = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a-mu_a)*(b-mu_b)).mean()
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return ((2*mu_a*mu_b+c1)*(2*cov+c2))/((mu_a**2+mu_b**2+c1)*(va+vb+c2))
def detail(a):
    g = gray(a); return np.mean(np.abs(np.diff(g,axis=0))) + np.mean(np.abs(np.diff(g,axis=1)))
def saturation(a):
    mx, mn = a.max(axis=2), a.min(axis=2); return float(np.mean((mx-mn)/np.maximum(mx,1e-6)))

d = sys.argv[1]
variants = ['input','lowlatency','detail','temporal','efrlfn','neural']
acc = {}
refs = sorted(glob.glob(f'{d}/*-reference.png'))
for ref in refs:
    n = os.path.basename(ref).split('-')[0]
    R = load(ref)
    for v in variants:
        path = f'{d}/{n}-{v}.png'
        if not os.path.exists(path): continue
        A = load(path)
        if v == 'input':
            A = np.asarray(Image.open(path).convert('RGB').resize((R.shape[1],R.shape[0]), Image.BICUBIC), dtype=np.float64)
            label = 'browser scaling (bicubic)'
        else:
            label = {'lowlatency':'Apple scaler 4x','detail':'Apple scaler + Lucid detail','temporal':'Apple temporal 4x','efrlfn':'EfRLFN 4x (Neural Engine)','neural':'EfRLFN + Lucid detail'}[v]
        acc.setdefault(label, []).append((psnr(A,R), ssim(A,R), detail(A), saturation(A)))
    acc.setdefault('reference (truth)', []).append((99.0, 1.0, detail(R), saturation(R)))

print(f"{'engine':32}{'PSNR':>8}{'SSIM':>8}{'detail':>9}{'sat':>8}")
for label in ['browser scaling (bicubic)','Apple scaler 4x','Apple scaler + Lucid detail','Apple temporal 4x','EfRLFN 4x (Neural Engine)','EfRLFN + Lucid detail','reference (truth)']:
    if label not in acc: continue
    v = np.mean(np.array(acc[label]), axis=0)
    print(f"{label:32}{v[0]:8.2f}{v[1]:8.4f}{v[2]:9.3f}{v[3]:8.4f}")
