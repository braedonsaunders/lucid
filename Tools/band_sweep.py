#!/usr/bin/env python3
"""Sweeps the band knobs and reports how far each result sits from the
reference's own fine/coarse balance. Lower is better."""
import glob, itertools, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/DerivedData/Build/Products/Debug/Lucid.app/Contents/MacOS/Lucid")
CLIPS = [("input-270p-300k", "reference-1080p"), ("input-360p-500k", "reference-1080p")]
BASE = json.load(open(os.path.join(ROOT, "Tools/tuning.json")))

def luma(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float64)

def bands(a):
    im = Image.fromarray(a.astype(np.uint8))
    b1 = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64)
    b25 = np.asarray(im.filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float64)
    return np.abs(a - b1).mean(), np.abs(b1 - b25).mean()

def correlation(a, b):
    """How much of the added fine detail lands where the truth has detail.
    Magnitude alone cannot tell real structure from amplified noise."""
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0

def measure(outdir):
    fine_err, coarse_err, rmse, corr = [], [], [], []
    for ref in sorted(glob.glob(f"{outdir}/*-reference.png")):
        got = ref.replace("-reference", "-detail")
        if not os.path.exists(got): return None
        R, A = luma(ref), luma(got)
        if A.shape != R.shape:
            A = np.asarray(Image.open(got).convert("L").resize(
                (R.shape[1], R.shape[0]), Image.LANCZOS), dtype=np.float64)
        rf, rc = bands(R); af, ac = bands(A)
        fine_err.append(abs(af - rf) / rf)
        coarse_err.append(abs(ac - rc) / rc)
        rmse.append(np.sqrt(((A - R) ** 2).mean()))
        hp = lambda x: x - np.asarray(Image.fromarray(x.astype(np.uint8))
                                      .filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64)
        corr.append(correlation(hp(A), hp(R)))
    if not fine_err: return None
    return (float(np.mean(fine_err)), float(np.mean(coarse_err)),
            float(np.mean(rmse)), float(np.mean(corr)))

def run(tag, overrides):
    tuning = dict(BASE, **overrides)
    # LUCID_TUNING names a file, it is not inline JSON.
    path = os.path.join(ROOT, f".build/bench/sweep-{tag}.json")
    with open(path, "w") as fh: json.dump(tuning, fh)
    fs, cs, rs, qs = [], [], [], []
    for i, (clip, ref) in enumerate(CLIPS):
        out = os.path.join(ROOT, f".build/bench/sweep-{tag}-{i}")
        shutil.rmtree(out, ignore_errors=True)
        subprocess.run([APP, "--bench", f".build/bench/{clip}.mp4",
                        f".build/bench/{ref}.mp4", out, "3", "3"],
                       cwd=ROOT, env=dict(os.environ, LUCID_TUNING=path),
                       capture_output=True, timeout=300)
        m = measure(out)
        shutil.rmtree(out, ignore_errors=True)
        if m is None: return None
        fs.append(m[0]); cs.append(m[1]); rs.append(m[2]); qs.append(m[3])
    return np.mean(fs), np.mean(cs), np.mean(rs), np.mean(qs)

grid = {"presharpen": [0.0, 0.25, 0.5, 0.8], "mid": [0.0, -0.6, -1.2, -1.8], "lobeScale": [0.5]}
keys = list(grid)
print(("{:>9}"*len(keys)).format(*keys) + f" | {'fine err':>9}{'coarse err':>11}{'RMSE':>8}{'corr':>8}{'score':>8}")
rows = []
for combo in itertools.product(*(grid[k] for k in keys)):
    ov = dict(zip(keys, combo))
    m = run("-".join(f"{v:g}" for v in combo), ov)
    if m is None:
        print(f"{combo} FAILED"); continue
    f, c, r, q = m
    score = 3.0 * f + 1.5 * c + 3.0 * (1.0 - q)
    rows.append((score, ov, f, c, r, q))
    print(("{:9.2f}"*len(combo)).format(*combo) + f" | {f*100:8.1f}%{c*100:10.1f}%{r:8.2f}{q:8.3f}{score:8.3f}")
rows.sort(key=lambda r: r[0])
print("\nbest:", json.dumps(rows[0][1]), f"score {rows[0][0]:.3f}")
