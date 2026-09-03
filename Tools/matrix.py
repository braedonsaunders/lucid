#!/usr/bin/env python3
"""Scale-invariance matrix: three-band spread across detail radii, per clip.

For each real clip, holds the settings fixed, sweeps the detail radius over
2/4/8 via LUCID_RADIUS, and reports the per-band spread across radii. Spread
is (max-min)/mean of the band energy, so it is directly comparable to the
single-clip baseline sweep: fine 50.8%, mid 14.9%, coarse 2.9%.

Baseline: the CURRENT build, which already contains gain normalisation
(2 - log2(r)/2, radius-scaled terms only) and the fixed pixel-spacing
sharpening lobe. There is no pre-fix "before" here.

Tuning comes from stock Tools/tuning.json via LUCID_TUNING (a file path, not
inline JSON), so the matrix always measures the checked-in defaults.
"""
import glob, json, math, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/DerivedData/Build/Products/Debug/Lucid.app/Contents/MacOS/Lucid")
REF = ".build/bench/reference-1080p.mp4"
CLIPS = [
    (".build/bench/input-270p-300k.mp4", "270p 300k"),
    (".build/bench/input-270p-900k.mp4", "270p 900k"),
    (".build/bench/input-360p-500k.mp4", "360p 500k"),
    (".build/bench/input-540p-1500k.mp4", "540p 1.5M"),
]
RADII = (2, 4, 8)
BASELINE_SWEEP = (50.8, 14.9, 2.9)  # prior single-clip sweep, fine/mid/coarse %
BASE = json.load(open(os.path.join(ROOT, "Tools/tuning.json")))
FIRST, COUNT = 3, 3


def git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


HEAD = git("rev-parse", "--short", "HEAD")
DIRTY = bool(git("status", "--short"))


def luma(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def bands(a):
    im = Image.fromarray(a.astype(np.uint8))
    b = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)
         for r in (1.0, 2.5, 6.0)]
    return np.array([np.abs(a - b[0]).mean(),   # fine
                     np.abs(b[0] - b[1]).mean(),  # mid
                     np.abs(b[1] - b[2]).mean()])  # coarse


def ref_width(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0])


def run(clip, radius):
    slug = os.path.basename(clip).replace(".mp4", "")
    tuning = dict(BASE)
    path = os.path.join(ROOT, f".build/bench/matrix-{slug}-r{radius}.json")
    with open(path, "w") as fh:
        json.dump(tuning, fh)
    out = os.path.join(ROOT, f".build/bench/matrix-{slug}-r{radius}")
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([APP, "--bench", clip, REF, out, str(FIRST), str(COUNT)],
                   cwd=ROOT, env=dict(os.environ, LUCID_TUNING=path,
                                      LUCID_RADIUS=str(radius)),
                   capture_output=True, timeout=600)
    os.remove(path)
    return out


def measure_detail(outdir):
    """Mean detail band energies, resized to the reference when needed."""
    refs = sorted(glob.glob(f"{outdir}/*-reference.png"))
    if not refs:
        return None, None
    det, ref = [], []
    for r in refs:
        R = luma(r)
        p = r.replace("-reference", "-detail")
        if not os.path.exists(p):
            return None, None
        A = luma(p)
        if A.shape != R.shape:
            A = np.asarray(Image.open(p).convert("L").resize(
                (R.shape[1], R.shape[0]), Image.LANCZOS), dtype=np.float64)
        det.append(bands(A))
        ref.append(bands(R))
    return np.mean(det, axis=0), np.mean(ref, axis=0)


def spread(rows):
    arr = np.array(rows)
    return (arr.max(axis=0) - arr.min(axis=0)) / arr.mean(axis=0) * 100.0


lines = []
lines.append("Lucid scale-invariance matrix")
lines.append(f"build: git {HEAD}{' (dirty tree)' if DIRTY else ''}; "
             "gain normalisation + fixed lobe spacing already in, no pre-fix baseline")
lines.append("tuning: stock Tools/tuning.json via LUCID_TUNING = "
             f"sharpness {BASE['sharpness']} fine {BASE['fine']} mid {BASE['mid']} "
             f"lobeScale {BASE['lobeScale']} micro {BASE['micro']} "
             f"adaptive {BASE['adaptive']} sourceDeblock {BASE['sourceDeblock']}")
lines.append("bands: fine |a-blur1|  mid |blur1-blur2.5|  coarse |blur2.5-blur6|; "
             "detail energies as % of reference")
lines.append("headline: spread = (max-min)/mean across radii 2/4/8, same settings")
lines.append(f"prior single-clip baseline for comparison: fine {BASELINE_SWEEP[0]:.1f}%  "
             f"mid {BASELINE_SWEEP[1]:.1f}%  coarse {BASELINE_SWEEP[2]:.1f}%")
lines.append("caveat: the bench renders a fixed 4x low-latency chain, so LUCID_RADIUS")
lines.append("  varies only the detail kernel scale (same synthetic setup as the radius")
lines.append("  sweep this is compared against, now repeated per real clip).")
lines.append("")

all_spreads = []
ok = True
for clip, label in CLIPS:
    full = os.path.join(ROOT, clip)
    if not os.path.exists(full):
        lines.append(f"{label}: MISSING {clip}")
        ok = False
        continue
    stretch = ref_width(os.path.join(ROOT, REF)) / ref_width(full)
    rows, ratios = [], []
    failed = False
    for radius in RADII:
        out = run(clip, radius)
        det, ref = measure_detail(out)
        if det is None:
            lines.append(f"{label} radius {radius}: BENCH FAILED, see {out}")
            ok, failed = False, True
            break
        rows.append(det)
        ratios.append(det / ref)
    if failed:
        continue
    s = spread(rows)
    all_spreads.append(s)
    lines.append(f"{label}  stretch {stretch:.2f}x")
    lines.append(f"  {'radius':8}{'fine':>9}{'mid':>9}{'coarse':>9}")
    for radius, r in zip(RADII, ratios):
        lines.append(f"  {radius:<8}{r[0]:9.1%}{r[1]:9.1%}{r[2]:9.1%}")
    lines.append(f"  {'spread':8}{s[0]:9.1f}%{s[1]:9.1f}%{s[2]:9.1f}%")
    lines.append("")

if all_spreads:
    mean = np.mean(all_spreads, axis=0)
    lines.append("mean spread across clips (the headline number):")
    lines.append(f"  fine {mean[0]:.1f}%  mid {mean[1]:.1f}%  coarse {mean[2]:.1f}%  "
                 f"total {mean.sum():.1f}%")
    lines.append(f"prior baseline: fine {BASELINE_SWEEP[0]:.1f}%  mid {BASELINE_SWEEP[1]:.1f}%  "
                 f"coarse {BASELINE_SWEEP[2]:.1f}%  total {sum(BASELINE_SWEEP):.1f}%")
    lines.append("  (fine variation with scale is expected: bigger upscales carry less")
    lines.append("   real detail per output pixel. Mid/coarse should sit still.)")

report = "\n".join(lines)
print(report)
with open(os.path.join(ROOT, ".build/bench/matrix-report.txt"), "w") as fh:
    fh.write(report + "\n")
sys.exit(0 if ok else 1)
