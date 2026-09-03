#!/usr/bin/env python3
"""Three-band quality matrix across real clips at different resolutions.

Baseline: the CURRENT build, which already contains gain normalisation
(2 - log2(r)/2, radius-scaled terms only) and the fixed pixel-spacing
sharpening lobe. There is no pre-fix "before" here; do not compare these
numbers against the synthetic radius-sweep spreads.

Each clip runs at the radius the live pipeline would actually use for it
(nearest power of two of reference/input width, i.e. 2**stageCount), and the
report separates the fine/mid/coarse bands so fine-band variation across
scale is visible instead of being averaged away.
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
BASE = json.load(open(os.path.join(ROOT, "Tools/tuning.json")))
FIRST, COUNT = 3, 3

try:
    HEAD = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
except OSError:
    HEAD = "unknown"


def luma(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64)


def bands(a):
    im = Image.fromarray(a.astype(np.uint8))
    b = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)
         for r in (1.0, 2.5, 6.0)]
    return np.array([np.abs(a - b[0]).mean(),   # fine
                     np.abs(b[0] - b[1]).mean(),  # mid
                     np.abs(b[1] - b[2]).mean()])  # coarse


def radius_for(stretch):
    """Nearest power of two of the stretch, clamped to the 1..3 stages the
    live stageCount can produce (radius 2..8)."""
    return 2 ** min(max(int(round(math.log2(max(stretch, 1)))), 1), 3)


def ref_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def run(clip, radius):
    slug = os.path.basename(clip).replace(".mp4", "")
    tuning = dict(BASE)
    path = os.path.join(ROOT, f".build/bench/matrix-{slug}.json")
    with open(path, "w") as fh:
        json.dump(tuning, fh)
    out = os.path.join(ROOT, f".build/bench/matrix-{slug}")
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([APP, "--bench", clip, REF, out, str(FIRST), str(COUNT)],
                   cwd=ROOT, env=dict(os.environ, LUCID_TUNING=path,
                                      LUCID_RADIUS=str(radius)),
                   capture_output=True, timeout=600)
    os.remove(path)
    return out


def measure(outdir):
    """Per-variant mean band energies, each resized to the reference."""
    refs = sorted(glob.glob(f"{outdir}/*-reference.png"))
    if not refs:
        return None
    acc = {}
    for ref in refs:
        R = luma(ref)
        acc.setdefault("reference", []).append(bands(R))
        for variant in ("input", "lowlatency", "detail"):
            p = ref.replace("-reference", f"-{variant}")
            if not os.path.exists(p):
                continue
            A = luma(p)
            if A.shape != R.shape:
                A = np.asarray(Image.open(p).convert("L").resize(
                    (R.shape[1], R.shape[0]), Image.LANCZOS), dtype=np.float64)
            acc.setdefault(variant, []).append(bands(A))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


lines = []
lines.append("Lucid three-band matrix")
lines.append(f"build: git {HEAD} + uncommitted Item A (stageCount; bench never calls")
lines.append("  stageCount, so the numbers below are unaffected by it)")
lines.append("  includes gain normalisation + fixed lobe spacing from the current HEAD")
lines.append("tuning: stock Tools/tuning.json; no pre-fix baseline exists for these numbers")
lines.append("caveat: the bench renders a fixed 4x low-latency chain for every clip,")
lines.append("  while the live pipeline would render the 540p clip at 2x. Radii below")
lines.append("  are the LUCID_RADIUS the detail stage actually ran at.")
lines.append("bands: fine |a-blur1|  mid |blur1-blur2.5|  coarse |blur2.5-blur6|")
lines.append("radius per clip: nearest power of two of ref/input width (live stageCount)")
lines.append("")

detail_rows = []
ok = True
for clip, label in CLIPS:
    full = os.path.join(ROOT, clip)
    if not os.path.exists(full):
        lines.append(f"{label}: MISSING {clip}")
        ok = False
        continue
    iw, _ = ref_size(full)
    rw, _ = ref_size(os.path.join(ROOT, REF))
    radius = radius_for(rw / iw)
    out = run(clip, radius)
    m = measure(out)
    if m is None or "detail" not in m:
        lines.append(f"{label} (radius {radius}): BENCH FAILED, see {out}")
        ok = False
        continue
    ref = m["reference"]
    lines.append(f"{label}  stretch {rw/iw:.2f}x  radius {radius}  dir {out}")
    lines.append(f"  {'variant':12}{'fine':>9}{'mid':>9}{'coarse':>9}"
                 f"{'fine/ref':>10}{'mid/ref':>9}{'coarse/ref':>11}")
    for variant in ("reference", "input", "lowlatency", "detail"):
        if variant not in m:
            continue
        v = m[variant]
        lines.append(f"  {variant:12}{v[0]:9.3f}{v[1]:9.3f}{v[2]:9.3f}"
                     f"{v[0]/ref[0]:10.1%}{v[1]/ref[1]:9.1%}{v[2]/ref[2]:11.1%}")
    lines.append("")
    detail_rows.append((label, m["detail"] / ref))

if detail_rows:
    lines.append("detail band balance vs reference across clips (spread = max-min):")
    lines.append(f"  {'clip':14}{'fine':>8}{'mid':>8}{'coarse':>8}")
    for label, r in detail_rows:
        lines.append(f"  {label:14}{r[0]:8.1%}{r[1]:8.1%}{r[2]:8.1%}")
    arr = np.array([r for _, r in detail_rows])
    spread = arr.max(axis=0) - arr.min(axis=0)
    lines.append(f"  {'spread':14}{spread[0]:8.1%}{spread[1]:8.1%}{spread[2]:8.1%}")
    lines.append("  (fine variation with scale is expected: bigger upscales carry")
    lines.append("   less real detail per output pixel. Mid/coarse should sit still.)")

report = "\n".join(lines)
print(report)
with open(os.path.join(ROOT, ".build/bench/matrix-report.txt"), "w") as fh:
    fh.write(report + "\n")
sys.exit(0 if ok else 1)
