#!/usr/bin/env python3
"""Autonomous tuner for the Apple-scaler pipeline.

Runs the offline bench over a parameter set, scores the result against the
untouched source frames, and keeps whatever wins. The objective is to match
what the picture looked like before compression: its detail at the scale detail
actually lives, its black level, its contrast and its colour. Those are the four
things a compressed stream loses and the four things the grade puts back, so
matching them is a stand-in for "looks right" that a machine can measure.

  python3 Tools/tune.py --rounds 2            # coordinate descent from current
  python3 Tools/tune.py --score-only          # just score the current settings
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/DerivedData/Build/Products/Debug/Lucid.app/Contents/MacOS/Lucid")
CLIPS = [
    (".build/bench/input-270p-300k.mp4", ".build/bench/reference-1080p.mp4"),
    (".build/bench/input-360p-500k.mp4", ".build/bench/reference-1080p.mp4"),
]
TUNING = os.path.join(ROOT, "Tools/tuning.json")

# name -> (low, high, step) for the coordinate search
KNOBS = {
    "sharpness":     (0.30, 1.60, 0.20),
    "fine":          (0.20, 2.00, 0.25),
    "deblock":       (0.00, 0.45, 0.10),
    "sourceDeblock": (0.00, 0.060, 0.012),
    "blackPoint":    (0.000, 0.070, 0.012),
    "whitePoint":    (0.930, 1.000, 0.015),
    "contrast":      (0.00, 0.45, 0.08),
    "saturation":    (1.00, 1.35, 0.06),
}
DEFAULTS = {"sharpness":0.85,"fine":0.90,"deblock":0.15,"sourceDeblock":0.030,
            "temporal":0.50,"blackPoint":0.020,"whitePoint":0.990,"contrast":0.180,"saturation":1.100}

def gray(a):    return np.asarray(Image.fromarray(a).convert("L"), dtype=np.float64)
def blur(a, r): return np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)

def features(path):
    """Detail at two scales, black level, contrast and colour saturation."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    g = gray(np.asarray(Image.open(path).convert("RGB")))
    b1, b2 = blur(g, 1.0), blur(g, 2.5)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    return {
        "fine":   float(np.mean(np.abs(g - b1))),
        "coarse": float(np.mean(np.abs(b1 - b2))),
        "black":  float(np.percentile(g, 1)),
        "contrast": float(g.std()),
        "sat":    float(np.mean((mx - mn) / np.maximum(mx, 1e-6))),
    }

WEIGHTS = {"fine": 3.0, "coarse": 1.0, "black": 1.5, "contrast": 1.5, "sat": 1.5}

def score(outdir):
    """Distance from the reference, lower is better. Relative error per feature
    so each counts comparably regardless of its units."""
    total, frames = 0.0, 0
    for ref in sorted(f for f in os.listdir(outdir) if f.endswith("-reference.png")):
        n = ref.split("-")[0]
        got = os.path.join(outdir, f"{n}-detail.png")
        if not os.path.exists(got): return 1e9   # a missing frame is a failure, not a skip
        r, d = features(os.path.join(outdir, ref)), features(got)
        for k, w in WEIGHTS.items():
            denom = max(abs(r[k]), 1e-6)
            total += w * abs(d[k] - r[k]) / denom
        frames += 1
    return total / max(frames, 1) if frames else 1e9

_run_counter = [0]

def run(params, tag):
    _run_counter[0] += 1
    tag = f"{tag}{_run_counter[0]}"          # unique folder per trial
    with open(TUNING, "w") as f: json.dump(params, f)
    outs = []
    for i, (clip, ref) in enumerate(CLIPS):
        if not os.path.exists(os.path.join(ROOT, clip)): continue
        out = os.path.join(ROOT, f".build/bench/tune-{tag}-{i}")
        shutil.rmtree(out, ignore_errors=True)
        env = dict(os.environ, LUCID_TUNING=TUNING)
        subprocess.run([APP, "--bench", clip, ref, out, "3", "3"], cwd=ROOT, env=env,
                       capture_output=True, timeout=400)
        outs.append(out)
    return float(np.mean([score(o) for o in outs])) if outs else 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    best = dict(DEFAULTS)
    if os.path.exists(TUNING):
        try: best.update(json.load(open(TUNING)))
        except Exception: pass

    base = run(best, "base")
    print(f"start score {base:.4f}  {json.dumps({k: round(v,3) for k,v in best.items()})}")
    if a.score_only: return

    for rnd in range(a.rounds):
        improved = False
        for knob, (lo, hi, step) in KNOBS.items():
            current = best[knob]
            for candidate in (current - step, current + step):
                if not (lo <= candidate <= hi): continue
                trial = dict(best); trial[knob] = round(candidate, 4)
                s = run(trial, "t")
                mark = ""
                if s < base - 1e-4:
                    base, best, improved, mark = s, trial, True, "  <- kept"
                print(f"  round {rnd+1} {knob:14s} {candidate:6.3f} -> {s:.4f}{mark}")
        print(f"round {rnd+1} best {base:.4f}  {json.dumps({k: round(v,3) for k,v in best.items()})}")
        if not improved:
            print("no further improvement"); break

    with open(TUNING, "w") as f: json.dump(best, f, indent=2)
    print(f"\nwrote {TUNING}")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
