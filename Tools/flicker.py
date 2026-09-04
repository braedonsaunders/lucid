#!/usr/bin/env python3
"""Measures how much the picture shimmers between frames.

Every other metric in this project scores one frame against one reference, and
is therefore structurally blind to the axis this measures. That blindness has
already cost us once: the per-frame table said the temporal stage was harmful on
every clip, because a stage that trades a little per-frame sharpness for
frame-to-frame steadiness can only ever look like a loss to an instrument that
sees one frame at a time.

It matters more now. A GAN synthesises plausible texture per frame
independently - nothing tells frame N what frame N-1 invented - so adversarial
training buys detail and pays for it in stability. That is the shimmer a viewer
notices, and until this tool existed nobody could say whether it was there or
how much.

  flicker = mean |output[t] - output[t-1]| over pixels the REFERENCE holds still

The mask is the load-bearing part. Taking it from the reference rather than from
our own output means it cannot be influenced by anything we did: it isolates
regions that genuinely should not be changing, so movement there is our
invention rather than the scene's. Measuring our output against itself would
simply reward a model that changes less, which a blur does perfectly.

  .venv-convert/bin/python Tools/flicker.py --frames 12
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/release/Build/Products/Release/Lucid.app/Contents/MacOS/Lucid")

# Pixels the reference moves by less than this between frames are treated as
# static. In 0-255 luma: below roughly one level is sensor and codec noise in the
# reference itself, not motion.
STATIC = 1.2


def render(stem, clip, reference, frames, tuning):
    """Runs the bench and returns its output and reference frames, in order."""
    out = tempfile.mkdtemp(prefix="flicker-")
    path = os.path.join(out, "tuning.json")
    with open(path, "w") as fh:
        json.dump(tuning, fh)
    env = dict(os.environ, LUCID_TUNING=path, LUCID_LEARNED="1")
    if stem:
        env["LUCID_MODEL_STEM"] = stem
    p = subprocess.run([APP, "--bench", clip, reference, out, "0", str(frames)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-500:], p.stderr[-500:])
        shutil.rmtree(out, ignore_errors=True)
        return None, None
    outputs, refs = [], []
    for name in sorted(f for f in os.listdir(out) if f.endswith("-detail.png")):
        stem_id = name.split("-")[0]
        ref = os.path.join(out, f"{stem_id}-reference.png")
        if not os.path.exists(ref):
            continue
        a = Image.open(os.path.join(out, name)).convert("L")
        r = Image.open(ref).convert("L")
        if a.size != r.size:
            a = a.resize(r.size, Image.LANCZOS)
        outputs.append(np.asarray(a, dtype=np.float64))
        refs.append(np.asarray(r, dtype=np.float64))
    shutil.rmtree(out, ignore_errors=True)
    return outputs, refs


def flicker(outputs, refs):
    """Mean frame-to-frame change in the output, over pixels the reference holds
    still. Also returns how much of the frame that mask covers, because a number
    computed over 2% of the picture is not comparable to one over 60%."""
    values, coverage = [], []
    for i in range(1, len(outputs)):
        moved = np.abs(refs[i] - refs[i - 1])
        static = moved < STATIC
        if static.sum() < 1000:
            continue
        values.append(float(np.abs(outputs[i] - outputs[i - 1])[static].mean()))
        coverage.append(float(static.mean()))
    if not values:
        return None, None
    return float(np.mean(values)), float(np.mean(coverage))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="crowdrun-360p-350k.mp4")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--tuning", default=os.path.join(ROOT, "Tools/tuning.json"))
    parser.add_argument("--stems", nargs="*",
                        default=["SPAN_x4_ch32ul1_", "SPAN_x4_ch32u_"],
                        help="model stems to compare, in order")
    args = parser.parse_args()

    known = {"crowdrun-360p-350k.mp4": "crowdrun-1080p.mp4",
             "dinner-360p-350k.mp4": "dinner-1080p.mp4"}
    clip = os.path.join(ROOT, "TestSite", args.clip)
    reference = os.path.join(ROOT, "TestSite", known.get(args.clip, ""))
    for p in (clip, reference):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    with open(args.tuning) as fh:
        tuning = json.load(fh)

    print(f"{args.clip}, {args.frames} frames")
    print("flicker is mean frame-to-frame change over pixels the reference holds")
    print("still, in 0-255 luma. Lower is steadier.\n")
    print(f"{'model':28s} {'flicker':>9s} {'static area':>12s}")
    results = {}
    for stem in args.stems:
        outputs, refs = render(stem, clip, reference, args.frames, tuning)
        if not outputs:
            print(f"{stem:28s}  render failed"); continue
        value, coverage = flicker(outputs, refs)
        if value is None:
            print(f"{stem:28s}  no static regions - wrong clip for this test"); continue
        results[stem] = value
        print(f"{stem:28s} {value:9.4f} {coverage*100:11.1f}%")

    if len(results) == 2:
        a, b = list(results.values())
        names = list(results.keys())
        change = (b - a) / a * 100
        print(f"\n{names[1]} shimmers {abs(change):.1f}% "
              f"{'MORE' if change > 0 else 'less'} than {names[0]}")


if __name__ == "__main__":
    main()
