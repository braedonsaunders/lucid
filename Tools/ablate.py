#!/usr/bin/env python3
"""Measures what each enhancement stage is actually worth.

Lucid has six independent stages after the upscaler and three of them ship
switched off. None was ever measured on its own against ground truth - they
were settled by eye, and by eye is how this project previously convinced itself
that a pipeline scoring *below* a plain Lanczos upscale looked good.

So: hold everything else fixed, toggle one stage at a time, and score the
result against a 1080p reference. For each stage this reports what it buys and
what it costs, which are the two numbers needed to decide whether it earns its
place in a 33 ms frame.

Correlation per spatial band is the metric, not band energy. Energy cannot tell
detail that was recovered from detail that was invented, and it has misled this
project twice.

  .venv-convert/bin/python Tools/ablate.py --clip sintel-scene-360p-300k.mp4
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, ".build/release/Build/Products/Release/Lucid.app/Contents/MacOS/Lucid")

# Every stage, with the value that turns it on and the default it ships with.
# `None` for a continuous control means "use whatever tuning.json says".
STAGES = [
    ("stageLoopFilter", "deblock (H.264 loop filter)", 0.0),
    ("stageCdef",       "CDEF directional filter",     0.0),
    ("stageDeband",     "deband + grain",              0.0),
    ("stageTaa",        "temporal anti-alias",         1.0),
    ("stageOklab",      "Oklab saturation",            1.0),
    ("stageSiting",     "chroma siting",               1.0),
]
# Continuous stages, measured by zeroing them rather than by a flag.
CONTINUOUS = [
    ("sharpness",     "contrast-adaptive sharpen"),
    ("fine",          "fine-detail band"),
    ("deblock",       "post-upscale deblock"),
    ("sourceDeblock", "source deblock"),
    ("contrast",      "tone contrast"),
]


def bands(gray):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    b = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64) for r in (1.0, 2.5, 6.0)]
    return [gray - b[0], b[0] - b[1], b[1] - b[2]]


def correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0


def score(directory):
    """Averages the per-band correlation over every frame the bench wrote."""
    outputs = sorted(f for f in os.listdir(directory) if f.endswith("-detail.png"))
    rows = []
    for name in outputs:
        stem = name.split("-")[0]
        ref = os.path.join(directory, f"{stem}-reference.png")
        if not os.path.exists(ref):
            continue
        a = np.asarray(Image.open(os.path.join(directory, name)).convert("L"), dtype=np.float64)
        r = np.asarray(Image.open(ref).convert("L"), dtype=np.float64)
        if a.shape != r.shape:
            a = np.asarray(Image.open(os.path.join(directory, name)).convert("L")
                           .resize((r.shape[1], r.shape[0]), Image.LANCZOS), dtype=np.float64)
        ab, rb = bands(a), bands(r)
        rows.append((correlation(ab[0], rb[0]), correlation(ab[1], rb[1]),
                     correlation(ab[2], rb[2]),
                     10 * np.log10(255 * 255 / max(((a - r) ** 2).mean(), 1e-9))))
    if not rows:
        return None
    m = np.array(rows).mean(axis=0)
    return {"fine": m[0], "mid": m[1], "coarse": m[2], "psnr": m[3]}


def run(tuning, clip, reference, frames, learned):
    """One bench run with a given tuning, returning its score and stage cost."""
    out = tempfile.mkdtemp(prefix="ablate-")
    path = os.path.join(out, "tuning.json")
    with open(path, "w") as fh:
        json.dump(tuning, fh)
    env = dict(os.environ, LUCID_TUNING=path)
    if learned:
        env["LUCID_LEARNED"] = "1"
    started = time.time()
    p = subprocess.run([APP, "--bench", clip, reference, out, "0", str(frames)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-600:], p.stderr[-600:])
        shutil.rmtree(out, ignore_errors=True)
        return None, None
    # The bench prints one summary line:
    #   frames N  ·  low-latency 4x: A ms  ·  detail stage: B ms  ·  temporal 4x: C ms
    # Match the label rather than scanning for numbers - taking the last number
    # on the line silently reported the temporal figure as the detail cost.
    upscale_ms = detail_ms = None
    for line in p.stdout.splitlines():
        m = re.search(r"detail stage:\s*([0-9.]+)", line)
        if m:
            detail_ms = float(m.group(1))
        m = re.search(r"low-latency 4.:\s*([0-9.]+)", line)
        if m:
            upscale_ms = float(m.group(1))
    result = score(out)
    shutil.rmtree(out, ignore_errors=True)
    return result, (detail_ms, upscale_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="bbb-360p-350k.mp4")
    parser.add_argument("--reference", default="")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--tuning", default=os.path.join(ROOT, "Tools/tuning.json"))
    parser.add_argument("--apple", action="store_true", help="measure over Apple's scaler instead")
    args = parser.parse_args()

    clip = args.clip if os.path.isabs(args.clip) else os.path.join(ROOT, "TestSite", args.clip)
    # The only clips with a true 1080p counterpart on disk.
    known = {"bbb-360p-350k.mp4": "bbb-1080p60-1700k.mp4",
             "bbb-360p-900k.mp4": "bbb-1080p60-1700k.mp4",
             "low-bitrate-test.mp4": "low-bitrate-1080p.mp4"}
    reference = args.reference or os.path.join(
        ROOT, "TestSite", known.get(os.path.basename(clip), ""))
    if not os.path.exists(clip):
        raise SystemExit(f"no clip at {clip}")
    if not os.path.exists(reference):
        raise SystemExit(f"no reference at {reference}")

    with open(args.tuning) as fh:
        base = json.load(fh)
    learned = not args.apple
    base_label = "SPAN on the Neural Engine" if learned else "Apple's scaler"
    print(f"base: {base_label}, {os.path.basename(clip)}, {args.frames} frames\n")

    reference_score, (reference_ms, upscale_ms) = run(base, clip, reference, args.frames, learned)
    if reference_score is None:
        raise SystemExit("baseline bench failed")
    print(f"{'stage':34s} {'fine':>8s} {'mid':>8s} {'coarse':>8s} {'PSNR':>8s} {'ms':>7s}")
    print(f"{'— everything as it ships —':34s} {reference_score['fine']:8.4f} "
          f"{reference_score['mid']:8.4f} {reference_score['coarse']:8.4f} "
          f"{reference_score['psnr']:8.2f} {reference_ms or 0:7.2f}")
    print(f"{'  (upscaler itself)':34s} {'':8s} {'':8s} {'':8s} {'':8s} {upscale_ms or 0:7.2f}")
    print()

    rows = []
    for key, label, shipped in STAGES:
        tuning = dict(base)
        # Flip it: measure what changing it does, in whichever direction it is
        # not currently set. An off stage is asked "would you help?"; an on
        # stage is asked "are you earning your place?".
        tuning[key] = 0.0 if base.get(key, shipped) > 0.5 else 1.0
        direction = "off→on" if tuning[key] > 0.5 else "on→off"
        result, (ms, _) = run(tuning, clip, reference, args.frames, learned)
        if result is None:
            print(f"{label:34s}  bench failed"); continue
        rows.append((label, direction, result, ms))

    for key, label in CONTINUOUS:
        if abs(float(base.get(key, 0))) < 1e-6:
            continue          # already zero; nothing to remove
        tuning = dict(base); tuning[key] = 0.0
        result, (ms, _) = run(tuning, clip, reference, args.frames, learned)
        if result is None:
            print(f"{label:34s}  bench failed"); continue
        rows.append((label, "on→off", result, ms))

    for label, direction, r, ms in rows:
        d = {k: r[k] - reference_score[k] for k in ("fine", "mid", "coarse", "psnr")}
        cost = (ms - reference_ms) if (ms is not None and reference_ms is not None) else 0.0
        print(f"{label + ' ' + direction:34s} {d['fine']:+8.4f} {d['mid']:+8.4f} "
              f"{d['coarse']:+8.4f} {d['psnr']:+8.2f} {cost:+7.2f}")
    print("\nDeltas are (with the change) minus (as it ships). A stage that is "
          "off and shows positive deltas is worth turning on; one that is on "
          "and shows positive deltas for turning it off is not earning its ms.")


if __name__ == "__main__":
    main()
