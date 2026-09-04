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

Three numbers, because the question has three sides and no single metric answers
it. LPIPS and DISTS say whether it LOOKS right, and they are the objective.
Band energy says whether there is ENOUGH detail - a picture can be perfectly
correlated with the truth and still carry a third of its high-frequency energy,
which is exactly what a purely reconstructive model does. Band correlation says
whether it is the RIGHT detail, and it is a guard against hallucination rather
than a target.

Using correlation alone was a mistake this project made for a long time: it is
scale-invariant, so it cannot distinguish 32% of the right detail from 100% of
it, and every stage that synthesised texture was scored as loss.

  .venv-convert/bin/python Tools/ablate.py --clip sintel-scene-360p-300k.mp4
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

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


def scored_frames(directory):
    """The frames the scorer reads. The tripwire hashes exactly this list and
    not the directory at large: a tripwire that hashes what the bench claims to
    have written can be satisfied by a stage that writes correctly and is then
    scored on something else. One definition, used by both, so they cannot
    drift apart."""
    return sorted(f for f in os.listdir(directory) if f.endswith("-detail.png"))


def score(directory):
    """Scores every frame the bench wrote, on the perceptual scoreboard.

    This used to average per-band correlation with the reference, and that is
    what made the verdicts below untrustworthy: correlation is scale-invariant
    and reference-anchored, so it scores ALL synthesised detail as loss. A stage
    whose whole job is to synthesise - grain, debanding, sharpening - could only
    ever measure as harmful. Those stages were then switched off on the strength
    of it.

    So the objective is DISTS and LPIPS, correlation is kept as a guard rather
    than a target, and BRISQUE is reported but never optimised against, because
    no-reference IQA is gameable by oversharpening and correlation will not catch
    a halo - a ringed edge still correlates perfectly well.
    """
    import torch
    from eval_checkpoint import perceptual, detail_ratio  # the one scoreboard

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    outputs = scored_frames(directory)
    rows = []
    for name in outputs:
        stem = name.split("-")[0]
        ref_path = os.path.join(directory, f"{stem}-reference.png")
        if not os.path.exists(ref_path):
            continue
        out = Image.open(os.path.join(directory, name)).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        if out.size != ref.size:
            out = out.resize(ref.size, Image.LANCZOS)
        lpips_v, dists_v, brisque_v = perceptual(out, ref, device)
        a = np.asarray(out.convert("L"), dtype=np.float64)
        r = np.asarray(ref.convert("L"), dtype=np.float64)
        rows.append((dists_v, lpips_v, detail_ratio(out, ref), brisque_v,
                     correlation(bands(a)[0], bands(r)[0])))
    if not rows:
        return None
    m = np.nanmean(np.array(rows, dtype=np.float64), axis=0)
    return {"dists": m[0], "lpips": m[1], "detail": m[2],
            "brisque": m[3], "fine": m[4]}


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
        return None, (None, None), None
    # The bench prints one summary line:
    #   frames N  ·  low-latency 4x: A ms  ·  detail stage: B ms  ·  temporal 4x: C ms
    # Match the label rather than scanning for numbers - taking the last number
    # on the line silently reported the temporal figure as the detail cost.
    #
    # The trailing "ms" is load-bearing. `describe()` prints a line with the SAME
    # label and a resolution after it - "low-latency 4x: 2560x1440 420v ..." - so
    # a pattern that stops at the first number captures 2560 and calls it
    # milliseconds. It only surfaces at small frame counts, because that is when
    # the real summary reports "n/a" for want of warm samples and there is
    # nothing later to overwrite the bad capture. A frame width is a completely
    # plausible millisecond figure, which is exactly why it went unnoticed.
    upscale_ms = detail_ms = None
    for line in p.stdout.splitlines():
        m = re.search(r"detail stage:\s*([0-9.]+)\s*ms", line)
        if m:
            detail_ms = float(m.group(1))
        m = re.search(r"low-latency 4.:\s*([0-9.]+)\s*ms", line)
        if m:
            upscale_ms = float(m.group(1))
    result = score(out)
    # Fingerprint the actual pixels, so a toggle that changed nothing can be
    # told apart from a toggle that changed nothing *useful*. See `fingerprint`.
    print_ = fingerprint(out)
    shutil.rmtree(out, ignore_errors=True)
    return result, (detail_ms, upscale_ms), print_


def fingerprint(directory):
    """A hash of every frame the bench wrote.

    This exists because four separate bugs in this project have had the same
    shape: a number that could not move, reading as a finding. A delivery
    counter that scored every frame as unmatched. An ablation harness that
    ignored the setting it was toggling. Colour tags applied only where absent,
    so the control could never take effect. In every case the code computing the
    number was correct, and what was wrong was whether the number was capable of
    varying at all. No amount of care in the metric catches that, and no test
    suite does either.

    So before any delta is believed, the pixels are compared. If flipping a
    stage produces byte-identical output, the honest report is INERT - the
    experiment did not run - and not 0.0000, which reads as "measured, no
    effect" and is how three of those four bugs survived as long as they did.
    """
    h = hashlib.sha256()
    for name in scored_frames(directory):
        with open(os.path.join(directory, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


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
    # A reference has to carry MORE fine-band detail than the input it is the
    # truth for. Two of the pairs that shipped with this repo do not:
    #
    #     pair          input   reference
    #     bbb           7.943     6.928    1080p60 at 1700 kbps - too compressed
    #     low-bitrate   4.598     1.158    a quarter of its own input
    #     crowdrun      8.110     8.254    valid
    #     dinner        0.928     0.944    valid
    #
    # A soft reference rewards softness, so every setting measured against one
    # is biased toward doing less. Both bad pairs are refused rather than
    # quietly scored.
    known = {"crowdrun-360p-350k.mp4": "crowdrun-1080p.mp4",
             "dinner-360p-350k.mp4": "dinner-1080p.mp4"}
    refused = {"bbb-360p-350k.mp4", "bbb-360p-900k.mp4", "low-bitrate-test.mp4"}
    if os.path.basename(clip) in refused and not args.reference:
        raise SystemExit(
            f"{os.path.basename(clip)} has no valid reference on this machine - its "
            "1080p counterpart carries less fine detail than the 360p input, so any "
            "score against it is biased toward softness. Use crowdrun-360p-350k.mp4 "
            "or dinner-360p-350k.mp4, both built from clean 4K masters.")
    reference = args.reference or os.path.join(
        ROOT, "TestSite", known.get(os.path.basename(clip), ""))
    if not os.path.exists(clip):
        raise SystemExit(f"no clip at {clip}")
    if not os.path.exists(reference):
        raise SystemExit(f"no reference at {reference}")

    with open(args.tuning) as fh:
        base = json.load(fh)
    learned = not args.apple
    base_label = "SPAN, trained ch32u" if learned else "Apple's scaler"
    print(f"base: {base_label}, {os.path.basename(clip)}, {args.frames} frames\n")

    reference_score, (reference_ms, upscale_ms), base_print = run(base, clip, reference, args.frames, learned)
    if reference_score is None:
        raise SystemExit("baseline bench failed")
    print(f"{'stage':34s} {'DISTS':>9s} {'LPIPS':>9s} {'detail':>9s} "
          f"{'BRISQUE':>9s} {'fine':>9s} {'ms':>7s}")
    print(f"{'— everything as it ships —':34s} {reference_score['dists']:9.4f} "
          f"{reference_score['lpips']:9.4f} {reference_score['detail']:9.4f} "
          f"{reference_score['brisque']:9.3f} {reference_score['fine']:9.4f} "
          f"{reference_ms or 0:7.2f}")
    print(f"{'  (upscaler itself)':34s} {'':9s} {'':9s} {'':9s} {'':9s} {'':9s} "
          f"{upscale_ms or 0:7.2f}")
    print()

    rows = []
    for key, label, shipped in STAGES:
        tuning = dict(base)
        # Flip it: measure what changing it does, in whichever direction it is
        # not currently set. An off stage is asked "would you help?"; an on
        # stage is asked "are you earning your place?".
        tuning[key] = 0.0 if base.get(key, shipped) > 0.5 else 1.0
        direction = "off→on" if tuning[key] > 0.5 else "on→off"
        result, (ms, _), variant_print = run(tuning, clip, reference, args.frames, learned)
        if result is None:
            print(f"{label:34s}  bench failed"); continue
        rows.append((label, direction, result, ms, variant_print == base_print))

    for key, label in CONTINUOUS:
        if abs(float(base.get(key, 0))) < 1e-6:
            continue          # already zero; nothing to remove
        tuning = dict(base); tuning[key] = 0.0
        result, (ms, _), variant_print = run(tuning, clip, reference, args.frames, learned)
        if result is None:
            print(f"{label:34s}  bench failed"); continue
        rows.append((label, "on→off", result, ms, variant_print == base_print))

    inert = []
    for label, direction, r, ms, identical in rows:
        if identical:
            # The pixels did not change, so there is nothing to score. Saying
            # INERT is the difference between "this stage does nothing" and
            # "this stage could not be reached", and only one of those is a
            # finding about the stage.
            print(f"{label + ' ' + direction:34s} {'INERT — output byte-identical, not measured':>49s}")
            inert.append(label)
            continue
        d = {k: r[k] - reference_score[k]
             for k in ("dists", "lpips", "detail", "brisque", "fine")}
        cost = (ms - reference_ms) if (ms is not None and reference_ms is not None) else 0.0
        # DISTS and LPIPS are distances, so a NEGATIVE delta is an improvement.
        # Printed with the sign flipped so every column reads "higher is better"
        # and a row cannot be misread at a glance.
        print(f"{label + ' ' + direction:34s} {-d['dists']:+9.4f} {-d['lpips']:+9.4f} "
              f"{d['detail']:+9.4f} {-d['brisque']:+9.3f} {d['fine']:+9.4f} {cost:+7.2f}")
    print("\nDeltas are (with the change) minus (as it ships), sign-corrected so "
          "POSITIVE IS BETTER in every column. DISTS is the objective; LPIPS "
          "seconds it. `detail` is fine-band energy against the truth's 1.00, so "
          "positive means more of the structure the source actually has.")
    if inert:
        print(f"\nINERT: {', '.join(inert)}. Flipping these changed no pixel at all, so "
              "the experiment did not run. That is a bug in the control or the harness, "
              "not a measurement of the stage - fix it before reading a zero as a result.")
    print("BRISQUE is reported, never optimised against: no-reference IQA is "
          "gameable by oversharpening, and `fine` will not catch it either, "
          "because a ringed edge still correlates perfectly well. If a row wins "
          "on BRISQUE while losing on DISTS, look at it before believing it.")


if __name__ == "__main__":
    main()
