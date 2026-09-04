#!/usr/bin/env python3
"""Quantises the shipping upscaler's ACTIVATIONS to int8 and measures whether it helps.

Why activations and not weights. Measured on the Neural Engine, this trunk is
activation-bandwidth bound rather than MAC bound: latency scales with channels
rather than channels squared (ch16 0.63x, ch20 0.75x, ch24 0.90x against ch28),
and int8 *weight* quantisation moved 30.6 ms to 29.3 ms - which is nothing,
because the weights are not the traffic. That result is the evidence for trying
this one: if the traffic is activations, quantising activations is the thing
that should move it. Nobody had tested it.

This is conversion-side only. No retraining, no corpus, no GPU.

  .venv-convert/bin/python Tools/quantize_w8a8.py                  # 640x360
  .venv-convert/bin/python Tools/quantize_w8a8.py --all            # whole ladder

CHECK THE OUTPUT LEVEL, NOT JUST THE METRIC. Two variants of this model have
already converted to a near-black image and still scored ABOVE the Lanczos
anchor, because correlation is scale-invariant. Every model this script produces
is run on a real frame and rejected outright if its mean level drifts more than
20% from the source, before any timing is believed. That check is the whole
reason to trust the numbers below it.
"""
import argparse, os, shutil, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_span import Unshuffled  # noqa: E402

import coremltools as ct
from coremltools.optimize.coreml import (  # noqa: E402
    OpLinearQuantizerConfig, OptimizationConfig,
    linear_quantize_activations, linear_quantize_weights,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = [(256, 144), (320, 180), (432, 240), (480, 270), (640, 360), (864, 480)]
CALIBRATION = os.path.join(ROOT, ".build/eval-corpus/lr")
BENCH = os.path.join(ROOT, ".build/modelbench")


# ---- the level check --------------------------------------------------------

def level_ratio(output, source):
    """Mean level of the output over the mean level of the source. A correct
    upscaler keeps the overall level of the picture; ~1.0 is right. This is the
    same test as model_gallery.sanity(), kept here so a bad conversion cannot
    reach the timing table."""
    a = np.asarray(output.convert("RGB"), dtype=np.float64).mean()
    b = np.asarray(source.convert("RGB"), dtype=np.float64).mean()
    return a / max(b, 1e-6)


def calibration_images(width, height, limit=12):
    """Real decoded frames at this tier, not noise. Activation ranges calibrated
    on random input would bracket values the network never actually sees."""
    if not os.path.isdir(CALIBRATION):
        return []
    picked = []
    for name in sorted(os.listdir(CALIBRATION)):
        try:
            image = Image.open(os.path.join(CALIBRATION, name)).convert("RGB")
        except Exception:
            continue
        if image.size == (width, height):
            picked.append(image)
        if len(picked) >= limit:
            break
    return picked


# ---- conversion -------------------------------------------------------------

def build_fp16(model, width, height, out_path):
    """The baseline, converted exactly the way the shipping ladder is, so the
    comparison is against what actually ships rather than a fresh recipe."""
    import runpy
    ns = runpy.run_path(os.path.join(ROOT, "Tools/convert_span.py"), run_name="not_main")
    wrapped = ns["ImageRange"](model).eval()
    example = torch.rand(1, 3, height, width)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=(1, 3, height, width),
                             color_layout=ct.colorlayout.RGB, scale=1 / 255.0)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ns["selective_fp16"](),
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(out_path)
    return mlmodel


def quantize(mlmodel, images, weights_too=True):
    """W8A8: int8 activations, and int8 weights alongside them.

    Weights on their own were already measured as worthless here. They are
    included because once activations are int8 the compute can actually run in
    int8, which is the only way the weight quantisation pays for itself."""
    sample = [{"input": image} for image in images]
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    quantized = linear_quantize_activations(mlmodel, config, sample_data=sample)
    if weights_too:
        quantized = linear_quantize_weights(quantized, config)
    return quantized


# ---- measurement ------------------------------------------------------------

def bench(path):
    """Milliseconds per prediction, measured by the same harness the ladder
    timings in LearnedUpscaler come from."""
    if not os.path.exists(BENCH):
        return None
    try:
        r = subprocess.run([BENCH, path], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    best = None
    for token in r.stdout.replace("ms", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.1 < value < 10000:
            best = value if best is None else min(best, value)
    return best


def predict(path, image):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    name = list(model.get_spec().description.input)[0].name
    out = model.predict({name: image})
    key = list(model.get_spec().description.output)[0].name
    return out[key]


def run_tier(model, width, height, keep):
    print(f"\n── {width}x{height} " + "─" * 40)
    images = calibration_images(width, height)
    if not images:
        print(f"   no {width}x{height} calibration frames in {CALIBRATION} - skipped")
        return None
    print(f"   {len(images)} calibration frames")

    work = os.path.join(ROOT, ".build/w8a8")
    os.makedirs(work, exist_ok=True)
    fp16_path = os.path.join(work, f"fp16_{width}x{height}.mlpackage")
    int8_path = os.path.join(work, f"w8a8_{width}x{height}.mlpackage")

    started = time.time()
    fp16 = build_fp16(model, width, height, fp16_path)
    print(f"   fp16 baseline converted ({time.time()-started:.0f}s)")

    started = time.time()
    try:
        w8a8 = quantize(fp16, images)
        w8a8.save(int8_path)
    except Exception as error:
        print(f"   W8A8 conversion FAILED: {repr(error)[:300]}")
        return None
    print(f"   W8A8 converted ({time.time()-started:.0f}s)")

    probe = images[0]
    fp16_out = predict(fp16_path, probe)
    int8_out = predict(int8_path, probe)
    fp16_level = level_ratio(fp16_out, probe)
    int8_level = level_ratio(int8_out, probe)

    a = np.asarray(fp16_out.convert("RGB"), dtype=np.float64)
    b = np.asarray(int8_out.convert("RGB"), dtype=np.float64)
    drift = float(np.abs(a - b).mean())

    fp16_ms = bench(fp16_path)
    int8_ms = bench(int8_path)

    size_fp16 = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(fp16_path) for f in fs) / 1024
    size_int8 = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(int8_path) for f in fs) / 1024

    ok = 0.8 < int8_level < 1.2
    print(f"   level     fp16 {fp16_level:.3f}   W8A8 {int8_level:.3f}   "
          f"{'OK' if ok else 'REJECTED - output level is wrong, ignore the timing'}")
    print(f"   drift     {drift:.2f}/255 mean absolute, fp16 vs W8A8")
    print(f"   size      fp16 {size_fp16:.0f} kB   W8A8 {size_int8:.0f} kB")
    if fp16_ms and int8_ms:
        print(f"   latency   fp16 {fp16_ms:.2f} ms   W8A8 {int8_ms:.2f} ms   "
              f"({int8_ms/fp16_ms:.2f}x)")
    else:
        print("   latency   modelbench unavailable")

    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return dict(size=f"{width}x{height}", fp16_ms=fp16_ms, int8_ms=int8_ms,
                level=int8_level, drift=drift, ok=ok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="Model/weights/span_ch32u.pth")
    parser.add_argument("--all", action="store_true", help="the whole ladder")
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--keep", action="store_true", help="keep the .mlpackages")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    channels = state.get("channels", 32)
    frames = state.get("frames", 1)
    model = Unshuffled(channels, frames=frames, version=state.get("version", 1)).eval()
    model.load_state_dict(state["model"] if "model" in state else state)
    print(f"{args.weights} step {state.get('step','?')}: ch{channels}, {frames} frame(s), "
          f"{sum(p.numel() for p in model.parameters())/1000:.0f}K parameters")

    if args.all:
        tiers = LADDER
    else:
        w, h = args.size.lower().split("x")
        tiers = [(int(w), int(h))]

    rows = [r for r in (run_tier(model, w, h, args.keep) for w, h in tiers) if r]
    if len(rows) > 1:
        print(f"\n{'size':>10} {'fp16':>9} {'W8A8':>9} {'ratio':>7} {'level':>7}  verdict")
        for r in rows:
            ratio = f"{r['int8_ms']/r['fp16_ms']:.2f}x" if r["fp16_ms"] and r["int8_ms"] else "-"
            print(f"{r['size']:>10} {r['fp16_ms'] or 0:>8.2f}m {r['int8_ms'] or 0:>8.2f}m "
                  f"{ratio:>7} {r['level']:>7.3f}  {'ok' if r['ok'] else 'REJECTED'}")


if __name__ == "__main__":
    main()
