#!/usr/bin/env python3
"""Runs every converted model over the same frames and writes a gallery.

A table cannot tell you whether a super-resolution model looks right; only
flipping between two versions of the same frame can. This produces the images
for TestSite/models.html, and the numbers beside them.

  .venv-convert/bin/python Tools/model_gallery.py
"""
import glob, json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageFilter
import coremltools as ct

OUT = "output/gallery"
# Apple's scaler and the shipping pipeline come from the offline bench, which
# has already written them at these frame indices.
BENCH = ".build/bench/rad4"


def bands(gray):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    b = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64) for r in (1.0, 2.5, 6.0)]
    return [gray - b[0], b[0] - b[1], b[1] - b[2]]


def correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0


def score(image, reference):
    """Correlation per band is the metric that separates detail that was
    recovered from detail that was invented. Energy alone cannot tell them
    apart, and has misled this project twice."""
    a = np.asarray(image.convert("L"), dtype=np.float64)
    r = np.asarray(reference.convert("L"), dtype=np.float64)
    if a.shape != r.shape:
        a = np.asarray(image.convert("L").resize(reference.size, Image.LANCZOS), dtype=np.float64)
    ab, rb = bands(a), bands(r)
    return {
        "fine": round(correlation(ab[0], rb[0]), 4),
        "mid": round(correlation(ab[1], rb[1]), 4),
        "coarse": round(correlation(ab[2], rb[2]), 4),
        "psnr": round(float(10 * np.log10(255 * 255 / max(((a - r) ** 2).mean(), 1e-9))), 2),
    }


def sanity(image, source):
    """A correct upscaler keeps the overall level of the picture. Catching that
    here matters: SPAN x2 converted at FLOAT16 silently returned a near-black
    image, and correlation is scale-invariant so it still scored respectably."""
    a = np.asarray(image.convert("RGB"), dtype=np.float64).mean()
    b = np.asarray(source.convert("RGB"), dtype=np.float64).mean()
    return 0.8 < a / max(b, 1e-6) < 1.2


def run_model(path, image):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    name = list(model.get_spec().description.input)[0].name
    started = time.perf_counter()
    result = model.predict({name: image})
    elapsed = (time.perf_counter() - started) * 1000
    out = list(result.values())[0]
    return (out if isinstance(out, Image.Image) else Image.fromarray(out)), elapsed


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = sorted(glob.glob(f"{BENCH}/*-input.png"))[:3]
    if not frames:
        print(f"no bench frames in {BENCH}; run --bench first"); return

    candidates = [
        ("ch32u (trained on our codec corpus)", "Model/SPAN_x4_ch32u_480x270.mlpackage", 4),
        ("SPAN ch28 (ships)",  "Model/SPAN_x4_ch28_480x270.mlpackage", 4),
        ("SPAN ch48",          "Model/SPAN_x4_ch48_480x270.mlpackage", 4),
        ("RealESRGAN x4v3",    "Model/RealESRGAN_x4v3_480x270.mlpackage", 4),
        ("EfRLFN x4",          "Model/EfRLFN_x4_480x270.mlpackage", 4),
    ]
    manifest = {"frames": []}

    for index, path in enumerate(frames):
        stem = os.path.basename(path).split("-")[0]
        source = Image.open(path).convert("RGB")
        reference = Image.open(path.replace("-input", "-reference")).convert("RGB")
        target = reference.size
        entry = {"id": stem, "source": f"{source.width}x{source.height}", "variants": []}

        def add(name, image, ms=None, note=""):
            if image.size != target:
                image = image.resize(target, Image.LANCZOS)
            filename = f"{stem}-{name.replace(' ', '_').replace('+', 'p')}.png"
            image.save(os.path.join(OUT, filename))
            entry["variants"].append({"name": name, "file": filename, "ms": ms,
                                      "note": note, **score(image, reference)})

        add("reference", reference, note="ground truth")
        add("source", source, note="what the browser gets")
        add("lanczos", source.resize(target, Image.LANCZOS), note="anchor")
        for label, bench_name, note in (("Apple scaler", "lowlatency", "the old upscaler"),
                                        ("Apple + grade", "detail", "old upscaler, new grade")):
            file = path.replace("-input", f"-{bench_name}")
            if os.path.exists(file):
                add(label, Image.open(file).convert("RGB"), note=note)

        for label, model_path, scale in candidates:
            if not os.path.isdir(model_path):
                continue
            try:
                output, ms = run_model(model_path, source)
                if not sanity(output, source):
                    print(f"  {label}: output level is wrong, skipping"); continue
                add(label, output, ms=round(ms, 2), note=f"learned {scale}x")
            except Exception as error:
                print(f"  {label}: {error}")

        manifest["frames"].append(entry)
        print(f"frame {stem}: {len(entry['variants'])} variants")

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {OUT}/manifest.json")


if __name__ == "__main__":
    main()
