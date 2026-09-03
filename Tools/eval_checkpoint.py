#!/usr/bin/env python3
"""Scores a trained checkpoint on footage it has never seen.

A model's own validation set is held-out *patches of its own corpus*, which
makes two models trained on different corpora incomparable: the gap between
their curves is partly "different test set", not "better model". This scores
any number of checkpoints on the same disjoint clip, next to a plain Lanczos
upscale — the anchor this project measures everything against, because the
shipping pipeline once scored below it without anyone noticing.

Reported per tier, because the tiers are not equally hard and an average over
them hides which end of the ladder is failing.

  .venv-convert/bin/python Tools/eval_checkpoint.py \\
      --corpus .build/eval-corpus \\
      Model/weights/span_ch32u.pth Model/weights/span_ch32u_animation_only.pth
"""
import argparse, collections, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_span import Unshuffled  # noqa: E402


def bands(gray):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    blurred = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)
               for r in (1.0, 2.5, 6.0)]
    return [gray - blurred[0], blurred[0] - blurred[1], blurred[1] - blurred[2]]


def correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0


def score(output, reference):
    a = np.asarray(output.convert("L"), dtype=np.float64)
    r = np.asarray(reference.convert("L"), dtype=np.float64)
    ab, rb = bands(a), bands(r)
    return (correlation(ab[0], rb[0]), correlation(ab[1], rb[1]),
            correlation(ab[2], rb[2]),
            10 * np.log10(255 * 255 / max(((a - r) ** 2).mean(), 1e-9)))


def load(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = Unshuffled(state.get("channels", 32)).eval().to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model, state.get("step", "?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=".build/eval-corpus")
    parser.add_argument("--device", default="mps")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "mps"
                          or torch.backends.mps.is_available() else "cpu")
    names = sorted(os.listdir(os.path.join(args.corpus, "lr")))
    print(f"{len(names)} pairs from {args.corpus}, on {device}\n")

    # The anchor first: whatever a model scores only means something relative to
    # doing nothing clever.
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for name in names:
        lr = Image.open(os.path.join(args.corpus, "lr", name)).convert("RGB")
        hr = Image.open(os.path.join(args.corpus, "hr", name)).convert("RGB")
        tier = f"{lr.width}x{lr.height}"
        rows["lanczos"][tier].append(score(lr.resize(hr.size, Image.LANCZOS), hr))

    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"missing: {path}"); continue
        model, step = load(path, device)
        label = os.path.basename(path).replace("span_", "").replace(".pth", "")
        with torch.no_grad():
            for name in names:
                lr = Image.open(os.path.join(args.corpus, "lr", name)).convert("RGB")
                hr = Image.open(os.path.join(args.corpus, "hr", name)).convert("RGB")
                x = torch.from_numpy(np.asarray(lr, dtype=np.float32) / 255)
                x = x.permute(2, 0, 1).unsqueeze(0).to(device)
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                out = Image.fromarray((y * 255).round().astype(np.uint8))
                rows[f"{label}@{step}"][f"{lr.width}x{lr.height}"].append(score(out, hr))

    tiers = sorted({t for v in rows.values() for t in v},
                   key=lambda s: int(s.split("x")[0]))
    print(f"{'variant':28s} {'tier':10s} {'fine':>8s} {'mid':>8s} {'coarse':>8s} {'PSNR':>8s}")
    for label in rows:
        for tier in tiers:
            if not rows[label][tier]:
                continue
            m = np.array(rows[label][tier]).mean(axis=0)
            print(f"{label:28s} {tier:10s} {m[0]:8.4f} {m[1]:8.4f} {m[2]:8.4f} {m[3]:8.2f}")
        allrows = [r for t in tiers for r in rows[label][t]]
        m = np.array(allrows).mean(axis=0)
        print(f"{label:28s} {'ALL':10s} {m[0]:8.4f} {m[1]:8.4f} {m[2]:8.4f} {m[3]:8.2f}\n")


if __name__ == "__main__":
    main()
