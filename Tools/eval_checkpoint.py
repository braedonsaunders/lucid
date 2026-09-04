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
from train_span import Unshuffled, bank_frames  # noqa: E402


# ---- the perceptual scoreboard -----------------------------------------------
#
# This project spent its whole life optimising fine-band correlation with ground
# truth. That metric was chosen for a good reason - Apple's scaler fooled us by
# inventing mottled foliage that looked like detail and was not - but it is
# scale-invariant and correlation-based, so it CANNOT reward invented detail by
# construction. Every downstream decision followed from that: sharpening off,
# deband off, grain off, GAN and diffusion losses never considered. The right
# answer to "our upscaler invents bad detail" is "invent good detail", not
# "never invent detail".
#
# So correlation is demoted to a guard - is this still the same scene - and the
# objective is now perceptual:
#
#   LPIPS    lower is better. Deep-feature distance; rewards detail that is
#            plausible, not detail that is identical.
#   DISTS    lower is better. Explicitly tolerant of texture that differs
#            pixel-wise but matches statistically, which is exactly what a good
#            hallucinated texture does.
#   detail   fine-band energy as a fraction of the reference's. 1.0 means as
#            much high-frequency structure as the truth. Under the old metric
#            this number was untrustworthy because it cannot tell recovered
#            from invented - under this one that is the point.
#   BRISQUE  lower is better, and needs no reference at all, so it says whether
#            the picture looks good on its own terms.

_lpips = None
_dists = None


def perceptual(output, reference, device):
    """LPIPS, DISTS and BRISQUE for one image pair, plus the detail ratio."""
    global _lpips, _dists
    import lpips as _lp, piq
    if _lpips is None:
        _lpips = _lp.LPIPS(net="alex", verbose=False).to(device)
        _dists = piq.DISTS().to(device)
    a = torch.from_numpy(np.asarray(output, dtype=np.float32) / 255)
    b = torch.from_numpy(np.asarray(reference, dtype=np.float32) / 255)
    a = a.permute(2, 0, 1).unsqueeze(0).to(device)
    b = b.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        # LPIPS wants -1..1, DISTS and BRISQUE want 0..1.
        l = float(_lpips(a * 2 - 1, b * 2 - 1))
        d = float(_dists(a, b))
        try:
            q = float(piq.brisque(a.clamp(0, 1)))
        except Exception:
            q = float("nan")
    return l, d, q


def detail_ratio(output, reference):
    """Fine-band energy relative to the reference. 1.0 is as much fine structure
    as the truth has; below 1 is soft, above 1 is embellished."""
    a = np.asarray(output.convert("L"), dtype=np.float64)
    r = np.asarray(reference.convert("L"), dtype=np.float64)
    ab, rb = bands(a), bands(r)
    energy = np.sqrt((rb[0] ** 2).mean())
    return float(np.sqrt((ab[0] ** 2).mean()) / energy) if energy > 0 else 0.0


def bands(gray):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    blurred = [np.asarray(im.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)
               for r in (1.0, 2.5, 6.0)]
    return [gray - blurred[0], blurred[0] - blurred[1], blurred[1] - blurred[2]]


def correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else 0.0


def score(output, reference, device):
    a = np.asarray(output.convert("L"), dtype=np.float64)
    r = np.asarray(reference.convert("L"), dtype=np.float64)
    ab, rb = bands(a), bands(r)
    l, d, q = perceptual(output, reference, device)
    return (l, d, q, detail_ratio(output, reference),
            correlation(ab[0], rb[0]),
            10 * np.log10(255 * 255 / max(((a - r) ** 2).mean(), 1e-9)))


def load(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    frames = state.get("frames", 1)
    model = Unshuffled(state.get("channels", 32), frames=frames,
                       version=state.get("version", 1)).eval().to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model, state.get("step", "?"), frames


def stack_input(corpus, name, frames, available):
    """Builds the model's input: oldest frame first, current frame last.

    When a temporal model is scored on a corpus without history - or at a scene
    cut in real playback - the current frame is repeated. That is the honest
    degenerate case: it tells the model nothing new rather than handing it an
    unrelated shot, which is the failure that would show up as ghosting."""
    from PIL import Image as _Image
    current = _Image.open(os.path.join(corpus, "lr", name)).convert("RGB")
    layers = []
    for directory in available[:frames - 1]:
        path = os.path.join(corpus, directory, name)
        layers.append(_Image.open(path).convert("RGB") if os.path.exists(path) else current)
    while len(layers) < frames - 1:
        layers.append(current)
    layers.append(current)
    array = np.concatenate([np.asarray(l, dtype=np.float32) / 255 for l in layers], axis=2)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=".build/eval-corpus")
    parser.add_argument("--device", default="mps")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "mps"
                          or torch.backends.mps.is_available() else "cpu")
    names = sorted(os.listdir(os.path.join(args.corpus, "lr")))
    _, available = bank_frames(args.corpus)
    history = f", {len(available)} history frame(s)" if available else ", no history"
    print(f"{len(names)} pairs from {args.corpus}, on {device}{history}\n")

    # The anchor first: whatever a model scores only means something relative to
    # doing nothing clever.
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for name in names:
        lr = Image.open(os.path.join(args.corpus, "lr", name)).convert("RGB")
        hr = Image.open(os.path.join(args.corpus, "hr", name)).convert("RGB")
        tier = f"{lr.width}x{lr.height}"
        rows["lanczos"][tier].append(score(lr.resize(hr.size, Image.LANCZOS), hr, device))

    # A sweep writes the same filename into one directory per setting, so a
    # label taken from the basename alone silently averages every checkpoint in
    # the sweep into a single row - four models, one number, and nothing in the
    # output says so. Where basenames collide, the directory is the label.
    basenames = [os.path.basename(p) for p in args.checkpoints]
    labels = {}
    for path, name in zip(args.checkpoints, basenames):
        stem = name.replace("span_", "").replace(".pth", "")
        if basenames.count(name) > 1:
            stem = f"{os.path.basename(os.path.dirname(path))}/{stem}"
        labels[path] = stem

    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"missing: {path}"); continue
        model, step, frames = load(path, device)
        label = labels[path]
        with torch.no_grad():
            for name in names:
                hr = Image.open(os.path.join(args.corpus, "hr", name)).convert("RGB")
                x, lr = stack_input(args.corpus, name, frames, available)
                y = model(x.to(device)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                out = Image.fromarray((y * 255).round().astype(np.uint8))
                rows[f"{label}@{step}"][f"{lr.width}x{lr.height}"].append(score(out, hr, device))

    tiers = sorted({t for v in rows.values() for t in v},
                   key=lambda s: int(s.split("x")[0]))
    print("LPIPS and DISTS lower is better. detail is fine-band energy against the")
    print("reference: 1.00 is as much high-frequency structure as the truth has.")
    print("corr is the old fidelity metric, kept as a guard, not the objective.\n")
    print(f"{'variant':26s} {'tier':10s} {'LPIPS':>8s} {'DISTS':>8s} "
          f"{'BRISQUE':>8s} {'detail':>7s} {'corr':>7s} {'PSNR':>7s}")
    for label in rows:
        for tier in tiers:
            if not rows[label][tier]:
                continue
            m = np.nanmean(np.array(rows[label][tier]), axis=0)
            print(f"{label:26s} {tier:10s} {m[0]:8.4f} {m[1]:8.4f} {m[2]:8.1f} "
                  f"{m[3]:7.2f} {m[4]:7.4f} {m[5]:7.2f}")
        allrows = [r for t in tiers for r in rows[label][t]]
        m = np.nanmean(np.array(allrows), axis=0)
        print(f"{label:26s} {'ALL':10s} {m[0]:8.4f} {m[1]:8.4f} {m[2]:8.1f} "
              f"{m[3]:7.2f} {m[4]:7.4f} {m[5]:7.2f}\n")


if __name__ == "__main__":
    main()
