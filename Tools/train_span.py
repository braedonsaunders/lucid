#!/usr/bin/env python3
"""Trains Lucid's upscaler on the codec-degradation corpus.

Two things make this worth doing rather than fine-tuning a stock checkpoint.

The architecture has no pretrained weights. Every published efficient-SR model
runs its trunk at the input resolution; measured on an M4 Pro's Neural Engine
that trunk is activation-bandwidth bound, so the cost is `channels x LR_area`
and the way to make it cheap is fewer pixels, not fewer channels. Folding a 2x2
block into the channel dimension first and letting the head do x8 takes 640x360
from 27.9 ms to 12.6 ms at slightly more capacity. Nobody ships weights for that.

And the degradation is the whole point. Stock checkpoints are trained on bicubic
downsampling, which is not what a 240 kbps VP9 stream does to a picture. That
mismatch - not the architecture - is the biggest single lever there is.

  .venv-convert/bin/python Tools/train_span.py --corpus .build/corpus4 --bank
  .venv-convert/bin/python Tools/train_span.py --corpus .build/corpus4 --steps 60000
"""
import argparse, json, math, os, random, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_span import SPAN  # noqa: E402  (loads the arch without BasicSR)

LR_PATCH = 64          # even, so the 2x unshuffle is exact
SCALE = 4
HR_PATCH = LR_PATCH * SCALE


# ---- model ------------------------------------------------------------------

class Unshuffled(nn.Module):
    """SPAN with its trunk run at quarter area.

    `img_range` is set to 1 and the mean to zero. Stock SPAN multiplies its
    input by 255 and never divides back, which pushes activations out of fp16
    range - that is what made two converted variants come back near-black while
    still scoring above the anchor, because correlation is scale-invariant.
    Training our own weights means the network simply works in 0-1 and the trap
    stops existing.
    """

    def __init__(self, channels=32, scale=SCALE):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.core = SPAN(num_in_ch=12, num_out_ch=3, feature_channels=channels,
                         upscale=scale * 2, img_range=1.0, rgb_mean=(0.0, 0.0, 0.0))
        # rgb_mean is shaped (1,3,1,1) and the trunk now takes 12 channels, so
        # replace it with something that broadcasts.
        self.core.mean = torch.zeros(1, 1, 1, 1)

    def forward(self, x):
        return self.core(self.unshuffle(x))


def charbonnier(a, b, eps=1e-3):
    """Robust L1. Plain L2 chases the average and gives back a blurred picture,
    which is exactly the failure this project is trying to escape."""
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


# ---- patch bank -------------------------------------------------------------

def pair_correlation(lr, hr):
    """How well a downscaled HR matches its LR, in luma. ~0.99 for a true pair;
    a pair from a different frame of the same shot lands near 0.5, and static
    scenes are the only ones that survive by accident."""
    from PIL import Image
    a = (0.299 * lr[..., 0] + 0.587 * lr[..., 1] + 0.114 * lr[..., 2]).astype(np.float64)
    small = Image.fromarray(hr).convert("L").resize((lr.shape[1], lr.shape[0]), Image.LANCZOS)
    b = np.asarray(small, dtype=np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0            # a flat patch carries no signal either way
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((x * y).mean() / np.sqrt((x * x).mean() * (y * y).mean()))


def report_composition(corpus):
    """Prints what the corpus is made of before a minute is spent training on it.

    Every corpus fault this project has hit was distributional and passed every
    structural check: pairs at 2x when 4x was needed, HR and LR from different
    frames, 100% animation, then 95% live action, then stale pairs from a
    superseded run blended in at 6%. Counts and sizes were right every time.
    Shape is the thing that goes wrong, so shape is what gets printed.
    """
    path = os.path.join(corpus, "manifest.jsonl")
    if not os.path.exists(path):
        print("no manifest - composition unknown"); return
    records = []
    for line in open(path):
        try: records.append(json.loads(line))
        except ValueError: continue
    if not records:
        print("empty manifest - composition unknown"); return

    def tally(label, values):
        counts = {}
        for v in values: counts[v] = counts.get(v, 0) + 1
        total = sum(counts.values()) or 1
        parts = ", ".join(f"{k} {100*v//total}%" for k, v in
                          sorted(counts.items(), key=lambda kv: -kv[1])[:8])
        print(f"  {label:12s} {parts}")

    print(f"corpus composition ({len(records)} pairs)")
    tally("content", ["live" if "netflix" in r.get("source", "") else "animation"
                      for r in records])
    tally("tier", [f"{r['lr']['w']}x{r['lr']['h']}" for r in records if "lr" in r])
    tally("codec", [r["lr"]["codec"] for r in records if "lr" in r])
    tally("bitrate", ["<150k" if r["lr"]["bitrate"] < 150_000 else
                      "150-400k" if r["lr"]["bitrate"] < 400_000 else ">400k"
                      for r in records if "lr" in r])
    tally("gop", [str(r["lr"]["gop"]) for r in records if "lr" in r])
    sources = {os.path.basename(r.get("source", "?")) for r in records}
    print(f"  {'sources':12s} {len(sources)} distinct")


def build_bank(corpus, out, per_pair=8, seed=0, min_correlation=0.85):
    """Pre-cuts training patches into a memory-mapped array.

    Decoding a 2560x1440 PNG per sample costs more than the training step does.
    Cutting patches once and memory-mapping them turns the input pipeline from
    the bottleneck into a rounding error.
    """
    from PIL import Image
    pairs = sorted(os.listdir(os.path.join(corpus, "lr")))
    if not pairs:
        raise SystemExit(f"no pairs in {corpus}/lr")
    report_composition(corpus)
    rng = random.Random(seed)
    total = len(pairs) * per_pair
    os.makedirs(out, exist_ok=True)
    lr_bank = np.lib.format.open_memmap(
        os.path.join(out, "lr.npy"), mode="w+", dtype=np.uint8,
        shape=(total, LR_PATCH, LR_PATCH, 3))
    hr_bank = np.lib.format.open_memmap(
        os.path.join(out, "hr.npy"), mode="w+", dtype=np.uint8,
        shape=(total, HR_PATCH, HR_PATCH, 3))

    written = 0
    rejected = []
    for index, name in enumerate(pairs):
        lr_path = os.path.join(corpus, "lr", name)
        hr_path = os.path.join(corpus, "hr", name)
        if not os.path.exists(hr_path):
            continue
        lr = np.asarray(Image.open(lr_path).convert("RGB"))
        hr = np.asarray(Image.open(hr_path).convert("RGB"))
        # The corpus is only useful if HR really is 4x LR. A 2x corpus trains
        # the wrong mapping and looks fine until the model ships, so check
        # rather than trust.
        if hr.shape[0] != lr.shape[0] * SCALE or hr.shape[1] != lr.shape[1] * SCALE:
            raise SystemExit(
                f"{name}: hr {hr.shape[1]}x{hr.shape[0]} is not {SCALE}x "
                f"lr {lr.shape[1]}x{lr.shape[0]}")
        if lr.shape[0] < LR_PATCH or lr.shape[1] < LR_PATCH:
            continue
        # Matching sizes prove nothing about matching content. The first corpus
        # passed the size check and was still 57% junk: HR and LR were decoded
        # by separate seeks and landed on different frames, so more than half
        # the pairs were the same shot at a different moment. Training on those
        # teaches the model to invent motion, and the only symptom was a
        # validation PSNR that looked merely disappointing.
        #
        # A real pair survives a round trip: downscale HR and it should look
        # like LR. Anything that does not correlate is dropped here rather than
        # quietly averaged into the weights.
        if pair_correlation(lr, hr) < min_correlation:
            rejected.append(name)
            continue
        for _ in range(per_pair):
            y = rng.randrange(0, lr.shape[0] - LR_PATCH + 1)
            x = rng.randrange(0, lr.shape[1] - LR_PATCH + 1)
            lr_bank[written] = lr[y:y + LR_PATCH, x:x + LR_PATCH]
            hr_bank[written] = hr[y * SCALE:y * SCALE + HR_PATCH,
                                  x * SCALE:x * SCALE + HR_PATCH]
            written += 1
        if index % 200 == 0:
            print(f"  {index}/{len(pairs)} pairs, {written} patches", flush=True)

    if rejected:
        print(f"rejected {len(rejected)} of {len(pairs)} pairs as mismatched "
              f"(first few: {', '.join(rejected[:4])})")
        # A tripwire for a broken corpus, not a quality bar. The first corpus
        # was 80% wrong; a fifth is the line between "some frames fell through"
        # and "the generator is broken again". Rejected pairs never train
        # either way, so what gets through is verified rather than assumed.
        if len(rejected) > len(pairs) // 5:
            raise SystemExit(
                f"{100*len(rejected)/len(pairs):.0f}% of the corpus is mismatched - "
                "fix the generator rather than training on what is left")
    lr_bank.flush(); hr_bank.flush()
    with open(os.path.join(out, "bank.json"), "w") as fh:
        json.dump({"count": written, "lr_patch": LR_PATCH, "scale": SCALE}, fh)
    print(f"wrote {written} patches to {out}")


def load_bank(out):
    with open(os.path.join(out, "bank.json")) as fh:
        meta = json.load(fh)
    lr = np.load(os.path.join(out, "lr.npy"), mmap_mode="r")
    hr = np.load(os.path.join(out, "hr.npy"), mmap_mode="r")
    return lr[:meta["count"]], hr[:meta["count"]]


# ---- metrics ----------------------------------------------------------------

def fine_band_correlation(a, b):
    """The metric this project judges upscalers by: correlation with the truth
    in the finest spatial band. Band *energy* cannot tell recovered detail from
    invented detail and has misled this project twice; correlation can."""
    def luma(t):
        return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)

    def blur(t, sigma):
        radius = max(1, int(sigma * 3))
        grid = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
        kernel = torch.exp(-(grid ** 2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        t = F.conv2d(t, kernel.view(1, 1, 1, -1), padding=(0, radius))
        return F.conv2d(t, kernel.view(1, 1, -1, 1), padding=(radius, 0))

    fa, fb = luma(a), luma(b)
    fa, fb = fa - blur(fa, 1.0), fb - blur(fb, 1.0)
    fa, fb = fa - fa.mean(), fb - fb.mean()
    denominator = torch.sqrt((fa * fa).mean() * (fb * fb).mean())
    return float((fa * fb).mean() / denominator) if denominator > 0 else 0.0


# ---- training ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=".build/corpus4")
    parser.add_argument("--bank-dir", default=".build/bank")
    parser.add_argument("--bank", action="store_true", help="build the patch bank and stop")
    parser.add_argument("--per-pair", type=int, default=8)
    parser.add_argument("--min-correlation", type=float, default=0.85,
                        help="reject pairs whose HR and LR are not the same frame")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="Model/weights")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    if args.bank:
        build_bank(args.corpus, args.bank_dir, args.per_pair,
                   min_correlation=args.min_correlation)
        return

    lr_bank, hr_bank = load_bank(args.bank_dir)
    count = len(lr_bank)
    # A held-out tail, taken by index so the same patches are never trained on.
    validation = max(256, count // 40)
    train_count = count - validation
    print(f"{train_count} training patches, {validation} held out")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Unshuffled(args.channels).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"ch{args.channels}u: {parameters/1000:.0f}K parameters, training on {device}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps, eta_min=args.lr / 100)
    start = 0
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"]); optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"]); start = state["step"]
        print(f"resumed from {args.resume} at step {start}")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    def batch(indices):
        lr = torch.from_numpy(np.ascontiguousarray(lr_bank[indices])).to(device)
        hr = torch.from_numpy(np.ascontiguousarray(hr_bank[indices])).to(device)
        lr = lr.permute(0, 3, 1, 2).float() / 255
        hr = hr.permute(0, 3, 1, 2).float() / 255
        return lr, hr

    began = time.time()
    running = 0.0
    for step in range(start, args.steps):
        indices = np.sort(rng.integers(0, train_count, size=args.batch))
        lr, hr = batch(indices)
        # Eight-way dihedral augmentation, applied to both halves identically.
        if rng.random() < 0.5:
            lr, hr = torch.flip(lr, [3]), torch.flip(hr, [3])
        if rng.random() < 0.5:
            lr, hr = torch.flip(lr, [2]), torch.flip(hr, [2])
        if rng.random() < 0.5:
            lr, hr = lr.transpose(2, 3), hr.transpose(2, 3)

        output = model(lr)
        loss = charbonnier(output, hr)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        # SPAN's re-parameterised convs can spike early; clipping costs nothing
        # and removes a class of silent divergence.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step(); schedule.step()
        running += float(loss)

        if (step + 1) % 200 == 0:
            rate = (step + 1 - start) / max(time.time() - began, 1e-6)
            print(f"  step {step+1}/{args.steps}  loss {running/200:.5f}  "
                  f"lr {schedule.get_last_lr()[0]:.2e}  {rate:.1f} it/s", flush=True)
            running = 0.0

        if (step + 1) % 2000 == 0 or step + 1 == args.steps:
            model.eval()
            psnr_total, fine_total, seen = 0.0, 0.0, 0
            with torch.no_grad():
                for offset in range(0, validation, args.batch):
                    indices = np.arange(train_count + offset,
                                        min(train_count + offset + args.batch, count))
                    if not len(indices):
                        break
                    lr, hr = batch(indices)
                    out = model(lr).clamp(0, 1)
                    mse = float(((out - hr) ** 2).mean())
                    psnr_total += 10 * math.log10(1.0 / max(mse, 1e-9)) * len(indices)
                    fine_total += fine_band_correlation(out, hr) * len(indices)
                    seen += len(indices)
            print(f"  ── step {step+1}: PSNR {psnr_total/seen:.2f} dB  "
                  f"fine {fine_total/seen:.4f}", flush=True)
            model.train()
            torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                        "schedule": schedule.state_dict(), "step": step + 1,
                        "channels": args.channels},
                       os.path.join(args.out, f"span_ch{args.channels}u.pth"))

    print(f"done in {(time.time()-began)/60:.1f} min → {args.out}/span_ch{args.channels}u.pth")


if __name__ == "__main__":
    main()
