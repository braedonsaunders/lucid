#!/usr/bin/env python3
"""Adversarially fine-tunes a trained upscaler so it produces detail, not blur.

The measurement that made this necessary: on footage it has never seen, ch32u
recovers 0.32 of the fine-band energy the ground truth has. Lanczos gets 0.24.
So the model is a third of the way there and the rest of the picture is
smoothness, which is exactly what an L1 objective asks for - L1's optimum for an
uncertain pixel is the mean of everything it could be, and the mean of plausible
textures is grey.

A discriminator changes the question from "is this pixel right" to "is this
texture real". That is the only known way to get the energy up, and it is what
every super-resolution model people call impressive is trained with.

Three deliberate choices:

  Train on VGG features, evaluate on LPIPS and DISTS. Both are deep-feature
  perceptual metrics and it would be trivial to optimise the one we report.
  Keeping them separate is the difference between a number that improves and a
  picture that improves.

  The generator is untouched. Only the training objective changes, so the graph
  still converts to Core ML exactly as before and the Neural Engine ladder is
  unaffected. Nothing about this costs a millisecond at playback.

  Start from the L1 checkpoint rather than from scratch. Adversarial training
  from random weights is unstable and slow; from a converged L1 model it is a
  refinement, which is the standard recipe and the cheap one.

  .venv-convert/bin/python Tools/finetune_gan.py \\
      --weights Model/weights/span_ch32u.pth --steps 20000
"""
import argparse, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_span import (Unshuffled, load_bank, charbonnier, pick_device, temporal_loss,  # noqa: E402
                        LR_PATCH, HR_PATCH)


class UNetDiscriminator(nn.Module):
    """Real-ESRGAN's discriminator shape: a small U-Net with spectral norm,
    producing a per-pixel real/fake map rather than one number for the image.

    Per-pixel matters here. A whole-image verdict lets the generator pass by
    getting most of the frame right; a per-pixel one makes it answer for every
    patch of texture, which is where the missing energy actually lives."""

    def __init__(self, channels=3, width=64):
        super().__init__()
        sn = spectral_norm
        self.conv0 = nn.Conv2d(channels, width, 3, 1, 1)
        self.down1 = sn(nn.Conv2d(width, width * 2, 4, 2, 1, bias=False))
        self.down2 = sn(nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False))
        self.down3 = sn(nn.Conv2d(width * 4, width * 8, 4, 2, 1, bias=False))
        self.up3 = sn(nn.Conv2d(width * 8, width * 4, 3, 1, 1, bias=False))
        self.up2 = sn(nn.Conv2d(width * 4, width * 2, 3, 1, 1, bias=False))
        self.up1 = sn(nn.Conv2d(width * 2, width, 3, 1, 1, bias=False))
        self.out0 = sn(nn.Conv2d(width, width, 3, 1, 1, bias=False))
        self.out1 = nn.Conv2d(width, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2, inplace=True)
        x0 = act(self.conv0(x))
        x1 = act(self.down1(x0))
        x2 = act(self.down2(x1))
        x3 = act(self.down3(x2))
        u3 = act(self.up3(F.interpolate(x3, scale_factor=2, mode="nearest"))) + x2
        u2 = act(self.up2(F.interpolate(u3, scale_factor=2, mode="nearest"))) + x1
        u1 = act(self.up1(F.interpolate(u2, scale_factor=2, mode="nearest"))) + x0
        return self.out1(act(self.out0(u1)))


class VGGFeatures(nn.Module):
    """Perceptual loss on VGG19 features, at the layers and weights Real-ESRGAN
    uses. Deliberately NOT LPIPS: LPIPS is what the result is judged by, and a
    model trained on its own scoreboard tells you nothing."""

    LAYERS = {"2": 0.1, "7": 0.1, "16": 1.0, "25": 1.0, "34": 1.0}

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = features[:35].eval().to(device)
        for p in self.slices.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, a, b):
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        loss = 0.0
        for index, layer in enumerate(self.slices):
            a, b = layer(a), layer(b)
            weight = self.LAYERS.get(str(index))
            if weight:
                loss = loss + weight * F.l1_loss(a, b)
        return loss


def detail_energy(x):
    """Fine-band energy: the number the L1 model is short on. Reported every
    interval so the thing this run exists to move is visible while it moves."""
    grey = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    blur = F.avg_pool2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return float(torch.sqrt(((grey - blur) ** 2).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="Model/weights/span_ch32u.pth")
    parser.add_argument("--bank-dir", default=".build/bank")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=32,
                        help="LR patch size for the adversarial stage. The bank "
                             "holds 64, but the discriminator and VGG run at the "
                             "HR size, so 64 means 256x256 through both of them - "
                             "measured at 3.11 s/step against 0.66 at 32. A "
                             "discriminator judges texture, and texture is local, "
                             "so the smaller crop costs nothing it needs.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pixel", type=float, default=1.0)
    parser.add_argument("--perceptual", type=float, default=1.0)
    parser.add_argument("--adversarial", type=float, default=0.1)
    parser.add_argument("--temporal", type=float, default=0.0,
                        help="weight on the temporal consistency loss. Needs a "
                             "bank built from a temporal corpus. This is where "
                             "it matters most: a discriminator judges one frame "
                             "at a time, so it rewards texture invented "
                             "independently on every frame")
    parser.add_argument("--out", default="Model/weights")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    lr_bank, hr_bank, frames = load_bank(args.bank_dir)
    count = len(lr_bank)
    validation = max(256, count // 40)
    train_count = count - validation

    # The temporal term reads a history frame out of the bank; the generator
    # still sees one frame, so what ships stays single-frame.
    if args.temporal > 0 and frames < 2:
        raise SystemExit("--temporal needs a bank built from a temporal corpus "
                         "(one with lr_t1 beside lr)")
    input_frames = 1 if args.temporal > 0 else frames

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    generator = Unshuffled(state.get("channels", 32), frames=input_frames,
                           version=state.get("version", 1)).to(device)
    generator.load_state_dict(state["model"])
    discriminator = UNetDiscriminator().to(device)
    perceptual = VGGFeatures(device)

    print(f"generator from {args.weights} at step {state.get('step')}, "
          f"{sum(p.numel() for p in generator.parameters())/1000:.0f}K parameters")
    print(f"discriminator {sum(p.numel() for p in discriminator.parameters())/1e6:.1f}M "
          f"(training only - it never ships and never converts)")
    terms = (f"{args.pixel} x L1 + {args.perceptual} x VGG + "
             f"{args.adversarial} x adversarial")
    if args.temporal > 0:
        terms += f" + {args.temporal} x temporal"
    print(f"loss: {terms}", flush=True)
    print(f"batch {args.batch} at {args.crop}x{args.crop} LR "
          f"({args.crop*4}x{args.crop*4} through the discriminator)", flush=True)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.steps, eta_min=args.lr / 20)
    rng = np.random.default_rng(0)
    bce = nn.BCEWithLogitsLoss()

    def batch(indices):
        lr = torch.from_numpy(np.ascontiguousarray(lr_bank[indices])).to(device)
        hr = torch.from_numpy(np.ascontiguousarray(hr_bank[indices])).to(device)
        lr = lr.permute(0, 3, 1, 2).float() / 255
        hr = hr.permute(0, 3, 1, 2).float() / 255
        if args.crop and args.crop < lr.shape[-1]:
            top = int(rng.integers(0, lr.shape[-2] - args.crop + 1))
            left = int(rng.integers(0, lr.shape[-1] - args.crop + 1))
            lr = lr[:, :, top:top + args.crop, left:left + args.crop]
            hr = hr[:, :, top * 4:(top + args.crop) * 4,
                    left * 4:(left + args.crop) * 4]
        return lr, hr

    # Before the first step, not at the first save: a run that cannot write its
    # result should fail in a second rather than an hour.
    os.makedirs(args.out, exist_ok=True)

    began = time.time()
    running = {"pixel": 0.0, "perc": 0.0, "adv": 0.0, "d": 0.0, "temporal": 0.0}
    for step in range(args.steps):
        indices = np.sort(rng.integers(0, train_count, size=args.batch))
        lr, hr = batch(indices)
        if rng.random() < 0.5:
            lr, hr = torch.flip(lr, [3]), torch.flip(hr, [3])

        # ---- generator ----
        for p in discriminator.parameters():
            p.requires_grad = False
        current = lr[:, -3 * input_frames:]
        fake = generator(current)
        pixel = charbonnier(fake, hr)
        perc = perceptual(fake.clamp(0, 1), hr)
        steadiness = torch.zeros((), device=device)
        if args.temporal > 0:
            previous = lr[:, -6:-3]
            steadiness = temporal_loss(fake, generator(previous), current, previous)
        # One discriminator pass, not two. Written twice this ran the whole
        # discriminator a second time just to get a tensor of ones the right
        # shape, which is the most expensive way imaginable to call ones_like.
        fake_logits = discriminator(fake)
        adversarial = bce(fake_logits, torch.ones_like(fake_logits))
        loss_g = (args.pixel * pixel + args.perceptual * perc
                  + args.adversarial * adversarial + args.temporal * steadiness)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step(); sched_g.step()

        # ---- discriminator ----
        for p in discriminator.parameters():
            p.requires_grad = True
        opt_d.zero_grad(set_to_none=True)
        real_pred = discriminator(hr)
        loss_real = bce(real_pred, torch.ones_like(real_pred))
        loss_real.backward()
        fake_pred = discriminator(fake.detach())
        loss_fake = bce(fake_pred, torch.zeros_like(fake_pred))
        loss_fake.backward()
        opt_d.step()

        running["pixel"] += float(pixel); running["perc"] += float(perc)
        running["adv"] += float(adversarial); running["d"] += float(loss_real + loss_fake)
        running["temporal"] += float(steadiness)

        if (step + 1) % 200 == 0:
            rate = (step + 1) / max(time.time() - began, 1e-6)
            print(f"  step {step+1}/{args.steps}  L1 {running['pixel']/200:.4f}  "
                  f"VGG {running['perc']/200:.4f}  adv {running['adv']/200:.3f}  "
                  f"D {running['d']/200:.3f}"
                  + (f"  temporal {running['temporal']/200:.4f}" if args.temporal > 0 else "")
                  + f"  {rate:.1f} it/s", flush=True)
            running = {k: 0.0 for k in running}

        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            generator.eval()
            with torch.no_grad():
                indices = np.arange(train_count, min(train_count + args.batch, count))
                lr, hr = batch(indices)
                out = generator(lr[:, -3 * input_frames:]).clamp(0, 1)
                mse = float(((out - hr) ** 2).mean())
                # The ratio this whole run exists to move.
                ratio = detail_energy(out) / max(detail_energy(hr), 1e-9)
            print(f"  ── step {step+1}: PSNR {10*math.log10(1/max(mse,1e-9)):.2f} dB  "
                  f"detail {ratio:.3f} of truth", flush=True)
            generator.train()
            name = (os.path.basename(args.weights).replace(".pth", "")
                    + ("_gantc" if args.temporal > 0 else "_gan") + ".pth")
            torch.save({"model": generator.state_dict(), "step": step + 1,
                        "channels": state.get("channels", 32), "frames": input_frames,
                        "version": state.get("version", 1)},
                       os.path.join(args.out, name))

    print(f"done in {(time.time()-began)/60:.1f} min")


if __name__ == "__main__":
    main()
