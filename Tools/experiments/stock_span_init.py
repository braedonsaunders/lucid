"""Explicit loader for the stock pretrained SPAN 2x checkpoint.

The vendor checkpoint (spanx2_ch48, 2,221,140 params) is NOT disguised as a
Lucid Unshuffled checkpoint: it has a full-LR trunk (3 input channels, no
pixel unshuffle), a factor-2 core upsampler, and the original input
range/mean (img_range 255, ImageNet mean). Loading it through
eval_checkpoint.load into Unshuffled would be a meaningless run; this
module keeps the representation explicit and refuses anything else.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from architectures.span_arch import Conv3XC, SPAN  # noqa: E402

STOCK_SHA256 = "561fd5cf419a23d4de1231ce258180f61aee4aa8caa1aaaa783769c7301847bc"
STOCK_PARAMS = 2221140
STOCK_CHANNELS = 48
STOCK_UPSCALE = 2
STOCK_IMG_RANGE = 255.0
STOCK_RGB_MEAN = (0.4488, 0.4371, 0.4040)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_stock_span2x(path, device="cpu"):
    """Loads stock spanx2_ch48 weights into their native SPAN(3ch, x2) geometry.

    Returns (model, report). The report pins the checkpoint hash, parameter
    count, geometry, and range/mean so a training receipt can cite it.
    """
    if digest(path) != STOCK_SHA256:
        raise ValueError("stock checkpoint sha256 differs from pinned spanx2_ch48")
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=STOCK_CHANNELS,
                 upscale=STOCK_UPSCALE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("params", state.get("params_ema", state))
    # Full BasicSR-style file ships unfused Conv3XC branches; refuse anything
    # that is not that exact representation.
    if not any(".conv.0.weight" in k for k in state):
        raise ValueError("stock checkpoint lacks unfused Conv3XC branches")
    model.load_state_dict(state, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != STOCK_PARAMS:
        raise ValueError(f"stock parameter count {count} != pinned {STOCK_PARAMS}")
    if hasattr(model, "core") or hasattr(model, "unshuffle"):
        raise ValueError("stock model must not carry an Unshuffled wrapper")
    if model.upsampler[1].upscale_factor != STOCK_UPSCALE:
        raise ValueError("stock upsampler is not factor 2")
    if model.conv_1.eval_conv.in_channels != 3:
        raise ValueError("stock trunk is not full-LR (conv_1 must take 3 channels)")
    mean = model.mean.flatten().tolist()
    if model.img_range != STOCK_IMG_RANGE or any(abs(a - b) > 1e-6 for a, b in zip(mean, STOCK_RGB_MEAN)):
        raise ValueError("stock input range/mean differ from the original architecture")
    model.eval()
    # Let the Conv3XC fusion run once from real weights, then freeze it so an
    # eval-mode forward cannot silently rebuild eval_conv from missing branches.
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for module in model.modules():
        if isinstance(module, Conv3XC):
            module.update_params()
            module.__dict__["update_params"] = lambda: None
    model.eval().to(device)
    report = {"checkpoint_sha256": STOCK_SHA256, "parameters": count,
              "channels": STOCK_CHANNELS, "upscale": STOCK_UPSCALE,
              "trunk": "full_lr", "in_channels": 3,
              "img_range": STOCK_IMG_RANGE, "rgb_mean": list(STOCK_RGB_MEAN)}
    return model, report


def stock_output_scale_note():
    return ("stock SPAN emits raw 0-1-range values (clamped *255 at export); "
            "it is a fine-tune init only, never a drop-in for the 0-1 "
            "unit-range Unshuffled trunk")
