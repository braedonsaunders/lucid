#!/usr/bin/env python3
"""Converts SPAN to Core ML at the input sizes Lucid actually sees.

Why 2x: in AIM 2024's challenge on AV1-compressed 540p->4K, no entry beat a
plain Lanczos anchor on PSNR. Learned super-resolution recovers real detail at
2x on compressed sources and synthesises plausible detail at 4x. So the learned
pass runs at 2x and Apple's scaler carries the rest.

Why fixed shapes: coremltools RangeDim keeps only the default shape on the
Neural Engine and pushes the others to GPU or CPU. EnumeratedShapes specialises
every listed shape at compile time, so they all stay on the ANE.

  .venv-convert/bin/python Tools/convert_span.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import importlib.util
import torch
import coremltools as ct

def load_arch():
    """Loads span_arch.py on its own. Importing it through the basicsr package
    pulls in the whole training stack - OpenCV and the rest - for a file that
    only needs torch. The one package symbol it wants is a registry decorator,
    so a no-op stands in for it."""
    root = os.path.join(os.path.dirname(__file__), "..", "Model", "SPAN")
    registry = type(sys)("basicsr.utils.registry")
    registry.ARCH_REGISTRY = type("R", (), {"register": staticmethod(lambda *a, **k: (lambda c: c))})()
    for name, module in (("basicsr", type(sys)("basicsr")),
                         ("basicsr.utils", type(sys)("basicsr.utils")),
                         ("basicsr.utils.registry", registry)):
        sys.modules.setdefault(name, module)
    spec = importlib.util.spec_from_file_location(
        "span_arch", os.path.join(root, "basicsr", "archs", "span_arch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPAN

SPAN = load_arch()

# The sizes a browser video actually arrives at, and whether the parameter cost
# fits the 8 ms budget there (measured 3.5 TMAC/s on this M4 Pro).
SIZES = [(256, 144), (426, 240), (480, 270), (640, 360)]
CHANNELS = 48
SCALE = 2
WEIGHTS = "Model/SPAN/weights/spanx2_ch48.pth"


def prepare(model, state, conv_class):
    """Loads a checkpoint and makes the fused weights correct for either kind.

    SPAN's Conv3XC recomputes its fused `eval_conv` from the unfused branches on
    every eval call. A full checkpoint ships those branches, so that recompute is
    right and must happen. A `slim` checkpoint ships only the fused weights, so
    the recompute would overwrite them with values derived from branches that
    were never in the file - which is silent, and produces a model that looks
    plausible and scores well while being wrong."""
    model.load_state_dict(state, strict=False)
    model.eval()
    has_unfused = any(".conv.0.weight" in k for k in state)
    if has_unfused:
        # Let the fusion run once so eval_conv is built from real weights, then
        # freeze it so tracing cannot repeat the work.
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    conv_class.update_params = lambda self: None
    return model


def load():
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=CHANNELS, upscale=SCALE)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = state.get("params", state.get("params_ema", state))
    model.load_state_dict(state, strict=True)
    # eval() is load-bearing here: SPAN's Conv3XC blocks collapse into a plain
    # 3x3 convolution only in eval mode, which is what makes the exported graph
    # ANE-shaped in the first place.
    model.eval()
    return model


# Converted at plain FLOAT16 some SPAN variants come back far too dark - ch28
# gives an output mean of 17.7 where torch says 92.0 - while others are fine.
# It is not activation overflow: the largest activation in the network is 54
# against fp16's limit of 65504. Bisecting by op type, holding any one of `mul`,
# `conv` or `pixel_shuffle` in fp32 fixes it. `pixel_shuffle` is the one to
# choose: there is exactly one in the graph and it only moves data around.
#
# The failure is silent and it scores well, because correlation is
# scale-invariant - a near-black output still measured above the Lanczos anchor.
# Always check the output level, never just the metric.
def selective_fp16():
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type != "mul")



class ImageRange(torch.nn.Module):
    """SPAN takes and returns 0-1, but a Core ML image output has to be 0-255 or
    it arrives essentially black. This puts the scaling inside the graph so the
    conversion stays a single image-in, image-out model."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return torch.clamp(self.inner(x) * 255.0, 0.0, 255.0)


def convert(model, width, height, precision=None, suffix=""):
    precision = precision or selective_fp16()
    wrapped = ImageRange(model).eval()
    example = torch.rand(1, 3, height, width)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    # Image in, image out: Core ML then owns the colour conversion and can keep
    # it next to the compute instead of us round-tripping through float arrays.
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=(1, 3, height, width),
                             color_layout=ct.colorlayout.RGB, scale=1 / 255.0)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )
    path = f"Model/SPAN_x{SCALE}_ch{CHANNELS}_{width}x{height}{suffix}.mlpackage"
    mlmodel.save(path)
    return path


if __name__ == "__main__":
    model = load()
    params = sum(p.numel() for p in model.parameters())
    print(f"SPAN ch{CHANNELS} x{SCALE}: {params/1000:.0f}K parameters")
    print(f"{'size':>12}{'GMAC':>9}{'predicted ms':>14}")
    for width, height in SIZES:
        macs = params * width * height / 1e9
        print(f"{width}x{height:<7}{macs:>9.1f}{macs/3.5:>14.1f}")
    print()
    for width, height in SIZES:
        try:
            path = convert(model, width, height)
            print(f"  wrote {path}")
        except Exception as error:
            print(f"  {width}x{height} failed: {error}")
