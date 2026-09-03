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


# SPAN scales its input by img_range=255 internally, which pushes activations
# out of fp16 range: converted at FLOAT16 the output mean comes back 12.3 where
# torch says 92.0. FLOAT32 reproduces torch exactly. Precision here is therefore
# a correctness setting, not a speed one.
class ImageRange(torch.nn.Module):
    """SPAN takes and returns 0-1, but a Core ML image output has to be 0-255 or
    it arrives essentially black. This puts the scaling inside the graph so the
    conversion stays a single image-in, image-out model."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return torch.clamp(self.inner(x) * 255.0, 0.0, 255.0)


def convert(model, width, height, precision=ct.precision.FLOAT32, suffix=""):
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
