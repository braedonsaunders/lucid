#!/usr/bin/env python3
"""Converts a trained unshuffled checkpoint to Core ML at every ladder size.

The model is one set of weights; Core ML wants one compiled graph per input
shape, because a fixed shape is what keeps the whole thing on the Neural Engine.
`ct.EnumeratedShapes` exists, but only the default shape reliably stays on ANE,
so the ladder is converted as separate packages.

Every width here is a multiple of 16. That is not tidiness: an unaligned width
costs the ANE an entire extra tile pass, measured at 23.0 ms against 13.6 ms for
the same picture. The two real streaming sizes that are not aligned - 426 and
854 wide - are converted at the next multiple of 16 and the frame is stretched
that last 1.4% on the way in, which the page undoes when it draws the result
into the video box.

  .venv-convert/bin/python Tools/convert_trained.py --weights Model/weights/span_ch32u.pth
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_span import Unshuffled  # noqa: E402

# The ladder, matching LR_TIERS in make_degradation.py and the target window of
# "below 720p" that Edge uses.
LADDER = [(256, 144), (320, 180), (432, 240), (480, 270), (640, 360), (864, 480)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="Model/weights/span_ch32u.pth")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    channels = state.get("channels", args.channels)
    frames = state.get("frames", 1)
    model = Unshuffled(channels, frames=frames).eval()
    model.load_state_dict(state["model"] if "model" in state else state)
    step = state.get("step", "?")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"ch{channels}u{'t' if frames > 1 else ''} from {args.weights} at "
          f"step {step}: {parameters/1000:.0f}K parameters, {frames} input frame(s)")

    # convert_span's converter carries the fp16 policy and the image in/out
    # wrapper; reuse it rather than restating the conversion rules here.
    import runpy
    ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "convert_span.py"),
                        run_name="not_main")
    globals_for_convert = dict(ns)
    exec(compile(open(os.path.join(os.path.dirname(__file__), "convert_span.py")).read()
                 .split("if __name__")[0], "convert_span", "exec"), globals_for_convert)
    globals_for_convert["CHANNELS"] = f"{channels}u{'t' if frames > 1 else ''}"
    globals_for_convert["SCALE"] = 4

    for width, height in LADDER:
        try:
            path = globals_for_convert["convert"](model, width, height, None, args.suffix)
            print(f"  wrote {path}")
        except Exception as error:
            print(f"  FAILED {width}x{height}: {repr(error)[:200]}")


if __name__ == "__main__":
    main()
