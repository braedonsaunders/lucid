#!/usr/bin/env python3
"""Inject only a neural upscaler's extra luminance detail into a stable base.

This deliberately rejects neural chroma and low-frequency tone changes. It is
an offline probe for the live hybrid pipeline, not a runtime dependency.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageFilter


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("neural")
    parser.add_argument("output")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--radius", type=float, default=1.5)
    args = parser.parse_args()

    base_image = Image.open(args.base).convert("RGB")
    neural_image = Image.open(args.neural).convert("RGB")
    if neural_image.size != base_image.size:
        raise SystemExit("base and neural images must have identical dimensions")

    base = np.asarray(base_image, dtype=np.float32) / 255.0
    neural = np.asarray(neural_image, dtype=np.float32) / 255.0
    base_y = luma(base)
    neural_y = luma(neural)

    base_blur = np.asarray(
        Image.fromarray(np.uint8(np.clip(base_y * 255.0, 0, 255)), "L").filter(
            ImageFilter.GaussianBlur(args.radius)
        ),
        dtype=np.float32,
    ) / 255.0
    neural_blur = np.asarray(
        Image.fromarray(np.uint8(np.clip(neural_y * 255.0, 0, 255)), "L").filter(
            ImageFilter.GaussianBlur(args.radius)
        ),
        dtype=np.float32,
    ) / 255.0

    # Difference of high-frequency residuals: the model may contribute detail,
    # but it cannot alter hue, saturation, exposure, or broad gradients.
    detail = (neural_y - neural_blur) - (base_y - base_blur)
    target_y = np.clip(base_y + args.strength * detail, 0.0, 1.0)
    scale = target_y / np.maximum(base_y, 1e-4)
    output = np.clip(base * scale[..., None], 0.0, 1.0)

    Image.fromarray(np.uint8(output * 255.0), "RGB").save(args.output)


if __name__ == "__main__":
    main()
