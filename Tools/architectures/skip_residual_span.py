"""Single-frame 2x SR with a fixed low-frequency bypass around the trunk.

The trunk inside SPAN synthesizes the whole output, including the DC and
low-frequency content that is already present in the input. That spends
capacity on the identity instead of on the high-frequency detail the
perceptual metrics actually score. This wrapper adds a fixed bilinear 2x
upsample of the input around the trunk, so the trunk learns only the
residual.

Placement matters. The training wrapper (Tools/train_span.py Unshuffled)
folds 2x2 LR pixels into channels with PixelUnshuffle(2) and then upscales
x4 inside the trunk, so the net mapping is LR -> 2x HR. The skip therefore
taps the pre-unshuffle LR tensor and upsamples it x2 directly: same output
geometry as the trunk, no phase bookkeeping, and a naive shape mismatch is
impossible because interpolate derives the output shape from the input.

The skip carries no parameters, so a skip checkpoint is structurally
identical to a control checkpoint: same state_dict keys, same parameter
count. Only the values differ, because the trunk head starts at zero (see
zero_trunk_head). The control path (plain Unshuffled) is untouched and
bit-for-bit reproducible.
"""
import torch
from torch import nn
from torch.nn import functional as F

from train_span import Unshuffled


class SkipUnshuffled(Unshuffled):
    """Unshuffled whose trunk output is a residual over a bilinear bypass.

    At construction the trunk head must be zeroed (zero_trunk_head) so the
    model starts as an exact bilinear upsample and learns the residual from
    there. Without that the model starts at trunk output plus bilinear,
    which overshoots and defeats the point.
    """

    def forward(self, x):
        """x is (B, 3, H, W) LR; returns (B, 3, 2H, 2W)."""
        if x.ndim != 4 or x.shape[1] != 3 or any(d % 2 for d in x.shape[-2:]):
            raise ValueError("even BCHW single-image input required")
        trunk = self.core(self.unshuffle(x))
        bypass = F.interpolate(x, scale_factor=2, mode="bilinear",
                               align_corners=False)
        if bypass.shape != trunk.shape:
            raise ValueError(f"bypass {tuple(bypass.shape)} disagrees with "
                             f"trunk {tuple(trunk.shape)}")
        return trunk + bypass


def zero_trunk_head(model):
    """Zero the last trunk convolution so the trunk starts at exactly zero.

    The head is upsampler[0], the plain convolution feeding the final pixel
    shuffle. Everything upstream is multiplied by zero weights, so the trunk
    contributes nothing regardless of the body initialization, and the full
    model starts as an exact bilinear upsample. The body keeps its (trained)
    weights, so features are already useful on the first residual update.
    """
    head = model.core.upsampler[0]
    if not isinstance(head, nn.Conv2d):
        raise ValueError("trunk head must be the plain upsampler convolution")
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()
    return model


def from_unshuffled(model):
    """Rebuild a loaded Unshuffled as a skip model with a zeroed trunk head.

    State dict keys are identical (the skip has no parameters), so the
    trained weights transfer exactly and only the head is reset.
    """
    skip = SkipUnshuffled(model.core.conv_1.sk.out_channels,
                          scale=model.core.upsampler[1].upscale_factor // 2,
                          frames=model.frames, version=model.version)
    skip.load_state_dict(model.state_dict(), strict=True)
    return zero_trunk_head(skip)
