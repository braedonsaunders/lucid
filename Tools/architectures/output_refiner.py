"""Output-resolution refiner head after the 2x SPAN pixelshuffle.

All current model capacity sits before the upsample: the trunk runs at
quarter area (pixel-unshuffled input) and a single convolution plus
pixelshuffle produces the output. Blocking and mosquito artifacts take
their visible shape at output resolution, which the trunk never sees the
way the evaluator does. This module appends two small 3x3 convolutions
(8 or 16 channels) operating after the pixelshuffle, as a residual on the
reconstruction.

An output-resolution convolution costs 16x a trunk convolution per
channel (4x area versus quarter area), so 2x8ch is about half of one
trunk convolution and 2x16ch about one. The final convolution is
zero-initialized, so the model starts life exactly equal to its backbone
and only learns away from it. Freezing the backbone isolates whether the
output-resolution signal carries anything the trunk could not emit.

This is an experimental training interface with no native playback or
quality claim. Core ML conversion and device timing are unverified.
"""
import torch
from torch import nn


class OutputRefinerSPAN(nn.Module):
    """Experimental 2x SPAN with a small post-upsample residual head."""

    def __init__(self, sr, *, train_backbone=False, head_channels=8):
        super().__init__()
        if type(head_channels) is not int or head_channels < 0:
            raise ValueError('nonnegative integer head channels required')
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        from .subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.head_channels = head_channels
        if head_channels:
            self.head = nn.Sequential(
                nn.Conv2d(3, head_channels, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(head_channels, head_channels, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(head_channels, 3, 3, padding=1),
            )
            # Start as the identity: the head contributes nothing until
            # training finds an output-resolution correction worth making.
            # This identity says nothing about quality after training.
            nn.init.zeros_(self.head[4].weight)
            nn.init.zeros_(self.head[4].bias)
        else:
            self.head = None

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def head_parameters(self):
        if self.head is not None:
            yield from self.head.parameters()

    def forward(self, image):
        base = self.sr(image)
        if self.head is None:
            return base
        return base + self.head(base)


def load_output_refiner_checkpoint(path, device='cpu'):
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'output_refiner2x'
            or checkpoint.get('scale') != 2):
        raise ValueError('versioned output-refiner 2x checkpoint required')
    channels = checkpoint.get('head_channels')
    if type(channels) is not int or channels < 0:
        raise ValueError('refiner head channels differ from training declaration')
    model = OutputRefinerSPAN(
        Unshuffled(checkpoint['channels'], scale=2, version=checkpoint['version']),
        head_channels=channels)
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device), checkpoint
