"""FRVSR-inspired previous-output input for an existing 2x SPAN graph.

The first experiment freezes the current-frame graph and learns only a new
history convolution. Joint training and a decoded-observation history ablation
are optional. Motion, rejection and state lifetime belong to the caller.
This is a prototype interface, not an installed native playback model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .subspace_adapter import fuse_convolutions


class RecurrentSPAN(nn.Module):
    def __init__(self, sr, *, train_backbone=False, history_source='sr'):
        super().__init__()
        if history_source not in ('sr', 'decoded'):
            raise ValueError('history source must be sr or decoded')
        self.history_source = history_source
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        # Previous 2x RGB -> 12 LR channels -> 48 channels at the SPAN trunk.
        self.history = nn.Conv2d(48, sr.core.conv_1.out_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.history.weight)

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def forward(self, current, aligned_previous, confidence):
        core = self.sr.core
        history = F.pixel_unshuffle(aligned_previous * confidence, 4)
        features = core.conv_1(self.sr.unshuffle(current)) + self.history(history)
        value = features
        first = None
        for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                   core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if i == 0:
                first = value
        value = core.conv_2(value)
        return core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1)))
