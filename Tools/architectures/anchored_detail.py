"""Experimental 2x detail branch conditioned on an immutable reconstruction.

Inspired by PixelIR's separately optimized fidelity/detail stages (Aug 2026),
not a reproduction of its flow teacher or transformer student. This compact
branch uses ordinary convolutions at half input resolution. It has no history,
noise input or temporal filtering, and its zero initialization preserves the
anchor exactly. A frozen base does not guarantee the added residual is faithful;
independent image-quality evidence is still required.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DetailBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
            nn.GELU(), nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1))

    def forward(self, x):
        return x + self.branch(x)


class AnchoredDetail(nn.Module):
    def __init__(self, anchor, channels=32, blocks=4, residual_lowpass=False):
        super().__init__()
        if channels < 4 or blocks < 1:
            raise ValueError('positive detail capacity required')
        self.anchor = anchor.eval().requires_grad_(False)
        self.channels, self.blocks = channels, blocks
        # Source 2x2 phases and reconstructed 4x4 phases share the same grid.
        self.input = nn.Conv2d(12 + 48, channels, 1)
        self.body = nn.Sequential(*(DetailBlock(channels, (1, 2, 3, 1)[i % 4]) for i in range(blocks)))
        self.head = nn.Conv2d(channels, 48, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.residual_lowpass = residual_lowpass
        if residual_lowpass:
            # Same fixed spatial operator as the diagnostic, now inside the
            # trainable branch. The anchor bypasses it. No temporal state.
            grid = torch.arange(-3, 4, dtype=torch.float32)
            kernel = torch.exp(-grid.square()/2)
            self.register_buffer('residual_kernel', kernel/kernel.sum())

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, image):
        base = self.anchor(image)
        condition = torch.cat((F.pixel_unshuffle(image, 2), F.pixel_unshuffle(base, 4)), dim=1)
        detail = F.pixel_shuffle(self.head(self.body(self.input(condition))), 4)
        if self.residual_lowpass:
            detail = F.conv2d(F.pad(detail, (3,3,0,0), mode='replicate'),
                              self.residual_kernel.reshape(1,1,1,7).expand(3,1,1,7), groups=3)
            detail = F.conv2d(F.pad(detail, (0,0,3,3), mode='replicate'),
                              self.residual_kernel.reshape(1,1,7,1).expand(3,1,7,1), groups=3)
        return base + detail
