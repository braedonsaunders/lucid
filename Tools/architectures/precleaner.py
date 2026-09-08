"""Small, identity-initialized RGB pre-cleaner ahead of an unchanged SR graph.

Independent architecture inspired by RealBasicVSR's clean-LR supervision idea.
The raw signal remains on a residual path; corrections are bounded to 1/8 of
the RGB range. Runtime quality and cost still require native measurements.
"""
import torch
from torch import nn


class Precleaner(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        if channels < 4:
            raise ValueError('at least four cleaner channels required')
        self.channels = channels
        self.features = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
        self.head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image):
        return (image + .125 * torch.tanh(self.head(self.features(image)))).clamp(0, 1)


class PrecleanedSPAN(nn.Module):
    def __init__(self, sr, cleaner_channels=16, *, train_backbone=False,
                 fused_sr=False, use_cleaner=True):
        super().__init__()
        if not isinstance(fused_sr, bool) or not isinstance(use_cleaner, bool):
            raise ValueError('explicit boolean cleaner and fusion policies required')
        if train_backbone and not fused_sr:
            raise ValueError('joint training requires a differentiable fused SR graph')
        if fused_sr:
            from .subspace_adapter import fuse_convolutions
            fuse_convolutions(sr)
        self.fused_sr = fused_sr
        self.use_cleaner = use_cleaner
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.cleaner = Precleaner(cleaner_channels)
        self.cleaner.requires_grad_(use_cleaner)

    @property
    def core(self):
        return self.sr.core

    @property
    def version(self):
        return self.sr.version

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def forward(self, image):
        return self.sr(self.cleaner(image) if self.use_cleaner else image)
