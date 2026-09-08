"""Local degradation conditioning of a frozen or jointly trained SPAN backbone.

This is a small codec-domain mechanism probe, not a DASR reproduction. The
oracle receives clean LR only in explicit training/diagnostic calls. Ordinary
inference derives every condition from the current decoded RGB image.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .subspace_adapter import fuse_convolutions


def local_curvature(image):
    gray = image[:, :1] * .2126 + image[:, 1:2] * .7152 + image[:, 2:3] * .0722
    padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
    dx = padded[..., 1:-1, :-2] - 2 * gray + padded[..., 1:-1, 2:]
    dy = padded[..., :-2, 1:-1] - 2 * gray + padded[..., 2:, 1:-1]
    return torch.cat((dx, dy), 1)


def oracle_degradation(decoded, clean):
    """Reference-only error magnitude; includes codec, chroma and resize error."""
    magnitude = (decoded - clean).abs().mean(1, keepdim=True)
    structure = (local_curvature(decoded) - local_curvature(clean)).abs().mean(1, keepdim=True)
    return (16 * F.avg_pool2d(torch.cat((magnitude, structure), 1), 5, stride=1,
                              padding=2, count_include_pad=False)).clamp(0, 1).detach()


class DegradationEstimator(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(5, channels, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(channels, 2, 3, padding=1), nn.Sigmoid())

    def forward(self, image):
        # Curvature responds to real texture too; supervision must distinguish it
        # from damage. No assumption about an aligned 8x8 codec grid is made.
        return self.layers(torch.cat((image, local_curvature(image).abs()), 1))


class DegradationConditionedSPAN(nn.Module):
    def __init__(self, sr, mode='estimated', estimator_channels=16, *,
                 train_backbone=False, use_conditioning=True):
        super().__init__()
        if not isinstance(train_backbone, bool) or not isinstance(use_conditioning, bool):
            raise ValueError('explicit boolean training and conditioning policies required')
        if mode not in ('constant', 'estimated', 'oracle'):
            raise ValueError('unknown degradation condition mode')
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x SR backbone required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('zero-mean unit-range backbone required')
        # Fold in FP32 once; Conv3XC otherwise refreshes cached weights inside
        # autocast, which would change the supposedly frozen inference graph.
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.mode = mode
        self.use_conditioning = use_conditioning
        self.estimator = DegradationEstimator(estimator_channels)
        channels = sr.core.conv_1.out_channels
        self.modulation = nn.Conv2d(2, channels * 2, 3, padding=1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.modulation.requires_grad_(use_conditioning)
        self.estimator.requires_grad_(use_conditioning and mode == 'estimated')

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

    def conditioned(self, image, condition):
        if not self.use_conditioning:
            return self.sr(image)
        core = self.sr.core
        features = core.conv_1(self.sr.unshuffle(image))
        values = self.modulation(F.avg_pool2d(condition, 2)).tanh()
        gain, bias = values.chunk(2, 1)
        features = features * (1 + .1 * gain).to(features.dtype) + (.02 * bias).to(features.dtype)
        current = features
        first = None
        for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                   core.block_4, core.block_5, core.block_6)):
            current, auxiliary, _ = block(current)
            if i == 0:
                first = current
        current = core.conv_2(current)
        return core.upsampler(core.conv_cat(torch.cat((features, current, first, auxiliary), 1)))

    def forward(self, image):
        if self.mode == 'oracle':
            raise ValueError('oracle has no deployable inference path; supply an explicit diagnostic condition')
        if not self.use_conditioning:
            return self.sr(image)
        condition = (torch.full_like(image[:, :2], .25) if self.mode == 'constant'
                     else self.estimator(image))
        return self.conditioned(image, condition)
