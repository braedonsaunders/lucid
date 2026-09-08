"""Frozen reconstruction with one learned per-pixel log-variance output channel."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .subspace_adapter import fuse_convolutions


def confidence_from_log_variance(log_variance):
    # Fixed before evaluation: confidence is 1/2 at 0.05 RGB RMS error.
    return .0025 / (log_variance.exp() + .0025)


def gaussian_error_nll(prediction, reference, log_variance):
    # The variance head cannot reduce its target error by changing the SR mean.
    error = (prediction.detach() - reference.detach()).square().mean(1, keepdim=True)
    return .5 * (log_variance + error * (-log_variance).exp()).mean()


class ConfidenceSPAN(nn.Module):
    def __init__(self, sr):
        super().__init__()
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x backbone required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean backbone required')
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(False)
        self.variance_head = nn.Conv2d(sr.core.conv_1.out_channels, 16, 3, padding=1)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias, math.log(.0025))

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
        # The shared still-image scorer can verify the unchanged mean alone.
        return self.sr(image)

    def predict_with_uncertainty(self, image):
        with torch.no_grad():
            core = self.sr.core
            features = core.conv_1(self.sr.unshuffle(image))
            value = features
            first = None
            for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                       core.block_4, core.block_5, core.block_6)):
                value, auxiliary, _ = block(value)
                if i == 0:
                    first = value
            value = core.conv_2(value)
            fused = core.conv_cat(torch.cat((features, value, first, auxiliary), 1))
            prediction = core.upsampler(fused)
        log_variance = F.pixel_shuffle(self.variance_head(fused.detach()), 4).float().clamp(-12, 0)
        return prediction, log_variance
