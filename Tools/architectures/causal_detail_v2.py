"""Unpromoted spatial-floor experiment: retain samples beside temporal features.

This is a controlled response to v1's measured detail loss, not a novelty claim.
The fixed interpolation filter uses ordinary Core ML-compatible convolutions.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .causal_detail import CausalDetail


class Lanczos2x(nn.Module):
    """Half-pixel-centered Lanczos-3, normalized phases and replicated borders.

No PIL dependency, learned filter, clipping or color mixing. Negative lobes can
overshoot; clip only after adding the learned correction. Border replication
differs from PIL's truncated-kernel normalization and is recorded explicitly.
    """
    def __init__(self):
        super().__init__()
        positions = torch.arange(-3, 4, dtype=torch.float64)
        phases = []
        for phase in (-0.25, 0.25):
            distance = phase-positions
            weights = torch.sinc(distance)*torch.sinc(distance/3)
            weights = torch.where(distance.abs() < 3, weights, 0)
            phases.append(weights/weights.sum())
        kernels = torch.stack([y[:, None]*x[None, :] for y in phases for x in phases])
        self.register_buffer('kernels', kernels[:, None].repeat(3, 1, 1, 1).float())

    def forward(self, frame):
        # Keep the fixed signal path accurate during mixed-precision training.
        # Core ML's FP16 export is checked independently against this FP32 path.
        with torch.autocast(device_type=frame.device.type, enabled=False):
            samples = F.conv2d(F.pad(frame.float(), (3, 3, 3, 3), mode='replicate'),
                               self.kernels.float(), groups=3)
            return F.pixel_shuffle(samples, 2)


class SeparableLanczos2x(Lanczos2x):
    """Equivalent fixed floor with two 1-D passes: 126 vs 588 MACs/LR pixel.

    This changes execution, not the reconstruction filter or learned weights.
    Measure native graph latency: reshape/transpose costs may erase MAC savings.
    """
    def __init__(self):
        super().__init__()
        positions = torch.arange(-3, 4, dtype=torch.float64)
        phases = []
        for phase in (-0.25, 0.25):
            distance = phase - positions
            weights = torch.sinc(distance) * torch.sinc(distance / 3)
            weights = torch.where(distance.abs() < 3, weights, 0)
            phases.append(weights / weights.sum())
        taps = torch.stack(phases).float().repeat(3, 1)
        self.register_buffer('horizontal', taps[:, None, None, :], persistent=False)
        self.register_buffer('vertical', taps[:, None, :, None], persistent=False)

    def forward(self, frame):
        with torch.autocast(device_type=frame.device.type, enabled=False):
            # Core ML variants have fixed image sizes. Materialize these sizes
            # during tracing to avoid dynamic scalar casts in the converter.
            b, c, h, w = (int(size) for size in frame.shape)
            x = F.conv2d(F.pad(frame.float(), (3, 3, 0, 0), mode='replicate'),
                         self.horizontal.float(), groups=3)
            x = x.reshape(b, c, 2, h, w).permute(0, 1, 3, 4, 2).reshape(b, c, h, w * 2)
            x = F.conv2d(F.pad(x, (0, 0, 3, 3), mode='replicate'),
                         self.vertical.float(), groups=3)
            return x.reshape(b, c, 2, h, w * 2).permute(0, 1, 3, 2, 4).reshape(b, c, h * 2, w * 2)


class CausalDetailV2(CausalDetail):
    """Same causal trunk, stronger fixed floor and an uncompressed pixel bypass."""
    def __init__(self, channels=32, blocks=4, scale=2, fold=4):
        if scale != 2:
            raise ValueError('v2 is a measured 2x experiment only')
        super().__init__(channels, blocks, scale, fold)
        self.floor = Lanczos2x()
        self.head = nn.Conv2d(channels + 3*fold*fold, 3*(scale*fold)**2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, frame, previous, valid):
        pixels = F.pixel_unshuffle(frame, self.fold)
        x = self.entry(pixels)
        history = previous * valid
        gate = torch.sigmoid(self.gate(torch.cat((x, history), dim=1))) * valid
        candidate = torch.tanh(x + self.history(history))
        state = (1-gate)*candidate + gate*history
        features = self.blocks(state + self.atoms(state))
        # No feature bottleneck can erase these input samples before the head.
        residual = F.pixel_shuffle(self.head(torch.cat((features, pixels), dim=1)), self.scale*self.fold)
        return self.floor(frame) + residual, state


def make_model(architecture, channels, blocks, scale=2):
    if architecture == 'causal_detail_v1':
        return CausalDetail(channels, blocks, scale)
    if architecture == 'causal_detail_v2':
        return CausalDetailV2(channels, blocks, scale)
    raise ValueError(f'unrecognized causal architecture: {architecture}')
