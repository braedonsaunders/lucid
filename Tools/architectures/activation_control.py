"""Input-conditioned control of a frozen 2x convolutional reconstruction graph.

Adaptation hypothesis informed by SPARK (September 2026), not its DiT/VAE model.
Channel identities and shift units come exclusively from training-source probes.
"""
import copy
import torch
from torch import nn


class ActivationControl(nn.Module):
    def __init__(self, anchor, selection, rms, dynamic=True):
        super().__init__()
        self.anchor = copy.deepcopy(anchor).eval()
        self.dynamic = bool(dynamic)
        self.frames = anchor.frames
        self.version = anchor.version
        core = self.anchor.core
        channels = core.conv_1.out_channels
        if self.frames != 1 or channels != 32 or core.upsampler[1].upscale_factor != 4:
            raise ValueError('fused single-frame 32-channel 2x anchor required')
        if core.img_range != 1 or torch.count_nonzero(core.mean):
            raise ValueError('zero-mean unit-range anchor required')
        masks = torch.zeros(6, 4, channels)
        if len(selection) != 6 or len(rms) != 6:
            raise ValueError('six measured block selections/scales required')
        for i, indices in enumerate(selection):
            if len(indices) != 4 or len(set(indices)) != 4 or any(type(v) is not int or not 0 <= v < channels for v in indices):
                raise ValueError('four unique in-range channels required')
            masks[i, torch.arange(4), indices] = 1
        units = torch.tensor(rms, dtype=torch.float32)
        if units.shape != (6, channels) or not torch.isfinite(units).all() or (units < 0).any():
            raise ValueError('finite nonnegative training RMS required')
        self.register_buffer('masks', masks)
        self.register_buffer('shift_units', units)
        self.controller = nn.Sequential(nn.LayerNorm(channels*2), nn.Linear(channels*2, 64),
                                        nn.SiLU(), nn.Linear(64, 6*2*4))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def coefficients(self, features):
        condition = torch.cat((features.mean((2, 3)), features.abs().mean((2, 3))), 1)
        if not self.dynamic:
            condition = torch.zeros_like(condition)
        return self.controller(condition).tanh().reshape(-1, 6, 2, 4)

    def forward(self, x):
        core = self.anchor.core
        features = core.conv_1(self.anchor.unshuffle(x))
        coefficients = self.coefficients(features)
        current = features
        first = None
        for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                   core.block_4, core.block_5, core.block_6)):
            current, auxiliary, _ = block(current)
            scale = 1 + .5 * (coefficients[:, i, 0] @ self.masks[i])
            shift = .2 * (coefficients[:, i, 1] @ self.masks[i]) * self.shift_units[i]
            current = current * scale[:, :, None, None] + shift[:, :, None, None]
            if i == 0:
                first = current
        current = core.conv_2(current)
        return core.upsampler(core.conv_cat(torch.cat((features, current, first, auxiliary), 1)))


def config_from_probe(probe, dynamic):
    if not probe.get('complete') or probe.get('weights_modified') is not False:
        raise ValueError('complete immutable activation probe required')
    return {'dynamic': dynamic,
            'selection': [probe['selections'][f'core.block_{i}']['top'] for i in range(1, 7)],
            'rms': [[max(0, v)**.5 for v in probe['concentration'][f'core.block_{i}']['mean_squared']]
                    for i in range(1, 7)]}
