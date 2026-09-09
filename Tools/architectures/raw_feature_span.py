"""Experimental raw-observation feature memory around a fixed-size SPAN trunk.

The memory encoder sees decoded LR directly, before source preprocessing and
generated RGB quantization. A current-only control uses the same encoder and
projection but discards carried features. Zero projection preserves the source
SR function at initialization. This is a training prototype, not a native
playback interface or a claim of measured quality/latency improvement.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class RawFeatureSPAN(nn.Module):
    requires_feature_state = True
    state_representation = 'aligned_raw_features_v1'

    def __init__(self, sr, *, train_backbone=False):
        super().__init__()
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        from .subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        channels = sr.core.conv_1.out_channels
        # Begin with pretrained observation features; this is an independent
        # trainable copy even when the reconstruction backbone stays frozen.
        self.raw_encoder = copy.deepcopy(sr.core.conv_1).requires_grad_(True)
        self.decay = nn.Conv2d(channels, channels, 1, groups=channels)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, -2.)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.project.weight)
        self.history_source = 'raw_features'
        self.motion_seed = 'search'

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def branch_parameters(self):
        for module in (self.raw_encoder, self.decay, self.project):
            yield from module.parameters()

    def forward(self, current, raw_current, confidence, state=None):
        """Predict 2x RGB and retain fresh decoded-observation features.

        current/raw_current: matched BCHW LR, even spatial dimensions.
        confidence: Bx1x2Hx2W validity of motion-aligned carried features.
        state: already motion-aligned BxCx(H/2)x(W/2), or None.

        On a reset, zero confidence removes history but retains the current raw
        features. After training, the current-only branch can improve a first
        frame too; first-frame equality is to this model's current-only output,
        not necessarily its original backbone.
        """
        if (current.ndim != 4 or current.shape[1] != 3 or
                current.shape != raw_current.shape or any(x % 2 for x in current.shape[-2:])):
            raise ValueError('matched even BCHW processed/raw inputs required')
        if confidence.shape != (current.shape[0], 1, current.shape[2] * 2, current.shape[3] * 2):
            raise ValueError('matched 2x confidence geometry required')
        core = self.sr.core
        trunk = core.conv_1(self.sr.unshuffle(current))
        observed = self.raw_encoder(self.sr.unshuffle(raw_current))
        if state is None:
            new_state = observed
        else:
            if state.shape != observed.shape:
                raise ValueError('raw feature-state geometry differs from trunk')
            valid = F.avg_pool2d(confidence, 4).clamp(0, 1)
            new_state = observed + torch.sigmoid(self.decay(trunk)) * state * valid
        features = trunk + self.project(new_state)
        value = features
        first = None
        for index, block in enumerate((core.block_1, core.block_2, core.block_3,
                                       core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if index == 0:
                first = value
        value = core.conv_2(value)
        output = core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1)))
        return output, new_state


def load_raw_feature_checkpoint(path, device='cpu'):
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'raw_feature_span2x' or checkpoint.get('scale') != 2
            or checkpoint.get('state_representation') != RawFeatureSPAN.state_representation
            or checkpoint.get('feature_source') != 'decoded_rgb8'):
        raise ValueError('versioned decoded-feature 2x checkpoint required')
    policy = checkpoint.get('source_motion_policy')
    declared = checkpoint.get('experiment', {}).get('args', {}).get('source_motion_policy')
    if policy not in ('raw', 'taa', 'search') or policy != declared:
        raise ValueError('raw feature source policy differs from training declaration')
    model = RawFeatureSPAN(Unshuffled(checkpoint['channels'], scale=2, version=checkpoint['version']))
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device), checkpoint
