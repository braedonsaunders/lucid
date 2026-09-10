"""Full-LR feature memory; preserves pixel phases before SPAN unshuffling.

Eight channels at LR resolution hold the same number of state values as
32 channels at half LR resolution. This is an experimental interface, with no
native playback or quality claim. A driver may deliberately warp packed state
as a matched phase-alignment ablation; the model itself always consumes LR state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FullLRFeatureSPAN(nn.Module):
    requires_feature_state = True
    state_representation = 'full_lr_observation_features_v1'

    def __init__(self, sr, *, train_backbone=False, state_channels=8, initial_decay_bias=-2.,
                 skip_residual=False):
        super().__init__()
        if not math.isfinite(initial_decay_bias):
            raise ValueError('finite initial decay bias required')
        if skip_residual not in (True, False):
            raise ValueError('skip residual must be a bool')
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        if state_channels < 3:
            raise ValueError('state must retain at least three decoded RGB channels')
        from .subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.state_channels = state_channels
        self.skip_residual = skip_residual
        self.raw_encoder = nn.Conv2d(3, state_channels, 3, padding=1, bias=False)
        nn.init.normal_(self.raw_encoder.weight, std=.01)
        # Preserve raw observations explicitly; remaining nonzero filters can
        # learn immediately once the zero projection takes its first update.
        with torch.no_grad():
            self.raw_encoder.weight[:3].zero_()
            for channel in range(3):
                self.raw_encoder.weight[channel, channel, 1, 1] = 1
        self.decay = nn.Conv2d(3, state_channels, 1)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, initial_decay_bias)
        self.project = nn.Conv2d(state_channels * 4, sr.core.conv_1.out_channels, 1, bias=False)
        nn.init.zeros_(self.project.weight)
        self.history_source = 'full_lr_features'
        self.motion_seed = 'search'

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def branch_parameters(self):
        for module in (self.raw_encoder, self.decay, self.project):
            yield from module.parameters()

    def forward(self, current, raw_current, confidence, state=None):
        if (current.ndim != 4 or current.shape[1] != 3 or current.shape != raw_current.shape
                or any(d % 2 for d in current.shape[-2:])):
            raise ValueError('matched even BCHW processed/raw inputs required')
        if confidence.shape != (current.shape[0], 1, *current.shape[-2:]):
            raise ValueError('full-LR confidence geometry required')
        observed = self.raw_encoder(raw_current)
        if state is None:
            state = observed
        else:
            if state.shape != observed.shape:
                raise ValueError('full-LR feature-state geometry differs from observation')
            state = observed + torch.sigmoid(self.decay(current)) * state * confidence.clamp(0, 1)
        core = self.sr.core
        features = core.conv_1(self.sr.unshuffle(current)) + self.project(F.pixel_unshuffle(state, 2))
        value, first = features, None
        for index, block in enumerate((core.block_1, core.block_2, core.block_3,
                                       core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if index == 0:
                first = value
        value = core.conv_2(value)
        output = core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1)))
        if self.skip_residual:
            bypass = F.interpolate(current, scale_factor=2, mode="bilinear",
                                   align_corners=False)
            if bypass.shape != output.shape:
                raise ValueError(f"skip bypass {tuple(bypass.shape)} disagrees with "
                                 f"trunk {tuple(output.shape)}")
            output = output + bypass
        return output, state


def load_full_lr_checkpoint(path, device='cpu'):
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'full_lr_feature_span2x' or checkpoint.get('scale') != 2
            or checkpoint.get('state_representation') != FullLRFeatureSPAN.state_representation
            or checkpoint.get('feature_source') != 'decoded_rgb8'):
        raise ValueError('versioned full-LR decoded-feature checkpoint required')
    args = checkpoint.get('experiment', {}).get('args', {})
    policy, warp = checkpoint.get('source_motion_policy'), checkpoint.get('state_warp')
    if policy not in ('raw', 'taa', 'search') or policy != args.get('source_motion_policy'):
        raise ValueError('full-LR source policy differs from training declaration')
    if warp not in ('full_lr', 'packed_half_lr') or warp != args.get('state_warp'):
        raise ValueError('feature alignment differs from training declaration')
    channels = checkpoint.get('state_channels')
    if (type(channels) is not int or channels < 3
            or channels != args.get('state_channels')):
        raise ValueError('feature state channels differ from training declaration')
    skip = checkpoint.get('skip_residual', False)
    if skip not in (True, False) or skip != args.get('sr_skip_residual', False):
        raise ValueError('skip residual differs from training declaration')
    model = FullLRFeatureSPAN(Unshuffled(checkpoint['channels'], scale=2, version=checkpoint['version']),
                             state_channels=checkpoint['state_channels'],
                             skip_residual=skip)
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device), checkpoint
