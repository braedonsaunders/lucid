"""Same-frame pre-TAA tap around a fixed-size SPAN trunk.

The native pipeline runs deband/TAA *before* SR, so the model only ever sees
post-accumulation frames and cannot distinguish history-smear from true
smoothness. This restores that destroyed evidence cheaply: the model sees BOTH
the TAA-accumulated frame (the ordinary SR input) AND the raw decoded current
frame at the same timestep. No cross-frame alignment, no state, no recurrence.

The tap is spatially registered same-resolution evidence, usable by local
filters from step one - it behaves like a residual, not a conditioning code.
The accumulated frame keeps the original 12 stem channels, so a single-frame
checkpoint stays a prefix of the widened input (the current frame is the only
frame here, trivially last); the raw tap enters only through a zero-initialized
side path, so every tap mode preserves the source SR function exactly at
initialization. This is a training prototype, not a native playback interface
or a claim of measured quality/latency improvement.
"""
import torch
from torch import nn
from torch.nn import functional as F

TAP_MODES = ('none', 'full', 'luma')


def luma(frame):
    """Native luma weights, matching native_stages.rgb_to_planes."""
    r, g, b = frame.split(1, 1)
    return .2126 * r + .7152 * g + .0722 * b


class TapSPAN(nn.Module):
    requires_feature_state = False
    state_representation = 'same_frame_tap_v1'

    def __init__(self, sr, *, train_backbone=False, tap='full'):
        super().__init__()
        if tap not in TAP_MODES:
            raise ValueError('tap must be none, full or luma')
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        from .subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.tap = tap
        channels = sr.core.conv_1.out_channels
        if tap == 'full':
            # 3 raw RGB channels x4 from the 2x2 unshuffle.
            self.tap_conv = nn.Conv2d(12, channels, 1, bias=False)
        elif tap == 'luma':
            # Single-channel luma residual x4 from the 2x2 unshuffle.
            self.tap_conv = nn.Conv2d(4, channels, 1, bias=False)
        else:
            self.tap_conv = None
        if self.tap_conv is not None:
            # Preserve the source SR function exactly at initialization; the
            # tap can only learn away from the baseline, never start from noise.
            nn.init.zeros_(self.tap_conv.weight)
        self.history_source = 'same_frame_tap'
        self.motion_seed = 'search'

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def branch_parameters(self):
        if self.tap_conv is not None:
            yield from self.tap_conv.parameters()

    def forward(self, processed, raw):
        """Predict 2x RGB from accumulated + raw decoded LR at one timestep.

        processed/raw: matched BCHW LR RGB, even spatial dimensions. processed
        is the ordinary (debanded, TAA-accumulated) SR input; raw is the
        decoded current frame before source preprocessing.
        """
        if (processed.ndim != 4 or processed.shape[1] != 3 or processed.shape != raw.shape
                or any(d % 2 for d in processed.shape[-2:])):
            raise ValueError('matched even BCHW processed/raw inputs required')
        core = self.sr.core
        features = core.conv_1(self.sr.unshuffle(processed))
        if self.tap == 'full':
            features = features + self.tap_conv(self.sr.unshuffle(raw))
        elif self.tap == 'luma':
            with torch.no_grad():
                residual = luma(raw) - luma(processed)
            features = features + self.tap_conv(self.sr.unshuffle(residual))
        value, first = features, None
        for index, block in enumerate((core.block_1, core.block_2, core.block_3,
                                       core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if index == 0:
                first = value
        value = core.conv_2(value)
        return core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1)))


def load_tap_checkpoint(path, device='cpu'):
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('architecture') != 'tap_span2x' or checkpoint.get('scale') != 2
            or checkpoint.get('state_representation') != TapSPAN.state_representation
            or checkpoint.get('feature_source') != 'decoded_rgb8'):
        raise ValueError('versioned same-frame-tap 2x checkpoint required')
    args = checkpoint.get('experiment', {}).get('args', {})
    tap = checkpoint.get('tap_mode')
    if tap not in TAP_MODES or tap != args.get('tap_mode'):
        raise ValueError('tap mode differs from training declaration')
    policy = checkpoint.get('source_motion_policy')
    if policy not in ('raw', 'taa', 'search') or policy != args.get('source_motion_policy'):
        raise ValueError('tap source policy differs from training declaration')
    model = TapSPAN(Unshuffled(checkpoint['channels'], scale=2, version=checkpoint['version']),
                    tap=tap)
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device), checkpoint
