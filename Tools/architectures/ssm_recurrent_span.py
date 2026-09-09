"""Selective-scan temporal state for the 2x SPAN reconstruction graph.

Replaces the fixed zero-initialized history convolution in RecurrentSPAN with
a content-gated linear recurrence over aligned previous features. The update
is a per-channel selective scan: the decay and the input gain are both
functions of the current frame, so the state keeps what the current frame
cannot predict and forgets the rest. No future frames, no attention, no
transformer. The added neural work is a history-encoding convolution, two
depthwise gates and elementwise state arithmetic. Motion estimation and feature
warping add separate costs; Core ML conversion and device placement are untested.

The caller supplies aligned RGB and aligned feature state. Forward returns
the 2x output and next feature state; this is an experimental sequence interface,
not the installed playback model. Native conversion and cost remain unverified.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SelectiveScanState(nn.Module):
    """Content-gated linear recurrence over aligned history features.

    h_t = decay(x_t) * (h_{t-1} * confidence) + gain(x_t) * history_t
    where decay and gain are per-channel gates in (0, 1) computed from the
    current-frame trunk input. Initialized to behave like the old history
    convolution at step zero: decay bias negative (prefer fresh input),
    gain path zero-initialized (add nothing until trained).
    """

    def __init__(self, channels):
        super().__init__()
        if channels < 8:
            raise ValueError('state channels must cover the trunk width')
        self.channels = channels
        # 48 history channels (pixel-unshuffled 2x RGB) -> trunk width.
        self.encode = nn.Conv2d(48, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.encode.weight)
        self.decay = nn.Conv2d(channels, channels, 1, groups=channels)
        nn.init.zeros_(self.decay.weight)
        # Start forgetful: sigmoid(-2) ~= 0.12 keeps little history until
        # training finds evidence the past helps. The old zero-init history
        # conv has the same zero contribution at initialization. This identity
        # says nothing about quality after training.
        nn.init.constant_(self.decay.bias, -2.0)
        self.gain = nn.Conv2d(channels, channels, 1, groups=channels)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, trunk_input, aligned_previous, confidence, state):
        """One selective-scan step.

        trunk_input: current-frame features at the SPAN trunk (B, C, H, W).
        aligned_previous: motion-warped previous 2x RGB (B, 3, 4H, 4W).
        confidence: warp validity (B, 1, 4H, 4W), zeros on cuts/occlusions.
        state: aligned previous scan state (B, C, H, W) or None.
        The caller passes zero confidence on first frames and explicit resets.
        """
        # Mask source samples before the spatial convolution, then mask its
        # destination below: invalid neighbors must neither leak into a valid
        # cell nor seed an invalid cell. Uniform partial confidence therefore
        # attenuates fresh history quadratically; this is conservative gating,
        # not a calibrated probability or a single confidence weighting.
        history = F.pixel_unshuffle(aligned_previous * confidence, 4)
        valid = F.avg_pool2d(confidence, 4).clamp(0, 1)
        encoded = self.encode(history)
        decay = torch.sigmoid(self.decay(trunk_input))
        gain = torch.sigmoid(self.gain(trunk_input)) * valid
        if state is None:
            new_state = gain * encoded
        else:
            if state.shape != encoded.shape:
                raise ValueError('scan state geometry differs from trunk input')
            new_state = decay * (state * valid) + gain * encoded
        return new_state, new_state


class SSMRecurrentSPAN(nn.Module):
    """Experimental SPAN with an explicit selective-scan state interface.

    Accepts the same constructor arguments and checkpoint guards as
    RecurrentSPAN (single-frame 2x unit-range source model). The SR trunk is
    frozen by default; only the scan module trains unless joint is set.
    """

    requires_feature_state = True

    def __init__(self, sr, *, train_backbone=False):
        super().__init__()
        if sr.frames != 1 or sr.core.upsampler[1].upscale_factor != 4:
            raise ValueError('single-frame 2x source model required')
        if sr.core.img_range != 1 or torch.count_nonzero(sr.core.mean):
            raise ValueError('unit-range zero-mean source model required')
        from .subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.history_source = 'sr'
        self.motion_seed = 'search'
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.scan = SelectiveScanState(sr.core.conv_1.out_channels)
        self.history_parameters = sum(p.numel() for p in self.scan.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def forward(self, current, aligned_previous, confidence, state=None):
        core = self.sr.core
        trunk_input = core.conv_1(self.sr.unshuffle(current))
        contribution, new_state = self.scan(trunk_input, aligned_previous, confidence, state)
        features = trunk_input + contribution
        value = features
        first = None
        for i, block in enumerate((core.block_1, core.block_2, core.block_3,
                                  core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if i == 0:
                first = value
        value = core.conv_2(value)
        return core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1))), new_state


def load_ssm_checkpoint(path, device='cpu'):
    """Load only versioned feature-state checkpoints for explicit sequence use."""
    from train_span import Unshuffled
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('architecture') != 'ssm_recurrent_span2x' or state.get('scale') != 2
            or state.get('state_representation') != 'aligned_scan_features_v1'):
        raise ValueError('versioned aligned-feature SSM checkpoint required')
    policy = state.get('source_motion_policy')
    declared = state.get('experiment', {}).get('args', {}).get('source_motion_policy')
    if policy not in ('raw', 'taa', 'search') or declared != policy:
        raise ValueError('SSM source policy differs from training declaration')
    model = SSMRecurrentSPAN(Unshuffled(state['channels'], scale=2, version=state['version']))
    model.load_state_dict(state['model'])
    return model.eval().to(device), state
