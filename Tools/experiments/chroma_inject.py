#!/usr/bin/env python3
"""Chroma-first capacity reallocation: wide luma trunk + small chroma branch
with cross-component luma-feature injection (CCALF-style). Research only.

Mechanism: a fixed RGB->YCbCr front-end (a foldable 1x1 matmul), a WIDE SPAN
trunk on LUMA only, and a SMALL learned chroma branch whose input is the
CONCATENATION of bicubic-upsampled chroma and DOWNSAMPLED LUMA FEATURES from
the luma trunk (cross-component injection: the field reconstructs chroma FROM
luma via cross-component models such as CCALF/CCSAO, because chroma is
subsampled and independently poor). A fixed YCbCr->RGB back-end restores RGB.

This is a REALLOCATION, not an expansion: total MACs are held approximately
equal to the equal-width RGB control at the same trunk width (see
`trunk_macs` / `report_macs`). Capacity moves from the 0.10-error luma plane
to the 0.30-0.38-error chroma planes (root-measured per-plane normalized
reconstruction error of the shipping model on 10 holdout frames, YCbCr, each
plane normalized by its own reference energy).

This module reuses luma_asym's front-end/back-end buffers, folding helper,
and warm-start machinery: ChromaInject(40) warm-starts its luma trunk from the
same function-preserving 40-channel widened RGB init as arm B, so the C-vs-B
comparison isolates the reallocation. The chroma branch starts near-identity
(zero-init residual on top of the bicubic path), so at initialization the
model behaves like the luma40 bicubic-chroma design and learns away from it.

TORCH-ONLY. Torch-level gains here have twice failed to survive into the
deployed native pipeline. Nothing in this file is a deployable-gain claim.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from luma_asym import (
    LUMA_WEIGHTS, RGB2YCC, YCC_BIAS, _buffers,
    folded_rgb_first_weight, warm_start_from_widened,
)


def _sync(module):
    """Refresh fused eval params when the module still has them.

    The FullLR wrappers call fuse_convolutions(sr), which replaces every
    Conv3XC with a plain fused Conv2d that has no update_params. Calling
    this instead of update_params() keeps the backbone callable both
    standalone (unfused, training-mode reparam) and fused inside a wrapper.
    """
    update = getattr(module, 'update_params', None)
    if update is not None:
        update()

CHROMA_BRANCH_CHANNELS = 8


class ChromaInject(nn.Module):
    """Unshuffled-compatible single-frame 2x backbone: Y-deep + injected chroma.

    Same call signature as Unshuffled: (B, 3*frames, H, W) in, (B, 3, 2H, 2W)
    out, plus .frames / .version / .unshuffle / .core for the training
    harness. Chroma is taken from the current (last) frame only.
    """

    def __init__(self, luma_channels=40, chroma_channels=CHROMA_BRANCH_CHANNELS,
                 scale=2, frames=1, version=1, chroma_mode='bilinear'):
        super().__init__()
        if version != 1:
            raise ValueError('only the v1 SPAN trunk is supported here')
        if frames < 1 or scale != 2:
            raise ValueError('single-scale 2x geometry required')
        if luma_channels < 8 or chroma_channels < 1:
            raise ValueError('positive luma/chroma widths required')
        if chroma_mode not in ('bicubic', 'bilinear'):
            raise ValueError('chroma upsampling must be bicubic or bilinear')
        # Bilinear is the default because coremltools implements no bicubic
        # upsample and this backbone (unlike luma_asym's null test) is meant
        # to convert; bicubic is kept only for research comparisons that
        # never ship. The learned residual on top is unaffected either way.
        self.chroma_mode = chroma_mode
        from architectures.span_arch import SPAN, Conv3XC
        self.frames = frames
        self.version = version
        self.luma_channels = luma_channels
        self.chroma_channels = chroma_channels
        fwd, fbias, bwd, bbias = _buffers()
        self.register_buffer('rgb2ycc', fwd)
        self.register_buffer('ycc_bias', fbias)
        self.register_buffer('ycc2rgb', bwd)
        self.register_buffer('rgb_bias', bbias)
        # Trace-safe color maps: fixed 1x1 convolutions instead of the
        # einsum/reshape below, so no shape unpack enters a traced CoreML
        # graph (mirrors luma_asym.LumaAsymmetric's ycc_conv/rgb_conv).
        self.ycc_conv = nn.Conv2d(3, 3, 1)
        self.ycc_conv.weight.data.copy_(fwd.view(3, 3, 1, 1))
        self.ycc_conv.bias.data.copy_(fbias)
        self.rgb_conv = nn.Conv2d(3, 3, 1)
        self.rgb_conv.weight.data.copy_(bwd.view(3, 3, 1, 1))
        self.rgb_conv.bias.data.copy_(bbias)
        for module in (self.ycc_conv, self.rgb_conv):
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.core = SPAN(num_in_ch=4 * frames, num_out_ch=1,
                         feature_channels=luma_channels,
                         upscale=scale * 2, img_range=1.0,
                         rgb_mean=(0.0, 0.0, 0.0))
        # SPAN's mean is shaped for 3 channels; the trunk takes 4*frames, so
        # the same zero broadcast the Unshuffled wrapper uses is needed here.
        self.core.mean = torch.zeros(1, 1, 1, 1)
        # Small chroma branch: 3x3 conv on (bicubic chroma_up(2ch) +
        # downsampled luma features) -> refined chroma_up(2ch), residual on
        # top of the bicubic path, zero-initialized so the model starts as
        # the bicubic-chroma design and learns away from it.
        inject_in = 2 + luma_channels
        self.chroma_refine = Conv3XC(inject_in, 2, gain1=1, s=1)
        with torch.no_grad():
            for module in self.chroma_refine.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
            self.chroma_refine.update_params()

    def to_ycc(self, x):
        channels = x.shape[1]
        frames = channels // 3
        parts = [self.ycc_conv(x[:, 3 * f:3 * f + 3]) for f in range(frames)]
        return torch.cat(parts, 1) if len(parts) > 1 else parts[0]

    def to_rgb(self, ycc):
        return self.rgb_conv(ycc)

    def chroma_from_luma_features(self, chroma_up, luma_features):
        """Refine bicubic chroma with injected luma features (residual)."""
        if (chroma_up.ndim != 4 or chroma_up.shape[1] != 2
                or luma_features.ndim != 4 or luma_features.shape[1] != self.luma_channels
                or chroma_up.shape[-2:] != luma_features.shape[-2:]):
            raise ValueError('matched 2x chroma-up / luma-feature geometry required')
        injected = torch.cat((chroma_up, luma_features), 1)
        _sync(self.chroma_refine)
        return chroma_up + self.chroma_refine(injected)

    def upsample_chroma(self, chroma):
        if self.chroma_mode == 'bicubic':
            return F.interpolate(chroma, scale_factor=2, mode='bicubic',
                                 align_corners=False, antialias=True)
        return F.interpolate(chroma, scale_factor=2, mode='bilinear',
                             align_corners=False)

    def forward(self, x, return_planes=False):
        if x.ndim != 4 or x.shape[1] != 3 * self.frames or any(d % 2 for d in x.shape[-2:]):
            raise ValueError('even (B, 3*frames, H, W) input required')
        ycc = self.to_ycc(x)
        luma = torch.cat([ycc[:, 3 * f:3 * f + 1] for f in range(self.frames)], 1)
        luma_up = self.core(self.unshuffle(luma))
        # Luma features for cross-component injection: a hook-free re-entry
        # through conv_1 only (1 conv, negligible vs the 19-conv trunk). The
        # graph is kept so injection features train the trunk. _sync is a
        # no-op once fuse_convolutions has replaced conv_1 with a Conv2d.
        _sync(self.core.conv_1)
        lr_features = self.core.conv_1(self.unshuffle(luma[:, -1:]))
        chroma = ycc[:, 3 * (self.frames - 1) + 1:3 * (self.frames - 1) + 3]
        chroma_up = self.upsample_chroma(chroma)
        luma_feat_up = F.interpolate(lr_features, size=chroma_up.shape[-2:],
                                     mode='bilinear', align_corners=False)
        chroma_ref = self.chroma_from_luma_features(chroma_up, luma_feat_up)
        out = self.to_rgb(torch.cat((luma_up, chroma_ref), 1))
        if return_planes:
            return out, {'luma_up': luma_up, 'chroma_up': chroma_up,
                         'chroma_ref': chroma_ref}
        return out


def warm_start_chroma_inject(model, wide):
    """Warm-start the luma trunk from the widened RGB init; zero chroma head.

    Reuses luma_asym.warm_start_from_widened for the trunk (same achromatic
    guarantees), and leaves the zero-initialized chroma branch untouched so
    the model starts exactly as the bicubic-chroma design.
    """
    warm_start_from_widened(model, wide)
    with torch.no_grad():
        for module in model.chroma_refine.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
        _sync(model.chroma_refine)
    return model


def conv_macs(in_ch, out_ch, k, h, w, out_h=None, out_w=None):
    """MACs of one same-padded convolution producing (out_h, out_w)."""
    return in_ch * out_ch * k * k * (out_h or h) * (out_w or w)


def span_trunk_macs(width, h, w, in_ch):
    """Eval-mode MACs of one SPAN trunk (fused Conv3XC) at LR (h, w).

    Per Conv3XC fused eval conv: 3x3. Per SPAB block: three width->width
    convs. Plus conv_1 (in_ch->width), conv_2 (width->width), conv_cat
    1x1 (4*width->width), upsampler conv 3x3 (width->out_ch*16).
    Returns (trunk_macs, upsampler_macs) with out_ch=1 (luma) unless noted.
    """
    per_conv = conv_macs(width, width, 3, h, w)
    blocks = 6 * 3 * per_conv
    first = conv_macs(in_ch, width, 3, h, w)
    last = conv_macs(width, width, 3, h, w)
    cat = conv_macs(4 * width, width, 1, h, w)
    return first + blocks + last + cat


def upsampler_macs(width, out_ch, h, w):
    return conv_macs(width, out_ch * 16, 3, h, w)


def chroma_branch_macs(luma_channels, h2, w2):
    """Eval-mode MACs of the chroma refinement Conv3XC at 2x size.

    gain1=1: 1x1 (inject_in->inject_in) + 3x3 (inject_in->2) + 1x1
    (2->2), fused at eval to a single 3x3 (inject_in->2).
    """
    inject_in = 2 + luma_channels
    return conv_macs(inject_in, 2, 3, h2, w2)


def report_macs(width=40, h=180, w=320, luma_channels=40,
                chroma_channels=CHROMA_BRANCH_CHANNELS):
    """Compare MACs: equal-width RGB control vs chroma-inject reallocation.

    LR working size (h, w) = quarter of the 640x360 input after 2x unshuffle
    (160x90); defaults use 180x320 (quarter of 720p) for readability.
    """
    # Arm B: RGB trunk, unshuffled 12ch in, 40 wide, 3ch out.
    rgb = span_trunk_macs(width, h, w, 12) + upsampler_macs(width, 3, h, w)
    # Arm C: luma trunk (4ch in, 1ch out) + chroma branch at 2x + conv_1
    # re-entry for injection features.
    luma = span_trunk_macs(luma_channels, h, w, 4) + upsampler_macs(luma_channels, 1, h, w)
    reinject = conv_macs(4, luma_channels, 3, h, w)
    branch = chroma_branch_macs(luma_channels, 2 * h, 2 * w)
    chroma_total = luma + reinject + branch
    return {'rgb_control': rgb, 'luma_trunk': luma,
            'inject_reeentry_conv_1': reinject, 'chroma_branch': branch,
            'chroma_inject_total': chroma_total,
            'ratio_c_over_b': chroma_total / rgb}


class FullLRChromaInjectSPAN(nn.Module):
    """Full-LR feature wrapper around a ChromaInject backbone (arm C only).

    Mirrors luma_asym.FullLRLumaSPAN line for line, except the chroma path
    uses the learned cross-component refinement instead of fixed bicubic.
    Arms A/B stay on the byte-identical shared wrapper; this class exists
    because the shared wrapper hard-codes a 12-channel RGB trunk input and
    has no chroma branch.
    """

    requires_feature_state = True
    state_representation = 'full_lr_observation_features_v1'

    def __init__(self, sr, *, train_backbone=False, state_channels=8,
                 initial_decay_bias=-2.):
        super().__init__()
        if not math.isfinite(initial_decay_bias):
            raise ValueError('finite initial decay bias required')
        if type(sr) is not ChromaInject:
            raise ValueError('chroma-inject backbone required')
        if state_channels < 3:
            raise ValueError('state must retain at least three decoded RGB channels')
        from architectures.subspace_adapter import fuse_convolutions
        fuse_convolutions(sr)
        self.sr = sr.eval().requires_grad_(train_backbone)
        self.state_channels = state_channels
        self.raw_encoder = nn.Conv2d(3, state_channels, 3, padding=1, bias=False)
        nn.init.normal_(self.raw_encoder.weight, std=.01)
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
        luma = self.sr.to_ycc(current)[:, :1]
        features = core.conv_1(self.sr.unshuffle(luma)) + self.project(F.pixel_unshuffle(state, 2))
        value, first = features, None
        for index, block in enumerate((core.block_1, core.block_2, core.block_3,
                                       core.block_4, core.block_5, core.block_6)):
            value, auxiliary, _ = block(value)
            if index == 0:
                first = value
        value = core.conv_2(value)
        luma_up = core.upsampler(core.conv_cat(torch.cat((features, value, first, auxiliary), 1)))
        chroma = self.sr.to_ycc(current)[:, 1:]
        chroma_up = self.sr.upsample_chroma(chroma)
        luma_feat_up = F.interpolate(features[:, :self.sr.luma_channels],
                                     size=chroma_up.shape[-2:],
                                     mode='bilinear', align_corners=False)
        chroma_ref = self.sr.chroma_from_luma_features(chroma_up, luma_feat_up)
        return self.sr.to_rgb(torch.cat((luma_up, chroma_ref), 1)), state


def load_arm_checkpoint(path, device='cpu'):
    """Rebuild the FullLR-wrapped backbone a train_chroma_arms checkpoint saved."""
    from architectures.full_lr_feature_span import FullLRFeatureSPAN
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('scale') != 2:
        raise ValueError('2x arm checkpoint required')
    backbone = checkpoint.get('backbone')
    args = checkpoint.get('experiment', {}).get('args', {})
    if backbone == 'chroma40':
        channels = checkpoint.get('luma_channels')
        if type(channels) is not int or channels != args.get('luma_channels'):
            raise ValueError('luma channels differ from training declaration')
        sr = ChromaInject(channels, scale=2,
                          version=checkpoint.get('version', 1))
        model = FullLRChromaInjectSPAN(sr, state_channels=checkpoint.get('state_channels', 8))
    elif backbone in ('rgb32', 'rgb40'):
        channels = checkpoint.get('channels')
        if type(channels) is not int or channels != args.get('backbone_channels'):
            raise ValueError('backbone channels differ from training declaration')
        sr = Unshuffled(channels, scale=2, version=checkpoint.get('version', 1))
        model = FullLRFeatureSPAN(sr, state_channels=checkpoint.get('state_channels', 8),
                                  skip_residual=False)
    else:
        raise ValueError('unknown arm backbone tag')
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device), checkpoint
