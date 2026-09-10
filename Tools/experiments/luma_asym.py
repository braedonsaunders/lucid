#!/usr/bin/env python3
"""Luma-dominant asymmetric trunk (Y-deep, chroma-cheap). Research only.

Mechanism: a fixed RGB->YCbCr front-end (a foldable 1x1 matmul), a deep SPAN
trunk on LUMA only (40 channels), and a zero-parameter plain-bicubic path for
CbCr. The trunk emits luma at 2x; a fixed YCbCr->RGB back-end restores RGB.

Why bicubic and not the tiny learned 8-channel head: bicubic is the sharpest
form of the claim (chroma carries no recoverable detail past 4:2:0 decoding),
adds zero parameters, and keeps the C-vs-B comparison to exactly one variable
(the asymmetry). A learned chroma head would confound width with asymmetry.

Foldability (deployment story, verified numerically in test_luma_asym.py):
  * The front-end folds EXACTLY into the trunk's first convolution: it is a
    per-pixel 1x1 linear map applied before the unshuffle permutation, so the
    composed first layer is an ordinary RGB convolution. No front-end op needs
    to ship.
  * The back-end does NOT fold into the trunk alone, because chroma bypasses
    the trunk. Deployment would keep one fixed 1x1 output map plus bicubic
    chroma. Stated plainly so nobody claims a single-conv deployment.

Warm start (NOT function-preserving, stated plainly): arm C's trunk blocks are
copied verbatim from the function-preserving 40-channel widened RGB init, and
its first/upsampler layers carry the luma projection of the widened weights.
On a grayscale input the C backbone therefore reproduces the widened RGB
backbone's luma output; on color input the chroma path deliberately diverges.

TORCH-ONLY. Torch-level gains here have twice failed to survive into the
deployed native pipeline. Nothing in this file is a deployable-gain claim.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# BT.601 full-range RGB -> YCbCr with Cb/Cr centred on 0.5 (inputs are 0-1).
RGB2YCC = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
])
YCC_BIAS = torch.tensor([0.0, 0.5, 0.5])


def _buffers():
    forward = RGB2YCC.clone()
    backward = torch.linalg.inv(forward)
    return forward, YCC_BIAS.clone(), backward, (-(backward @ YCC_BIAS)).clone()


class LumaAsymmetric(nn.Module):
    """Unshuffled-compatible single-frame 2x backbone with a Y-only trunk.

    Same call signature as Unshuffled: (B, 3*frames, H, W) in, (B, 3, 2H, 2W)
    out, plus .frames / .version / .unshuffle / .core for the training
    harness. Chroma is taken from the current (last) frame only.
    """

    def __init__(self, luma_channels=40, scale=2, frames=1, version=1,
                 chroma_mode='bicubic'):
        super().__init__()
        if version != 1:
            raise ValueError('only the v1 SPAN trunk is supported here')
        if frames < 1 or scale != 2:
            raise ValueError('single-scale 2x geometry required')
        if chroma_mode not in ('bicubic', 'bilinear'):
            raise ValueError('chroma upsampling must be bicubic or bilinear')
        # Bicubic is the research default; bilinear exists ONLY because
        # coremltools implements no bicubic upsample, so a traced deployment
        # shape of this module must substitute it. The trunk is untouched.
        self.chroma_mode = chroma_mode
        from architectures.span_arch import SPAN
        self.frames = frames
        self.version = version
        self.luma_channels = luma_channels
        fwd, fbias, bwd, bbias = _buffers()
        self.register_buffer('rgb2ycc', fwd)
        self.register_buffer('ycc_bias', fbias)
        self.register_buffer('ycc2rgb', bwd)
        self.register_buffer('rgb_bias', bbias)
        # Trace-safe color maps: the same fixed matmuls as 1x1 convolutions so
        # no shape unpack, view, or einsum ever enters the traced graph. Kept
        # in lockstep with the buffers by construction (same tensors).
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

    def to_ycc(self, x):
        parts = [self.ycc_conv(x[:, 3 * f:3 * f + 3]) for f in range(self.frames)]
        return torch.cat(parts, 1) if len(parts) > 1 else parts[0]

    def to_rgb(self, ycc):
        return self.rgb_conv(ycc)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 * self.frames or any(d % 2 for d in x.shape[-2:]):
            raise ValueError('even (B, 3*frames, H, W) input required')
        ycc = self.to_ycc(x)
        luma = torch.cat([ycc[:, 3 * f:3 * f + 1] for f in range(self.frames)], 1)
        luma_up = self.core(self.unshuffle(luma))
        chroma = ycc[:, 3 * (self.frames - 1) + 1:3 * (self.frames - 1) + 3]
        if self.chroma_mode == 'bicubic':
            chroma_up = F.interpolate(chroma, scale_factor=2, mode='bicubic',
                                      align_corners=False, antialias=True)
        else:
            chroma_up = F.interpolate(chroma, scale_factor=2, mode='bilinear',
                                      align_corners=False)
        return self.to_rgb(torch.cat((luma_up, chroma_up), 1))


def folded_rgb_first_weight(conv_weight, rgb2ycc_row):
    """Compose the fixed RGB->Y row through the unshuffle into one RGB conv.

    conv_weight is (out, 4*frames, k, k) on unshuffled luma; the result is
    (out, 12*frames, k, k) on unshuffled RGB with an identical response.

    Indexing follows PixelUnshuffle(2) exactly: unshuffled channel
    (in_channel * 4 + subpixel), so RGB input channel (f, c) subpixel k sits
    at ((f*3+c)*4+k) and luma input channel f subpixel k at (f*4+k).
    """
    out, packed, k, _ = conv_weight.shape
    if packed % 4:
        raise ValueError('unshuffled four-phase input required')
    frames = packed // 4
    row = rgb2ycc_row.reshape(-1).detach()
    folded = torch.zeros(out, packed * 3, k, k, dtype=conv_weight.dtype)
    for frame in range(frames):
        for sub in range(4):
            column = conv_weight[:, frame * 4 + sub]
            for c in range(3):
                folded[:, (frame * 3 + c) * 4 + sub] = column * row[c]
    return folded


def warm_start_from_widened(model, wide):
    """Project a widened 40ch RGB Unshuffled into the asymmetric backbone.

    Both models are fused first (the training harness fuses the SR backbone
    on wrap anyway, so the fused graph is exactly what trains). After fusion
    every trunk convolution is an ordinary conv with identical 40ch geometry
    on both sides and transfers verbatim; only the first convolution (fold
    RGB->Y over its input channels) and the upsampler (luma row over its
    output rows) are projected. A grayscale input then reproduces the widened
    backbone's luma response. NOT function-preserving on color input: the
    chroma path deliberately diverges. Stated, not hidden.
    """
    from architectures.subspace_adapter import fuse_convolutions
    from architectures.span_arch import Conv3XC
    if model.luma_channels != wide.core.conv_1.eval_conv.out_channels:
        raise ValueError('widened width must match the luma trunk width')
    m = torch.tensor(LUMA_WEIGHTS, dtype=torch.float32)
    for module in wide.modules():
        if isinstance(module, Conv3XC):
            module.update_params()
    fuse_convolutions(wide)
    fuse_convolutions(model)
    sources = dict(wide.named_modules())
    with torch.no_grad():
        for name, layer in model.named_modules():
            if name not in sources or not isinstance(layer, nn.Conv2d):
                continue
            src = sources[name]
            if not isinstance(src, nn.Conv2d):
                raise ValueError(f'fused-geometry mismatch at {name}')
            if name == 'core.conv_1':
                layer.weight.copy_(_fold_first(src.weight, m))
                if layer.bias is not None:
                    layer.bias.copy_(src.bias)
            elif 'upsampler.0' in name:
                _project_upsampler(layer, src, m)
            else:
                layer.weight.copy_(src.weight)
                if layer.bias is not None:
                    layer.bias.copy_(src.bias)
    return model


def _fold_first(weight, m=None):
    """Sum the widened first conv over RGB: (o, 12*frames, k, k) -> (o, 4*frames, k, k).

    PixelUnshuffle(2) orders channels channel-major: unshuffled index
    (in_channel * 4 + subpixel). RGB frame f, color c, subpixel k sits at
    ((f*3+c)*4+k); luma frame f, subpixel k at (f*4+k).

    The sum is UNWEIGHTED (m unused, kept for signature symmetry): on an
    achromatic input x_c = t the widened response is t*sum_c W_c and luma is
    t (luma weights sum to 1), so the uniform sum reproduces the widened
    first-layer response on the achromatic axis exactly. A luma-weighted sum
    would shrink it by sum_c m_c per channel and start the trunk off-scale.
    """
    out, packed, k, _ = weight.shape
    if packed % 12:
        raise ValueError('RGB unshuffled input required')
    frames = packed // 12
    folded = torch.zeros(out, packed // 3, k, k, dtype=weight.dtype)
    for frame in range(frames):
        for sub in range(4):
            acc = torch.zeros_like(folded[:, 0])
            for c in range(3):
                acc = acc + weight[:, (frame * 3 + c) * 4 + sub]
            folded[:, frame * 4 + sub] = acc
    return folded


def _project_upsampler(layer, src, m):
    """RGB 48-row upsampler -> Y 16-row upsampler via the luma row.

    On an achromatic input each widened RGB output equals t and the luma
    weights sum to 1, so the m-weighted row reproduces t with unit gain and
    matches the exact-inverse back-end. PixelShuffle(4) is channel-major: RGB
    row (c*16+s) carries color c at subpixel s, so luma row s is the
    m-weighted sum over c.
    """
    if layer.weight.shape[0] * 3 != src.weight.shape[0]:
        raise ValueError('3:1 RGB-to-luma upsampler geometry required')
    out = layer.weight.shape[0]
    folded = torch.zeros_like(layer.weight)
    for s in range(out):
        acc = torch.zeros_like(folded[0])
        for c in range(3):
            acc = acc + src.weight[c * out + s] * m[c]
        folded[s] = acc
    layer.weight.copy_(folded)
    if layer.bias is not None:
        bias = torch.zeros_like(layer.bias)
        for s in range(out):
            acc = torch.zeros((), dtype=src.bias.dtype)
            for c in range(3):
                acc = acc + src.bias[c * out + s] * m[c]
            bias[s] = acc
        layer.bias.copy_(bias)


class FullLRLumaSPAN(nn.Module):
    """Full-LR feature wrapper around a LumaAsymmetric backbone (arm C only).

    Mirrors architectures.full_lr_feature_span.FullLRFeatureSPAN line for
    line, except the trunk's direct path consumes UNSHUFFLED LUMA instead of
    unshuffled RGB. raw_encoder / decay / project / state update are
    identical (project is zero-init here too), so the ONLY difference between
    arm C and arm B is the asymmetry treatment itself. Arms A/B stay on the
    byte-identical shared wrapper; this class exists because the shared
    wrapper's `core.conv_1(self.sr.unshuffle(current))` call hard-codes a
    12-channel RGB trunk input.
    """

    requires_feature_state = True
    state_representation = 'full_lr_observation_features_v1'

    def __init__(self, sr, *, train_backbone=False, state_channels=8,
                 initial_decay_bias=-2.):
        super().__init__()
        import math
        if not math.isfinite(initial_decay_bias):
            raise ValueError('finite initial decay bias required')
        if type(sr) is not LumaAsymmetric:
            raise ValueError('luma-asymmetric backbone required')
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
        if self.sr.chroma_mode == 'bicubic':
            chroma_up = F.interpolate(chroma, scale_factor=2, mode='bicubic',
                                      align_corners=False, antialias=True)
        else:
            chroma_up = F.interpolate(chroma, scale_factor=2, mode='bilinear',
                                      align_corners=False)
        return self.sr.to_rgb(torch.cat((luma_up, chroma_up), 1)), state


def load_arm_checkpoint(path, device='cpu'):
    """Rebuild the FullLR-wrapped backbone a train_luma_arms checkpoint saved."""
    from architectures.full_lr_feature_span import FullLRFeatureSPAN
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('scale') != 2:
        raise ValueError('2x arm checkpoint required')
    backbone = checkpoint.get('backbone')
    args = checkpoint.get('experiment', {}).get('args', {})
    if backbone == 'luma40':
        channels = checkpoint.get('luma_channels')
        if type(channels) is not int or channels != args.get('luma_channels'):
            raise ValueError('luma channels differ from training declaration')
        sr = LumaAsymmetric(channels, scale=2,
                            version=checkpoint.get('version', 1))
        model = FullLRLumaSPAN(sr, state_channels=checkpoint.get('state_channels', 8))
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
