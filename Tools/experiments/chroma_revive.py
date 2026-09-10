#!/usr/bin/env python3
"""r97: chroma revival -- true-identity wrap + deeper, multi-scale chroma
branch. Research only.

r86 proved the learned chroma path carries enormous quality (destroying it
costs +45.5% LPIPS / +82.8% DISTS) and r82 found chroma reconstruction error
3.3x luma, so chroma is the clearest remaining capacity-reallocation target.
The first attempt (`chroma_inject.py`'s `ChromaInject` + `train_chroma_arms.py`
arm `chroma40`, 2k steps) came back NULL/worse: +6.46% LPIPS / +2.0% DISTS
vs the matched `width_40` control. Two diagnosed causes:

  1. `warm_start_chroma_inject` is NOT function-preserving on color input.
     `ChromaInject` builds a fresh Y-only SPAN trunk warm-started from a
     widened RGB init (achromatic axis only), then bolts on a chroma branch
     that is a zero-init residual on a BICUBIC/BILINEAR-upsampled LR chroma
     path. At step 0 this is strictly worse than the trained control's own
     (good, learned) chroma: the model starts ~3.2% behind before a single
     gradient step, so the arm has to spend its 2000-step budget recovering
     ground it never needed to give up.
  2. The chroma branch was a SINGLE Conv3XC fed a SINGLE conv_1 re-entry --
     too little capacity to both recover from (1) and learn a materially
     better cross-component mapping in the same budget.

This module fixes both, using r94's (`nearpixel_span.py`) proven
de-risking technique -- wrap the ALREADY-TRAINED model and make the new path
an exact identity/zero at construction, rather than building a fresh
architecture with an approximate warm start:

  Fix 1 (true identity): `ChromaReviveSPAN` does not rebuild a trunk at all.
  It wraps a trained, unmodified `Unshuffled` backbone directly (a hand
  replay of `SPAN.forward` -- see `chroma_trunk_replay` -- exactly like
  `nearpixel_span.trunk_pre_shuffle`, kept in sync by a regression test) and
  takes the residual base as the MODEL'S OWN chroma output (`to_ycc(model
  output)`), never a bicubic/bilinear LR upsample. The new branch's delta is
  zero-initialized (see `ChromaReviveBranch`), so at construction
  `ChromaReviveSPAN(model)(x)` reproduces `model(x)` up to the RGB<->YCbCr
  matrix-inverse rounding floor (~1e-5, the same bound `chroma_inject.py`'s
  own `test_rgb_to_ycbcr_roundtrip_is_exact` uses) -- not "starts behind",
  bit-exact at step 0, in torch AND after Core ML conversion on real frames.
  This is strictly stronger de-risking than the original design, and it
  means `ChromaReviveSPAN` vs a plain, UNMODIFIED pretrained control isolates
  the branch's effect with zero other confounds (no widening involved at
  all) -- a cleaner ablation than the original chroma40-vs-width_40 pair.

  Fix 2 (more capacity, multi-scale): `ChromaReviveBranch` is a stack of
  `branch_depth` (default 3) Conv3XC blocks -- materially deeper than one --
  fed the concatenation of the native LR chroma plus THREE depths of the
  trunk's own quarter-resolution feature stack (shallow/post-conv_1, deep/
  post-conv_2, and the pre-PixelShuffle decision-level features) upsampled to
  the chroma grid, PLUS the model's own final luma output downsampled back
  to that grid -- a genuinely different, post-decision, full-resolution cue
  that none of the pre-decision trunk features carry. Only the branch's LAST
  Conv3XC is zero-initialized (r94's single-zero-point technique): earlier
  layers keep Conv3XC's normal, non-degenerate default init, so the branch
  has live internal capacity and a real gradient at the final layer from
  step 1 (earlier layers pick up gradient from step 2 onward, once the final
  layer is no longer exactly zero -- the standard zero-init-residual warm-up,
  not a dead network).

The chroma path is bilinear throughout (coremltools implements no bicubic
upsample; the original design's hardcoded-bicubic mistake is not repeated
here) and uses only plain/depthwise/pointwise convs and `F.interpolate` --
no gather, no index_select, no data-dependent control flow.

TORCH-ONLY. Torch-level gains here have repeatedly failed to survive into
the deployed native pipeline. Nothing in this file is a deployable-gain
claim; init-parity and convertibility are engineering properties, not
quality claims.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from luma_asym import _buffers


def _zero_init_conv3xc(block):
    """Zero every Conv2d inside a Conv3XC (the training-mode conv+sk path
    AND the cached eval_conv), then refresh the eval-mode cache so the block
    outputs exactly zero regardless of `.training`."""
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
    update = getattr(block, 'update_params', None)
    if update is not None:
        update()


def _sync(module):
    """Refresh a Conv3XC's fused eval params before a standalone call.

    No-op once `fuse_convolutions` has replaced the module with a plain
    Conv2d (no `update_params`), matching the convention already used by
    `chroma_inject.py` and `luma_asym.py` for the same situation.
    """
    update = getattr(module, 'update_params', None)
    if update is not None:
        update()


def chroma_trunk_replay(core, x):
    """Hand-written replay of `SPAN.forward` (`span_arch.py`), capturing
    three depths of the trunk's own quarter-resolution feature stack instead
    of only the final output. Existing, UNMODIFIED, pretrained submodules
    only -- nothing here is new or reinitialized. Kept in sync with
    `SPAN.forward` by
    `test_chroma_revive.py::test_trunk_replay_matches_span_forward`.

    Args:
      core: a `span_arch.SPAN` instance (version 1 topology).
      x: unshuffled input, `(B, 12*frames, H/2, W/2)` for an RGB `Unshuffled`
         trunk (matches `core.conv_1`'s expected input channels).

    Returns a dict:
      shallow:     core.conv_1 output               (B, width, H/2, W/2)
      deep:        core.conv_2 output (post block_6) (B, width, H/2, W/2)
      pre_shuffle: pre-PixelShuffle decision features
                   (B, 3*factor**2, H/2, W/2)
      output:      the trunk's own final output, bit-exact with `core(x)`
                   (B, 3, H/2*factor, W/2*factor)
    """
    xin = (x - core.mean) * core.img_range
    out_feature = core.conv_1(xin)
    out_b1, _, _ = core.block_1(out_feature)
    out_b2, _, _ = core.block_2(out_b1)
    out_b3, _, _ = core.block_3(out_b2)
    out_b4, _, _ = core.block_4(out_b3)
    out_b5, _, _ = core.block_5(out_b4)
    out_b6, out_b5_2, _ = core.block_6(out_b5)
    deep = core.conv_2(out_b6)
    pre_cat = core.conv_cat(torch.cat([out_feature, deep, out_b1, out_b5_2], 1))
    pre_shuffle = core.upsampler[0](pre_cat)
    output = core.upsampler[1](pre_shuffle)
    return {'shallow': out_feature, 'deep': deep, 'pre_shuffle': pre_shuffle, 'output': output}


class ChromaReviveBranch(nn.Module):
    """Deep, multi-scale chroma-refinement head (see module docstring).

    Input channels: 2 (native LR chroma) + `shallow_ch` + `deep_ch` +
    `pre_shuffle_ch` (three depths of the trunk's own quarter-resolution
    feature stack, bilinear-upsampled to the chroma grid) + 1 (the model's
    own final luma output, bilinear-downsampled back to the chroma grid).
    Output: 2-channel chroma delta at the same LR chroma resolution, exactly
    zero at construction.
    """

    def __init__(self, in_channels, width=32, depth=3):
        super().__init__()
        if depth < 2:
            raise ValueError('materially deeper than one Conv3XC required (depth >= 2)')
        if width < 8:
            raise ValueError('positive branch width required')
        if in_channels < 3:
            raise ValueError('at least chroma plus one feature cue required')
        from architectures.span_arch import Conv3XC
        self.depth = depth
        self.width = width
        self.in_channels = in_channels
        stem = []
        c = in_channels
        for _ in range(depth - 1):
            stem.append(Conv3XC(c, width, gain1=2, s=1, relu=True))
            c = width
        self.stem = nn.ModuleList(stem)
        self.project = Conv3XC(c, 2, gain1=1, s=1)
        _zero_init_conv3xc(self.project)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError('branch input channel count mismatch')
        for layer in self.stem:
            _sync(layer)
            x = layer(x)
        _sync(self.project)
        return self.project(x)


class ChromaReviveSPAN(nn.Module):
    """Bit-exact-at-construction wrap of a trained single-frame `Unshuffled`
    SPAN backbone, adding a deep, multi-scale chroma-revival branch.

    `ChromaReviveSPAN(model)(x)` reproduces `model(x)` at construction (see
    `test_chroma_revive.py::test_wrapped_model_matches_control_at_init`);
    `branch` is the only free-standing new module, and only its LAST layer
    starts at zero (see `ChromaReviveBranch`).
    """

    def __init__(self, model, branch_channels=32, branch_depth=3):
        super().__init__()
        if model.frames != 1:
            raise ValueError('single-frame backbone required for the standalone wrap')
        if model.version != 1:
            raise ValueError('only the v1 SPAN trunk is supported here')
        self.model = model
        self.version = model.version
        self.frames = model.frames
        self.unshuffle = model.unshuffle
        factor = model.core.upsampler[1].upscale_factor
        if factor % 2:
            raise ValueError('trunk PixelShuffle factor must be even (= 2 * scale)')
        fwd, fbias, bwd, bbias = _buffers()
        self.register_buffer('rgb2ycc', fwd)
        self.register_buffer('ycc_bias', fbias)
        self.register_buffer('ycc2rgb', bwd)
        self.register_buffer('rgb_bias', bbias)
        # Trace-safe color maps: fixed 1x1 convolutions, matching the
        # convention in luma_asym.py / chroma_inject.py so no einsum/reshape
        # on a runtime-derived shape ever enters a traced CoreML graph.
        self.ycc_conv = nn.Conv2d(3, 3, 1)
        self.ycc_conv.weight.data.copy_(fwd.view(3, 3, 1, 1))
        self.ycc_conv.bias.data.copy_(fbias)
        # Full (bias-carrying) inverse map, kept ONLY for
        # test_chroma_revive.py's linearity-equivalence check -- NOT used in
        # forward(). See chroma_delta_to_rgb below for why.
        self.rgb_conv = nn.Conv2d(3, 3, 1)
        self.rgb_conv.weight.data.copy_(bwd.view(3, 3, 1, 1))
        self.rgb_conv.bias.data.copy_(bbias)
        # The output reconstruction never rebuilds RGB from a full YCbCr
        # vector (that first CoreML gate attempt measured a real ~0.05-mean/
        # 1.0-max NATIVE init-parity gap on real frames, traced to fp16
        # quantization of the RGB<->YCbCr roundtrip -- an error source the
        # control path never goes through at all). Because to_rgb is affine
        # (to_rgb(v) = M @ v + b), for any chroma delta d,
        # to_rgb(cat(Y, Cb+d_cb, Cr+d_cr)) - to_rgb(cat(Y, Cb, Cr)) == M[:,1:] @ d
        # EXACTLY (the bias cancels), so adding that linear-only delta
        # directly to the trunk's own RGB output is mathematically identical
        # to the full roundtrip for any delta, and at construction (delta
        # exactly 0) it makes the identity property precision-proof: conv
        # weights are irrelevant when the input is exactly 0 in ANY float
        # format, since 0 * finite_weight == 0 exactly (no rounding).
        self.chroma_delta_to_rgb = nn.Conv2d(2, 3, 1, bias=False)
        self.chroma_delta_to_rgb.weight.data.copy_(bwd[:, 1:].reshape(3, 2, 1, 1))
        for module in (self.ycc_conv, self.rgb_conv, self.chroma_delta_to_rgb):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 16, 16)
            feats = chroma_trunk_replay(model.core, model.unshuffle(probe))
        shallow_ch = feats['shallow'].shape[1]
        deep_ch = feats['deep'].shape[1]
        pre_shuffle_ch = feats['pre_shuffle'].shape[1]
        self.branch_in_channels = 2 + shallow_ch + deep_ch + pre_shuffle_ch + 1
        self.branch = ChromaReviveBranch(self.branch_in_channels, width=branch_channels, depth=branch_depth)

    @property
    def core(self):
        """Read-only passthrough (same convention as `nearpixel_span.NearPixelSPAN.core`)
        so generic trainer diagnostics that read `model.core...` keep working
        unmodified against the wrapped model, without double-registering the
        SPAN trunk as a submodule under two different attribute names."""
        return self.model.core

    def to_ycc(self, x):
        return self.ycc_conv(x)

    def to_rgb(self, ycc):
        """Full YCbCr->RGB map. Kept for the linearity-equivalence test only
        -- forward() uses `chroma_delta_to_rgb` instead (see its docstring)."""
        return self.rgb_conv(ycc)

    def forward(self, x, return_planes=False):
        if x.ndim != 4 or x.shape[1] != 3 or any(d % 2 for d in x.shape[-2:]):
            raise ValueError('even (B, 3, H, W) single-frame RGB input required')
        unshuffled = self.model.unshuffle(x)
        feats = chroma_trunk_replay(self.model.core, unshuffled)
        rgb_out = feats['output']  # bit-exact with self.model(x)
        # The branch runs at the trunk's own native quarter-resolution grid
        # (shallow/deep/pre_shuffle already live there -- no upsampling
        # needed for them), not at the full LR chroma resolution: an earlier
        # full-LR-resolution version of this branch cost +87.5% MACs (the
        # branch alone approached the control's entire compute budget); the
        # quarter-res version below is the shipped design (see the README's
        # cost section for the measured comparison).
        quarter_size = feats['shallow'].shape[-2:]
        chroma_lr = self.to_ycc(x)[:, 1:]
        chroma_quarter = F.interpolate(chroma_lr, size=quarter_size, mode='bilinear', align_corners=False)
        y_full = self.to_ycc(rgb_out)[:, :1]
        y_quarter = F.interpolate(y_full, size=quarter_size, mode='bilinear', align_corners=False)
        branch_in = torch.cat((chroma_quarter, feats['shallow'], feats['deep'], feats['pre_shuffle'], y_quarter), dim=1)
        delta_quarter = self.branch(branch_in)
        delta_up = F.interpolate(delta_quarter, size=rgb_out.shape[-2:], mode='bilinear', align_corners=False)
        rgb_delta = self.chroma_delta_to_rgb(delta_up)
        out = rgb_out + rgb_delta
        if return_planes:
            return out, {'rgb_out': rgb_out, 'delta_quarter': delta_quarter,
                        'delta_up': delta_up, 'rgb_delta': rgb_delta}
        return out


class FullLRChromaReviveSPAN(nn.Module):
    """Full-LR feature wrapper around a `ChromaReviveSPAN` backbone.

    Mirrors `chroma_inject.FullLRChromaInjectSPAN` / `luma_asym.FullLRLumaSPAN`
    line for line for the temporal state machinery (raw_encoder / decay /
    project / state update are identical, and `project` is zero-init here
    too -- injecting state changes nothing at construction). The only
    architectural difference from those two wrappers is that this one calls
    into `ChromaReviveSPAN`'s own multi-scale replay instead of a single
    conv_1 re-entry or a fixed bicubic chroma path.

    `branch_parameters()` deliberately includes `self.sr.branch` (the new
    chroma-revival branch) alongside the wrapper's own new state machinery,
    so it trains at the FAST branch learning rate (2e-4 in
    `train_chroma_arms.py`) rather than the slow joint-backbone rate
    (2e-5) reserved for fine-tuning the already-good pretrained trunk. In
    the CLOSED chroma40 arm, the new chroma branch's parameters lived inside
    `model.sr.parameters()` and so only ever trained at the slow joint rate
    if `--joint` was passed -- a brand-new, randomly-relevant set of weights
    given the same conservative rate designed for a trunk that is already
    correct. This is a plausible third contributing factor to that null,
    alongside the two diagnosed causes this module fixes directly; giving
    the branch its own fast-rate group is a natural, low-risk correction
    that requires no change to the training recipe itself (see
    `train_chroma_arms.py`'s arm-conditional optimizer group scoping).
    """

    requires_feature_state = True
    state_representation = 'full_lr_observation_features_v1'

    def __init__(self, sr, *, train_backbone=False, state_channels=8,
                 initial_decay_bias=-2.):
        super().__init__()
        if not math.isfinite(initial_decay_bias):
            raise ValueError('finite initial decay bias required')
        if type(sr) is not ChromaReviveSPAN:
            raise ValueError('chroma-revive backbone required')
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
        # The chroma-revival branch must always remain trainable regardless
        # of `train_backbone` (--joint): it is new, never-before-trained
        # capacity, not part of the pretrained trunk `train_backbone` gates.
        self.sr.branch.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.sr.eval()
        return self

    def branch_parameters(self):
        for module in (self.raw_encoder, self.decay, self.project, self.sr.branch):
            yield from module.parameters()

    def backbone_parameters(self):
        """The pretrained trunk only (excludes the chroma-revival branch,
        which is scoped separately by `branch_parameters()` so it trains at
        the branch learning rate rather than the slow joint-backbone rate --
        see the class docstring)."""
        return self.sr.model.parameters()

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
        unshuffled = self.sr.unshuffle(current)
        xin = (unshuffled - core.mean) * core.img_range
        out_feature = core.conv_1(xin) + self.project(F.pixel_unshuffle(state, 2))
        out_b1, _, _ = core.block_1(out_feature)
        out_b2, _, _ = core.block_2(out_b1)
        out_b3, _, _ = core.block_3(out_b2)
        out_b4, _, _ = core.block_4(out_b3)
        out_b5, _, _ = core.block_5(out_b4)
        out_b6, out_b5_2, _ = core.block_6(out_b5)
        deep = core.conv_2(out_b6)
        pre_cat = core.conv_cat(torch.cat([out_feature, deep, out_b1, out_b5_2], 1))
        pre_shuffle = core.upsampler[0](pre_cat)
        # core.upsampler[1] (PixelShuffle) yields RGB, not YCbCr -- must go
        # through to_ycc before slicing luma/chroma (a bare [:, :1] here
        # would silently take the raw R channel as if it were Y).
        rgb_full = core.upsampler[1](pre_shuffle)
        # State injection lands only on the luma-domain trunk (matches the
        # existing FullLR convention in full_lr_feature_span.py /
        # chroma_inject.FullLRChromaInjectSPAN / luma_asym.FullLRLumaSPAN);
        # the chroma branch runs at the trunk's own quarter-resolution grid,
        # exactly as ChromaReviveSPAN.forward does for the single-frame case
        # (see that method's docstring for the cost rationale), and the
        # output is reconstructed via the same precision-proof RGB-space
        # delta (see `chroma_delta_to_rgb`'s docstring) rather than a full
        # YCbCr roundtrip.
        quarter_size = out_feature.shape[-2:]
        chroma_lr = self.sr.to_ycc(current)[:, 1:]
        chroma_quarter = F.interpolate(chroma_lr, size=quarter_size, mode='bilinear', align_corners=False)
        y_full = self.sr.to_ycc(rgb_full)[:, :1]
        y_quarter = F.interpolate(y_full, size=quarter_size, mode='bilinear', align_corners=False)
        branch_in = torch.cat((chroma_quarter, out_feature, deep, pre_shuffle, y_quarter), dim=1)
        delta_quarter = self.sr.branch(branch_in)
        delta_up = F.interpolate(delta_quarter, size=rgb_full.shape[-2:], mode='bilinear', align_corners=False)
        rgb_delta = self.sr.chroma_delta_to_rgb(delta_up)
        output = rgb_full + rgb_delta
        return output, state


def report_params(model_frames1):
    """Parameter accounting (real module, not a formula) for a fresh
    `ChromaReviveSPAN(model_frames1)`. MAC accounting is measured with real
    forward hooks in `r97_convert_gate.py` against the actual shipping
    checkpoint, per the task's own "forward-hook measured, like r94"
    requirement -- not duplicated here as a static estimate.
    """
    wrapped = ChromaReviveSPAN(model_frames1)
    control_params = sum(p.numel() for p in model_frames1.parameters())
    added_trainable = sum(p.numel() for p in wrapped.branch.parameters())
    added_frozen = (sum(p.numel() for p in wrapped.ycc_conv.parameters())
                    + sum(p.numel() for p in wrapped.rgb_conv.parameters()))
    return {'control_total': control_params,
            'branch_in_channels': wrapped.branch_in_channels,
            'branch_width': wrapped.branch.width, 'branch_depth': wrapped.branch.depth,
            'added_trainable': added_trainable, 'added_frozen_colormap': added_frozen,
            'added_trainable_fraction_of_control': added_trainable / control_params}


def load_arm_checkpoint(path, device='cpu'):
    """Rebuild the FullLR-wrapped backbone a train_chroma_arms 'chroma_revive'
    checkpoint saved (matches the `load_arm_checkpoint` convention already
    established by `chroma_inject.py` / `luma_asym.py`)."""
    from architectures.full_lr_feature_span import FullLRFeatureSPAN
    from train_span import Unshuffled
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('scale') != 2:
        raise ValueError('2x arm checkpoint required')
    backbone = checkpoint.get('backbone')
    args = checkpoint.get('experiment', {}).get('args', {})
    if backbone == 'chroma_revive':
        channels = checkpoint.get('channels')
        if type(channels) is not int or channels != args.get('backbone_channels'):
            raise ValueError('backbone channels differ from training declaration')
        base = Unshuffled(channels, scale=2, version=checkpoint.get('version', 1))
        sr = ChromaReviveSPAN(base, branch_channels=checkpoint.get('branch_channels', 32),
                              branch_depth=checkpoint.get('branch_depth', 3))
        model = FullLRChromaReviveSPAN(sr, state_channels=checkpoint.get('state_channels', 8))
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
