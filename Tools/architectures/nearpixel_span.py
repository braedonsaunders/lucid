"""SPANV2-style concatenated near-pixel branch for Lucid's Unshuffled head.

r90 (`.build/quality-breakthrough-r90-ideamine/README.md`, Rank 2) proposes
porting SPANV2's (NTIRE 2026 ESR winner, XiaomiMM, arXiv:2604.03198v1) parallel
"near-pixel" branch: a depthwise conv initialized to copy the center pixel
(nearest-neighbor upsampling at init) that runs alongside the SPAN trunk and is
CONCATENATED with the trunk's own features before the final fusion + pixel
shuffle. Idea 1 (thr_26vvpn49ph, CLOSED NULL) tried an ADDITIVE identity skip
instead and lost because a near-identity additive init forces the trunk's own
head to be zeroed, and that head never recovers at the fine-tune LR (norm 0.19
vs control 6.32 after 2000 steps). Concatenation is supposed to sidestep this:
nothing pretrained has to be zeroed, because the new path is genuinely separate.

Deriving s (the near-pixel PixelShuffle factor) for Lucid, concretely
-----------------------------------------------------------------------------
Lucid's model is NOT plain SPAN. `Unshuffled` (Tools/train_span.py) does:

    unshuffled = nn.PixelUnshuffle(2)(x)                      # x: (B,3,H,W)
    output      = SPAN(..., upscale=scale*2)(unshuffled)      # -> (B,3,2H,2W) for scale=2

i.e. the trunk runs at quarter area and its own PixelShuffle factor is
`scale*2`, not `scale`. For the shipping "big2k" checkpoint (verified by
loading `.build/paired-ladder-r10/big_2k_raw.pth`: channels=32, scale=2,
frames=1), the net upscale is 2x and the trunk's PixelShuffle factor
(`core.upsampler[1].upscale_factor`) is **4**, confirmed by reading
`skip_residual_span.py`'s own `scale=model.core.upsampler[1].upscale_factor // 2`
convention.

A naive port of SPANV2 would put a depthwise conv straight on the 3-channel LR
image with s = the *net* upscale (2 for big2k) - that produces a PixelShuffle
(2)-ready 3*2^2=12-channel tensor at full LR resolution (H, W), which cannot be
concatenated with the trunk's own pre-shuffle features: those live at quarter
resolution (H/2, W/2) with 3*4^2=48 channels (because the trunk's own
PixelShuffle factor is 4, not 2 - see `pixelshuffle_block` in span_arch.py,
`upsampler[0]` is `conv_layer(feature_channels, out_channels*upscale_factor**2)`).
So the correct choice is **s = the trunk's PixelShuffle factor (4 for big2k),
applied to the pixel-UNshuffled current-frame RGB**, not s = net scale applied
to raw RGB. That keeps the near-pixel branch living at the same (H/2, W/2)
grid and the same 48-channel-per-3 layout as the trunk head, so the two can be
concatenated and pass through one shared, unmodified `PixelShuffle(4)`.

This is verified below by direct derivation of the required channel mapping,
not by producing something that merely "looks about right": `NearPixelBranch`
copies each of the 12 pixel-unshuffled input channels into the correct
subset of the 48 output channels such that `PixelShuffle(4)` of its output is
*bit-exact* nearest-neighbor 2x upsampling of the original image (see
`test_nearpixel_span.py::test_near_pixel_branch_matches_nearest_neighbor`).

Concatenation point and the "nothing to unlearn" property
-----------------------------------------------------------------------------
r90's literal description concatenates raw 32-channel trunk features with the
48-channel near-pixel features, then fuses 80 -> 48 with a fresh depthwise +
pointwise pair. That fresh pointwise conv has no principled non-random init,
so the wrapped model does NOT start bit-identical to the control model - a
weaker de-risking property than Idea 1's own diagnosis wants.

This module concatenates one level later instead: both branches are already
projected to PixelShuffle(4)-ready 48-channel tensors (trunk via its existing,
UNMODIFIED `upsampler[0]` conv; near-pixel via `NearPixelBranch`) before
concatenation. That lets the new fusion pair (`depthwise_fuse` +
`pointwise_fuse`) be initialized as an exact identity on the trunk half and an
exact zero on the near-pixel half, so `NearPixelSPAN(model)(x)` is bit-exact
with `model(x)` at construction time -- strictly stronger than "the trunk head
is untouched": the *entire model* is untouched at init, and only fresh,
never-before-trained parameters (near_branch's copy-initialized depthwise conv,
and the fusion pair) are free to move. See
`test_nearpixel_span.py::test_wrapped_model_matches_control_at_init`.

This is a deliberate, documented adaptation of r90's literal design, not a
guess: the trunk-facing 32ch vs 48ch concatenation point is an implementation
choice the source idea does not fix, and this choice was made specifically to
satisfy the de-risking goal from the CLOSED Idea 1 finding.
"""
import torch
from torch import nn


class NearPixelBranch(nn.Module):
    """Depthwise pixel-repeat prior over a PixelUnshuffle(2)'d RGB frame.

    Input: (B, 12, H/2, W/2) - the output of `nn.PixelUnshuffle(2)` applied to
    a (B, 3, H, W) RGB frame (Lucid's own stem; see `Unshuffled.unshuffle`).
    Output: (B, 3*F*F, H/2, W/2), PixelShuffle(F)-ready, F = 2 * scale.

    At init, `nn.PixelShuffle(F)(branch(unshuffle(x)))` is exactly
    `F.interpolate(x, scale_factor=scale, mode="nearest")` - see the unit test.
    Two sub-steps, both plain dense/depthwise convs so there is no gather or
    fancy-indexing op in the graph (a real Core ML conversion risk that a
    grouped-conv-plus-channel-permute design would carry; a fixed permutation
    expressed as a frozen 1x1 conv is unambiguously convertible instead):

      conv_near: true depthwise (groups=12) 3x3, 12 -> 12*scale*scale
                 channels. Center-tap-1 / else-0 init: for each of the 12
                 unshuffled input channels, every one of its `scale*scale`
                 output channels starts as an exact copy of that channel
                 (nearest-neighbor repeat at the pixel-unshuffle sub-position
                 granularity). Trainable.
      perm:      frozen (non-trainable) 1x1 conv, a permutation matrix that
                 reorders conv_near's PyTorch-grouped-conv channel layout into
                 the interleaved layout `nn.PixelShuffle` expects. Carries no
                 learned content; it exists purely so the *value* placed at
                 each output channel by conv_near lands at the position
                 PixelShuffle will read as the correct (row, col) offset.
    """

    def __init__(self, scale):
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.scale = scale
        self.factor = 2 * scale  # trunk's own PixelShuffle factor
        reps = scale * scale
        self.conv_near = nn.Conv2d(12, 12 * reps, 3, padding=1, groups=12, bias=False)
        with torch.no_grad():
            self.conv_near.weight.zero_()
            self.conv_near.weight[:, 0, 1, 1] = 1.0

        self.perm = nn.Conv2d(12 * reps, 3 * self.factor * self.factor, 1, bias=False)
        with torch.no_grad():
            self.perm.weight.zero_()
            for c in range(3):
                for prow in range(2):
                    for pcol in range(2):
                        k = c * 4 + prow * 2 + pcol
                        for mrow in range(scale):
                            for mcol in range(scale):
                                m = mrow * scale + mcol
                                src_idx = k * reps + m
                                di = prow * scale + mrow
                                dj = pcol * scale + mcol
                                dst_idx = c * self.factor * self.factor + di * self.factor + dj
                                self.perm.weight[dst_idx, src_idx, 0, 0] = 1.0
        for p in self.perm.parameters():
            p.requires_grad_(False)

    def forward(self, unshuffled_rgb):
        if unshuffled_rgb.shape[1] != 12:
            raise ValueError("expected a PixelUnshuffle(2)'d RGB frame (12 channels)")
        return self.perm(self.conv_near(unshuffled_rgb))


def trunk_pre_shuffle(core, unshuffled_x):
    """Replays `SPAN.forward` (span_arch.py) up to, and including,
    `upsampler[0]` - the conv that produces PixelShuffle-ready features -
    but stops before `upsampler[1]` (the PixelShuffle itself). Existing,
    UNMODIFIED, pretrained submodules only; nothing here is new or reinitialized.
    Kept in sync with `SPAN.forward` by
    `test_nearpixel_span.py::test_trunk_pre_shuffle_matches_span_forward`.
    """
    core.mean = core.mean.type_as(unshuffled_x)  # SPAN.forward's own lazy device/dtype move (span_arch.py:227); core.mean is a plain tensor, not a registered buffer, so .cuda() never touches it
    x = (unshuffled_x - core.mean) * core.img_range
    out_feature = core.conv_1(x)
    out_b1, _, _ = core.block_1(out_feature)
    out_b2, _, _ = core.block_2(out_b1)
    out_b3, _, _ = core.block_3(out_b2)
    out_b4, _, _ = core.block_4(out_b3)
    out_b5, _, _ = core.block_5(out_b4)
    out_b6, out_b5_2, _ = core.block_6(out_b5)
    out_b6 = core.conv_2(out_b6)
    out = core.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
    return core.upsampler[0](out)


class NearPixelSPAN(nn.Module):
    """Wraps a trained `Unshuffled` model with a concatenated near-pixel branch.

    `NearPixelSPAN(model)(x)` is bit-exact with `model(x)` at construction
    (see the init-parity test); `depthwise_fuse` and `pointwise_fuse` are new,
    never-before-trained parameters free to move from that starting point,
    and `near_branch.conv_near` is a fresh, copy-initialized (not zeroed)
    trainable branch. Nothing pretrained is destroyed or zeroed.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.version = model.version
        core = model.core
        self.pixel_shuffle = core.upsampler[1]
        factor = self.pixel_shuffle.upscale_factor
        if factor % 2 != 0:
            raise ValueError("trunk PixelShuffle factor must be even (= 2 * scale)")
        scale = factor // 2
        self.near_branch = NearPixelBranch(scale)

        trunk_ch = 3 * factor * factor
        near_ch = 3 * factor * factor
        concat_ch = trunk_ch + near_ch
        self.trunk_ch = trunk_ch

        self.depthwise_fuse = nn.Conv2d(concat_ch, concat_ch, 3, padding=1,
                                        groups=concat_ch, bias=True)
        self.pointwise_fuse = nn.Conv2d(concat_ch, trunk_ch, 1, bias=True)
        with torch.no_grad():
            self.depthwise_fuse.weight.zero_()
            self.depthwise_fuse.weight[:, 0, 1, 1] = 1.0
            self.depthwise_fuse.bias.zero_()
            self.pointwise_fuse.weight.zero_()
            self.pointwise_fuse.weight[:trunk_ch, :trunk_ch, 0, 0] = torch.eye(trunk_ch)
            self.pointwise_fuse.bias.zero_()

    @property
    def core(self):
        """Read-only passthrough so generic trainer diagnostics (e.g. the
        adversarial head-gradient-ratio print, which reads `model.core...`
        for every architecture that isn't anchored_detail/anchored_lowpass
        or subspace) keep working unmodified against the wrapped model."""
        return self.model.core

    def forward(self, x):
        if x.shape[1] < 3:
            raise ValueError("expected at least one RGB frame's worth of channels")
        current_rgb = x[:, -3:]
        unshuffled = self.model.unshuffle(x)
        pre_shuffle_trunk = trunk_pre_shuffle(self.model.core, unshuffled)
        near_input = unshuffled[:, -12:]  # current frame's own unshuffled channels
        near = self.near_branch(near_input)
        concat = torch.cat([pre_shuffle_trunk, near], dim=1)
        fused = self.pointwise_fuse(self.depthwise_fuse(concat))
        return self.pixel_shuffle(fused)

    def fusion_weight_summary(self):
        """L1 norm of the pointwise fusion weight columns feeding from the
        near-pixel branch vs from the trunk. Used to check the mechanism
        actually engaged during training (r90's falsification criterion):
        if the near-pixel columns stay near their zero init, the branch was
        never used regardless of what the loss curve says."""
        w = self.pointwise_fuse.weight.detach()
        trunk_w = w[:, :self.trunk_ch]
        near_w = w[:, self.trunk_ch:]
        return {
            "trunk_l1": float(trunk_w.abs().sum()),
            "near_pixel_l1": float(near_w.abs().sum()),
            "near_pixel_fraction": float(near_w.abs().sum() / (trunk_w.abs().sum() + near_w.abs().sum() + 1e-12)),
        }


def from_unshuffled(model):
    """Small convenience wrapper matching the `from_unshuffled` naming
    convention used by `skip_residual_span.py` for the additive-skip variant."""
    return NearPixelSPAN(model)
