"""A causal, scale-parameterised reconstruction trunk derived from NanoVSR.

Upstream: https://github.com/filippawlicki/nanovsr (MIT, (c) 2026 Filip
Pawlicki, Marcel Kanduka, Marcin Pucek, Kamil Dobies), paper
https://arxiv.org/html/2607.10495v1. The RepVGG reparameterisation and the
bilinear-base residual are theirs; the causal single-branch operation, the
configurable scale and the Core ML conversion path are this file's adaptation.
No upstream weights are used - this trunk is trained from scratch on Lucid's
own bank with Lucid's own recipe, which is the entire point.

**Why this exists.** Every architecture evaluation in this repo - NanoVSR,
PLKSR-Rep, PiperSR, EfRLFN - scored somebody else's *pretrained checkpoint*
against `lucidbig2k`, and `lucidbig2k` carries this project's reference-target
recipe and paired-DINO critic, worth 10-15% LPIPS over an L1 baseline by the
paired-ladder ledger. So each comparison handed the home model a large recipe
advantage and then asked the visitor to clear a 3% bar. NanoVSR's pretrained
bidirectional model came within 0.7 points of that bar anyway (+2.34% LPIPS /
+2.84% DISTS) and was recorded as "Reject". That is not evidence about the
architecture; it is evidence about whose recipe the weights came from.

Architecture is therefore a variable this repo has never actually varied. This
makes it one.

**Three deliberate departures from upstream, each for a stated reason:**

1. **Causal.** Upstream runs a forward *and* a backward recurrent pass and
   fuses them, so every output frame depends on future frames. That is
   disqualifying for live playback, and the repo's own protocol note says so.
   Only the forward branch is kept.

2. **Configurable scale.** Upstream is fixed 4x (two PixelShuffle stages).
   Lucid ships 2x, and the matched control (`ch32/b6`) is 2x, so the upsampler
   is built from `scale` and the bilinear base uses the same factor.

3. **Optional pixel-unshuffle.** Upstream runs its trunk at full LR
   resolution; SPAN's `Unshuffled` runs at quarter area. At equal channels and
   blocks that is roughly a 4x compute difference, so a parameter-matched
   comparison would be a latency-mismatched one. `unshuffle=True` puts this
   trunk on the same footing, and the honest budget-matched configuration is
   chosen by measuring Core ML latency rather than by counting parameters -
   the discipline r105 established.

`frames` is accepted so a single-frame arm can be compared against `ch32/b6`
with the recurrence removed as a confound; the recurrent state is what a later
temporal arm would switch on, and is what upstream's value is really about.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RepBlock(nn.Module):
    """RepVGG block: 3x3 + 1x1 + identity at train time, one 3x3 at deploy.

    Structurally the same trick as SPAN's Conv3XC - extra training-time
    branches that fold away - but with a different branch set and a LeakyReLU
    rather than SPAN's SiLU/sigmoid attention. Keeping the fold means the
    exported graph is dense 3x3 convolutions, which is what r69 measured the
    Core ML path to be fastest on.
    """

    def __init__(self, in_channels, out_channels, deploy=False, norm='batch'):
        super().__init__()
        if norm not in ('batch', 'none'):
            raise ValueError("norm must be 'batch' or 'none'")
        self.in_channels, self.out_channels, self.deploy = in_channels, out_channels, deploy
        self.norm = norm
        self.activation = nn.LeakyReLU(0.1, inplace=False)
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        elif norm == 'batch':
            self.rbr_identity = (nn.BatchNorm2d(in_channels)
                                 if out_channels == in_channels else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            # r114b. Upstream carries BatchNorm because that is how RepVGG's
            # branches fold, but BN is long documented as harmful in
            # super-resolution (EDSR removed it for this reason) and this
            # project trains at batch 4 on 96px crops - close to the worst case
            # for batch statistics. r114's arm lost 4.58 dB PSNR while
            # over-texturing, which is what unstable normalisation produces
            # rather than what a differently-shaped 3x3 stack produces.
            #
            # The branches keep their topology and lose the normalisation, so
            # the fold stays exact and the comparison isolates block design.
            # Note this converges on the choice SPAN's Conv3XC already made.
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=True)
            self.rbr_identity = (nn.Parameter(torch.ones(in_channels))
                                 if out_channels == in_channels else None)

    def forward(self, x):
        if self.deploy:
            return self.activation(self.rbr_reparam(x))
        if self.norm == 'none':
            out = self.rbr_dense(x) + self.rbr_1x1(x)
            if self.rbr_identity is not None:
                out = out + x * self.rbr_identity.view(1, -1, 1, 1)
            return self.activation(out)
        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.activation(self.rbr_dense(x) + self.rbr_1x1(x) + identity)

    # ---- reparameterisation ------------------------------------------------

    def _fuse(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Conv2d):  # norm='none': already a plain conv
            bias = branch.bias if branch.bias is not None else torch.zeros(
                branch.out_channels, device=branch.weight.device, dtype=branch.weight.dtype)
            return branch.weight, bias
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            norm = branch[1]
        else:  # a bare BatchNorm identity branch
            kernel = torch.zeros(self.in_channels, self.in_channels, 3, 3,
                                 device=branch.weight.device, dtype=branch.weight.dtype)
            for channel in range(self.in_channels):
                kernel[channel, channel, 1, 1] = 1
            norm = branch
        std = (norm.running_var + norm.eps).sqrt()
        scale = (norm.weight / std).reshape(-1, 1, 1, 1)
        return kernel * scale, norm.bias - norm.running_mean * norm.weight / std

    def fused(self):
        """The single 3x3 this block is equivalent to at inference."""
        kernel3, bias3 = self._fuse(self.rbr_dense)
        kernel1, bias1 = self._fuse(self.rbr_1x1)
        if self.norm == 'none':
            if self.rbr_identity is None:
                kernel_id, bias_id = 0, 0
            else:
                kernel_id = torch.zeros_like(kernel3)
                for channel in range(self.in_channels):
                    kernel_id[channel, channel, 1, 1] = self.rbr_identity[channel]
                bias_id = 0
        else:
            kernel_id, bias_id = self._fuse(self.rbr_identity)
        if not torch.is_tensor(kernel1):
            kernel1 = torch.zeros_like(kernel3)
        else:
            kernel1 = F.pad(kernel1, [1, 1, 1, 1])
        if not torch.is_tensor(kernel_id):
            kernel_id = torch.zeros_like(kernel3)
        return kernel3 + kernel1 + kernel_id, bias3 + bias1 + bias_id

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.fused()
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for name in ('rbr_dense', 'rbr_1x1', 'rbr_identity'):
            if hasattr(self, name):
                delattr(self, name)
        self.deploy = True


class NanoTrunk(nn.Module):
    """Causal NanoVSR-derived trunk at a configurable scale.

    Output geometry matches `train_span.Unshuffled`: (B, 3, H*scale, W*scale)
    from (B, 3*frames, H, W) with the current frame last, so it is a drop-in
    for the matched-recipe trainer and for the existing conversion path.
    """

    def __init__(self, channels=32, scale=2, frames=1, blocks=7,
                 unshuffle=True, deploy=False, norm='batch'):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError('net scale must be 2 or 4')
        if blocks < 1:
            raise ValueError('at least one block required')
        self.channels, self.scale, self.frames = channels, scale, frames
        self.blocks, self.unshuffle_trunk = blocks, unshuffle

        # Quarter-area trunk, matching Unshuffled, so a comparison against it is
        # about the block design rather than about where the trunk runs.
        self.unshuffle = nn.PixelUnshuffle(2) if unshuffle else None
        in_channels = 3 * frames * (4 if unshuffle else 1)
        trunk_scale = scale * 2 if unshuffle else scale

        self.norm = norm
        self.feat_extract = RepBlock(in_channels, channels, deploy=deploy, norm=norm)
        self.forward_net = nn.Sequential(
            *[RepBlock(channels, channels, deploy=deploy, norm=norm) for _ in range(blocks)])
        self.fusion = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.upsampler = nn.Sequential(
            nn.Conv2d(channels, 3 * trunk_scale * trunk_scale, 3, 1, 1, bias=True),
            nn.PixelShuffle(trunk_scale))

    def forward(self, x, state=None, return_state=False):
        """One frame in, one frame out, with optional carried recurrent state.

        `state` is the trunk feature map from the previous frame. Upstream
        propagates exactly this along the sequence - it is what makes NanoVSR a
        *video* model rather than an image model applied per frame, and r114
        deliberately ran without it so the architecture comparison against SPAN
        was single-frame on both sides.

        Turning it on is the point of the architecture. r83 measured that 74% of
        this model's error sits in bands where the input carries 2.5% of its
        energy; no per-frame capacity invents that information, and successive
        frames genuinely contain it. Causal by construction: the state only ever
        comes from frames already shown, so nothing here can depend on a future
        frame the way upstream's backward pass does.
        """
        current = x[:, -3:] if self.frames > 1 else x
        features = self.unshuffle(x) if self.unshuffle is not None else x
        features = self.feat_extract(features)
        if state is not None:
            # Additive carry, as upstream does it: the block stack sees the new
            # frame's features plus what it propagated forward, so a detail that
            # is only resolvable across frames can survive into this one.
            features = features + state
        features = self.forward_net(features)
        carried = features
        features = self.fusion(features)
        residual = self.upsampler(features)
        # Upstream's bilinear base: the network learns a correction rather than
        # the whole picture. SPAN's Unshuffled has no such global skip, and this
        # is one of the concrete things the comparison is testing.
        base = F.interpolate(current, scale_factor=self.scale,
                             mode='bilinear', align_corners=False)
        output = base + residual
        return (output, carried) if return_state else output

    def forward_sequence(self, frames):
        """Run a clip causally, carrying state forward.

        `frames` is (B, T, 3*frames_per_step, H, W). Returns (B, T, 3, H*s, W*s).
        The state starts at zero, which is also what the first frame of a real
        stream sees, so training matches playback rather than assuming a warm
        history the live path never has.
        """
        if frames.dim() != 5:
            raise ValueError('expected (batch, time, channels, height, width)')
        state, outputs = None, []
        for index in range(frames.shape[1]):
            output, state = self(frames[:, index], state=state, return_state=True)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def switch_to_deploy(self):
        for module in self.modules():
            if module is not self and hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()


def fused_parameters(model):
    """Parameter count after RepBlock folding - what actually converts and ships."""
    import copy
    clone = copy.deepcopy(model).eval()
    clone.switch_to_deploy()
    return sum(p.numel() for p in clone.parameters())
