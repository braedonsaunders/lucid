"""Experimental causal, polyphase video reconstruction for native deployment.

Engineering hypothesis, not a reproduction or a novelty claim. See
Benchmarks/frontier/2026-architecture-direction.md for research and limitations.
No trained weights from this model are enabled in Lucid.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AtomMixer(nn.Module):
    """Image-conditioned low-rank channel mixing at the reduced spatial grid."""
    def __init__(self, channels, atoms=8):
        super().__init__()
        self.analysis = nn.Conv2d(channels, atoms, 1, bias=False)
        self.synthesis = nn.Conv2d(atoms, channels, 1, bias=False)
        self.router = nn.Conv2d(channels, atoms, 1)

    def forward(self, x):
        weights = torch.softmax(self.router(x.mean((2, 3), keepdim=True)), dim=1)
        return self.synthesis(self.analysis(x) * weights)


class DetailBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.project = nn.Conv2d(channels, channels * 2, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        a, b = self.project(self.spatial(x)).chunk(2, dim=1)
        return x + self.out(a * torch.tanh(b))


class CausalDetail(nn.Module):
    """One current RGB frame plus explicit bounded temporal feature state.

Pixel unshuffle retains all input samples while reducing the trunk area 16×.
Both state tensors are caller-owned: reset them on cuts, seeks and size changes.
`valid` is a B×1×1×1 tensor so resets remain data, not traced Python branches.
The zero-initialized reconstruction head starts at a bilinear baseline. It
must be trained and pass quality gates before any use in the product.
    """
    def __init__(self, channels=32, blocks=4, scale=2, fold=4):
        super().__init__()
        if scale not in (2, 4) or fold not in (2, 4) or channels < 8 or blocks < 1:
            raise ValueError('unsupported architecture configuration')
        self.channels, self.scale, self.fold = channels, scale, fold
        self.entry = nn.Conv2d(3 * fold * fold, channels, 3, padding=1)
        self.history = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.atoms = AtomMixer(channels)
        self.blocks = nn.Sequential(*[DetailBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 3 * (scale * fold)**2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # A conservative initial history preference, subsequently learned.
        nn.init.constant_(self.gate.bias, -2)

    def initial_state(self, frame):
        b, _, h, w = frame.shape
        if h % self.fold or w % self.fold:
            raise ValueError('input dimensions must be divisible by the polyphase fold')
        return frame.new_zeros(b, self.channels, h // self.fold, w // self.fold)

    def forward(self, frame, previous, valid):
        x = self.entry(F.pixel_unshuffle(frame, self.fold))
        history = previous * valid
        gate = torch.sigmoid(self.gate(torch.cat((x, history), dim=1))) * valid
        candidate = torch.tanh(x + self.history(history))
        state = (1 - gate) * candidate + gate * history
        features = self.blocks(state + self.atoms(state))
        residual = F.pixel_shuffle(self.head(features), self.scale * self.fold)
        base = F.interpolate(frame, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return base + residual, state
