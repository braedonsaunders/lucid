"""Experimental updates in the complement of a frozen convolution subspace.

Unlike a gradient hook, the parameterization survives Adam's coordinate scaling.
The adapters merge into ordinary convolutions for inference. Orthogonal weight
updates do NOT guarantee image fidelity through the nonlinear reconstruction.
This is not a reproduction of FreqOrtho-SR's diffusion or pixel-LoRA training.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .span_arch import Conv3XC


def fuse_convolutions(module):
    """Materialize the existing SPAN inference graph before adapter training."""
    for name, child in list(module.named_children()):
        if isinstance(child, Conv3XC):
            child.update_params()
            replacement = copy.deepcopy(child.eval_conv)
            if child.has_relu:
                replacement = nn.Sequential(replacement, nn.LeakyReLU(.05))
            setattr(module, name, replacement)
        else:
            fuse_convolutions(child)


class SubspaceConv(nn.Module):
    def __init__(self, conv, constrained=True, energy=.95):
        super().__init__()
        if conv.groups != 1 or conv.bias is None or conv.padding_mode != 'zeros':
            raise ValueError('dense biased zero-padded convolution required')
        if not 0 < energy < 1:
            raise ValueError('energy must lie strictly between zero and one')
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        self.weight_shape = tuple(conv.weight.shape)
        matrix = torch.cat((conv.weight.detach().flatten(1), conv.bias.detach()[:, None]), 1)
        # CPU float64 gives both matched arms the same fixed SVD coordinates.
        u, s, _ = torch.linalg.svd(matrix.cpu().double(), full_matrices=True)
        power = s.square()
        rank = int(torch.searchsorted(power.cumsum(0) / power.sum(), energy)) + 1 if power.sum() > 0 else 0
        self.protected_rank = rank
        self.constrained = constrained
        self.register_buffer('anchor', matrix.clone())
        self.register_buffer('protected_basis', u[:, :rank].to(matrix))
        self.register_buffer('basis', u[:, rank if constrained else 0:].to(matrix))
        self.coefficients = nn.Parameter(matrix.new_zeros((self.basis.shape[1], matrix.shape[1])))

    def delta(self):
        # Do not quantize the projection itself under the BF16 training context.
        with torch.autocast(device_type=self.anchor.device.type, enabled=False):
            return self.basis @ self.coefficients

    def effective_matrix(self):
        return self.anchor + self.delta()

    def forward(self, x):
        matrix = self.effective_matrix()
        return F.conv2d(x, matrix[:, :-1].reshape(self.weight_shape), matrix[:, -1],
                        self.stride, self.padding, self.dilation)

    def merged(self):
        out_channels, in_channels, h, w = self.weight_shape
        conv = nn.Conv2d(in_channels, out_channels, (h, w), self.stride, self.padding,
                         self.dilation, bias=True, device=self.anchor.device, dtype=self.anchor.dtype)
        with torch.no_grad():
            matrix = self.effective_matrix()
            conv.weight.copy_(matrix[:, :-1].reshape(self.weight_shape))
            conv.bias.copy_(matrix[:, -1])
        return conv


def attach_adapters(model, constrained=True, energy=.95):
    model = copy.deepcopy(model).eval()
    fuse_convolutions(model)
    model.requires_grad_(False)
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, SubspaceConv(child, constrained, energy))
            else:
                replace(child)
    replace(model)
    return model


def merge_adapters(model):
    model = copy.deepcopy(model).eval()
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, SubspaceConv):
                setattr(module, name, child.merged())
            else:
                replace(child)
    replace(model)
    return model


@torch.no_grad()
def adapter_report(model):
    rows = []
    for name, layer in model.named_modules():
        if isinstance(layer, SubspaceConv):
            delta = layer.delta()
            actual_delta = layer.effective_matrix() - layer.anchor
            projected = layer.protected_basis.T @ actual_delta
            rows.append({'layer': name, 'constrained': layer.constrained,
                'protected_rank': layer.protected_rank, 'free_rank': layer.basis.shape[1],
                'coefficients': layer.coefficients.numel(),
                'delta_norm': float(delta.norm()), 'protected_update_norm': float(projected.norm())})
    return rows
