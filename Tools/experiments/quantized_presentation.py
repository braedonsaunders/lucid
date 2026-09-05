"""Fixed 2x presentation of shipping's 4x RGB8 output, inside the inference graph."""
import torch
from torch import nn
from torch.nn import functional as F


def rgb8(x):
    return torch.floor(x.clamp(0, 255)+.5)


def bicubic_half(x):
    """Separable Catmull-Rom antialias filtering with clipped RGB8 intermediates.

    The 2:1 interior coefficients are exact binary fractions. Boundary weights
    renormalize over valid pixels; horizontal rounding matches PIL's RGB path.
    """
    if x.shape[1] != 3:
        raise ValueError('RGB input required')
    weights = x.new_tensor([-3, -9, 29, 111, 111, 29, -9, -3])/256
    horizontal = weights.reshape(1, 1, 1, 8)
    vertical = weights.reshape(1, 1, 8, 1)
    hnorm = F.conv2d(torch.ones_like(x[:, :1, :1]), horizontal, stride=(1, 2), padding=(0, 3))
    x = rgb8(F.conv2d(x, horizontal.repeat(3, 1, 1, 1), stride=(1, 2), padding=(0, 3), groups=3)/hnorm)
    vnorm = F.conv2d(torch.ones_like(x[:, :1, :, :1]), vertical, stride=(2, 1), padding=(3, 0))
    return rgb8(F.conv2d(x, vertical.repeat(3, 1, 1, 1), stride=(2, 1), padding=(3, 0), groups=3)/vnorm)


class QuantizedPresentation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return bicubic_half(rgb8(self.model(image)*255))/255
