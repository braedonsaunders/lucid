"""Convert the trained 4x head to direct 2x using exact linear phase averaging.

Equivalent to 2x2 area pooling of the *unclipped floating-point* 4x output.
This is not equivalent to bicubic presentation or pooling after RGB8 clipping.
No training or trunk changes. Keep separate from shipping model assets.
"""
import copy
import torch
from torch import nn


def fold_head(model):
    result = copy.deepcopy(model).eval()
    head, shuffle = result.core.upsampler
    if not isinstance(head, nn.Conv2d) or not isinstance(shuffle, nn.PixelShuffle):
        raise ValueError('expected convolution followed by PixelShuffle')
    if shuffle.upscale_factor != 8 or head.out_channels != 192:
        raise ValueError('expected RGB 8-phase shipping head')
    folded = nn.Conv2d(head.in_channels, 48, head.kernel_size,
                       stride=head.stride, padding=head.padding,
                       dilation=head.dilation, bias=head.bias is not None,
                       device=head.weight.device, dtype=head.weight.dtype)
    def phases(value):
        # PixelShuffle channel order is RGB, vertical phase, horizontal phase.
        shape = value.shape[1:]
        return value.reshape(3, 4, 2, 4, 2, *shape).mean((2, 4)).reshape(48, *shape)
    with torch.no_grad():
        folded.weight.copy_(phases(head.weight))
        if head.bias is not None:
            folded.bias.copy_(phases(head.bias))
    result.core.upsampler = nn.Sequential(folded, nn.PixelShuffle(4))
    return result
