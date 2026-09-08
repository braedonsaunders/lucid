"""Differentiable shipping 2x output-stage proxy (sharpen, tone, adaptive grain).

Fixed shipping recipe: radius/reference radius 2, sharpness .2, no additional
detail bands/deblock, neutral saturation and static grain. RGB/420 conversion
and crop-coordinate grain statistics inherit the source proxy's limitations.
"""
import torch

from native_stages import _gather, quantize8, rgb_to_planes, planes_to_rgb, smoothstep


def sharpen_luma(source, sharpness=.2):
    height, width = source.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=source.device),
                          torch.arange(width, device=source.device), indexing='ij')
    neighbors = lambda r: [_gather(source, x + dx * r, y + dy * r)
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    tight, medium, wide = neighbors(1), neighbors(2), neighbors(4)
    mean = sum(medium) / 9
    variance = (sum(v.square() for v in medium) / 9 - mean.square()).clamp_min(0)
    sigma = torch.where(variance > 0, variance.clamp_min(1e-12).sqrt(), 0)
    activation = smoothstep(.004, .030, sigma)
    cross = torch.stack([tight[i] for i in (1, 3, 4, 5, 7)])
    mn, mx = cross.amin(0), cross.amax(0)
    hit_min = mn / (4 * mx.clamp_min(1e-4))
    hit_max = (1 - mx) / (4 * mn - 4).clamp_min(-4 + 1e-4)
    lobe = torch.maximum(-hit_min, hit_max).clamp(-.1875, 0) * sharpness * activation
    result = (lobe * sum(tight[i] for i in (1, 3, 5, 7)) + source) / (4 * lobe + 1)
    all_neighbors = torch.stack(tight + medium + wide)
    return result.maximum(all_neighbors.amin(0)).minimum(all_neighbors.amax(0))


def grain_scale(raw_luma):
    """Native off-grid integer second-difference statistic on the decoded LR."""
    h, w = raw_luma.shape[-2:]
    yy = torch.arange(4, h - 2, 8, device=raw_luma.device)
    xx = torch.arange(4, w - 2, 8, device=raw_luma.device)
    if not len(yy) or not len(xx):
        return raw_luma.new_ones((raw_luma.shape[0], 1, 1, 1))
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    difference = (2 * _gather(raw_luma, x, y) - _gather(raw_luma, x - 1, y)
                  - _gather(raw_luma, x + 1, y)).abs()
    detail = (difference * 8192).floor().mean((-1, -2), keepdim=True) / 8192
    return (detail / .02).clamp(.25, 1)


def grade_luma(source, scale, grain=.01):
    y = ((source * 255 - 16) / 219 - .02) / .97
    y = y.clamp(0, 1)
    curve = y * y * (3 - 2 * y)
    y = y + .2 * (curve - y)
    height, width = source.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device),
                            torch.arange(width, device=source.device), indexing='ij')
    inner = (.06711056 * xx + .00583715 * yy).frac()
    noise = (52.9829189 * inner).frac() - .5
    y = (y + noise * grain * scale).clamp(0, 1)
    return (y * 219 + 16) / 255


def postprocess_rgb(output, decoded_lr):
    luma, chroma = rgb_to_planes(quantize8(output))
    with torch.no_grad():
        raw_luma, _ = rgb_to_planes(decoded_lr)
        scale = grain_scale(raw_luma)
    sharpened = quantize8(sharpen_luma(luma))
    graded = quantize8(grade_luma(sharpened, scale))
    return planes_to_rgb(graded, chroma)
