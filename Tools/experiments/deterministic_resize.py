"""Separable bicubic-antialias resampling with deterministic matrix-product backward."""
from functools import lru_cache
import torch
from torch.nn import functional as F


@lru_cache(maxsize=16)
def _weights(source, target, device):
    # Use this backend's forward impulse responses, retaining its filter and
    # coordinate rounding. These fixed coefficients need no resampler backward.
    impulses = torch.eye(source, device=device).reshape(source, 1, 1, source)
    return F.interpolate(impulses, (1, target), mode='bicubic',
                         align_corners=False, antialias=True)[:, 0, 0].t().contiguous()


def bicubic(image, size):
    if image.ndim != 4 or min(*image.shape[-2:], *size) < 1:
        raise ValueError('positive NCHW geometry required')
    height, width = image.shape[-2:]
    with torch.autocast(device_type=image.device.type, enabled=False):
        horizontal = _weights(width, size[1], image.device)
        vertical = _weights(height, size[0], image.device)
        return vertical @ (image.float() @ horizontal.t())
