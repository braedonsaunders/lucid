"""Torch emulation of Lucid's source-resolution deband and temporal stages.

Pipeline order is decoded LR -> deband -> motion-aligned TAA -> model. These
are NOT post-model stages. Kernels follow DetailEnhancer.swift. RGB/420 packing
is an approximation of VideoToolbox and requires native output validation.
Hard decisions match native forward behavior; quantization uses a straight-
through derivative, useful when a learned cleaner precedes these stages.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StageSettings:
    threshold: float = .008
    guard: float = .005
    radius: float = 16
    iterations: int = 2
    feedback: float = .5
    gamma: float = 1.25
    motion: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in
                   (self.threshold, self.guard, self.radius, self.feedback, self.gamma)):
            raise ValueError('finite nonnegative stage settings required')
        if self.feedback > 1 or self.iterations < 1:
            raise ValueError('feedback <=1 and positive iterations required')


def quantize8(x):
    bounded = x.clamp(0, 1)
    return bounded + ((bounded * 255).round() / 255 - bounded).detach()


def smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _gather(image, x, y):
    """Gather shared spatial coordinates, clamped to the image boundary."""
    n, c, h, w = image.shape
    index = y.clamp(0, h - 1).long() * w + x.clamp(0, w - 1).long()
    return image.flatten(2)[:, :, index.flatten()].reshape(n, c, *index.shape)


def _round_away(x):
    # Metal round(), unlike torch.round(), rounds half-way values away from 0.
    return x.sign() * (x.abs() + .5).floor()


def deband(source, frame=1, settings=StageSettings()):
    """One/two-channel plane, including uint32 hash and immediate-neighbor guard."""
    _, _, height, width = source.shape
    y, x = torch.meshgrid(torch.arange(height, device=source.device),
                          torch.arange(width, device=source.device), indexing='ij')
    largest = torch.zeros_like(source[:, :1])
    if settings.guard > 0:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                largest = torch.maximum(largest, (_gather(source, x + dx, y + dy) - source).abs().amax(1, keepdim=True))
    eligible = largest <= settings.guard if settings.guard > 0 else torch.ones_like(largest, dtype=torch.bool)
    hashed = ((x * 73856093) ^ (y * 19349663) ^ (int(frame) * 83492791)) & 0xffffffff
    result = source
    for i in range(1, settings.iterations + 1):
        hashed = (hashed * 1664525 + 1013904223) & 0xffffffff
        distance = (hashed & 0xffff).float() / 65535
        hashed = (hashed * 1664525 + 1013904223) & 0xffffffff
        angle = (hashed & 0xffff).float() / 65535 * 6.2831853
        ox = distance * i * settings.radius * angle.cos()
        oy = distance * i * settings.radius * angle.sin()
        average = sum(_gather(source, x + _round_away(dx), y + _round_away(dy))
                      for dx, dy in ((ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox))) * .25
        result = torch.where(eligible & ((result - average).abs() <= settings.threshold / i), average, result)
    return result


@torch.no_grad()
def motion_blocks(current, previous, *, stationary_override=True):
    """Native two-pass 8px block correspondence, returned as Bx4xBHxBW.

    Match decisions are deliberately detached. Confidence and displacements
    depend on decoded inputs, never HR targets or model-generated features.
    The default preserves legacy TAA recipes. The current app uses the full
    integer search result, selected here by disabling stationary overrides.
    """
    n, _, height, width = current.shape
    cy, cx = torch.meshgrid(torch.arange(0, height, 8, device=current.device),
                            torch.arange(0, width, 8, device=current.device), indexing='ij')
    cx, cy = (cx + 4).clamp_max(width - 1), (cy + 4).clamp_max(height - 1)
    py, px = torch.meshgrid(torch.arange(-4, 5, 2, device=current.device),
                            torch.arange(-4, 5, 2, device=current.device), indexing='ij')
    sx = (cx[..., None] + px.flatten()).clamp(0, width - 1)
    sy = (cy[..., None] + py.flatten()).clamp(0, height - 1)
    samples = _gather(current, sx, sy)[:, 0]
    stationary = (samples - _gather(previous, sx, sy)[:, 0]).abs().mean(-1)
    best = torch.full_like(stationary, 1e6)
    error = torch.ones_like(stationary)
    displacement = torch.zeros(n, *cx.shape, 2, device=current.device)
    for radius, stride in ((8, 2), (1, 1)):
        origin = displacement.clone()
        for dy in range(-radius, radius + 1, stride):
            for dx in range(-radius, radius + 1, stride):
                delta = origin + origin.new_tensor([dx, dy])
                xx = (sx[None] + delta[..., 0, None]).clamp(0, width - 1).long()
                yy = (sy[None] + delta[..., 1, None]).clamp(0, height - 1).long()
                index = (yy * width + xx).flatten(1)
                matched = previous[:, 0].flatten(1).gather(1, index).reshape_as(samples)
                cost = (samples - matched).abs().mean(-1)
                regularized = cost + .00015 * delta.square().sum(-1)
                wins = regularized < best
                best = torch.where(wins, regularized, best)
                error = torch.where(wins, cost, error)
                displacement = torch.where(wins[..., None], delta, displacement)
    if stationary_override:
        stationary_match = (stationary <= 6.5 / 255) | (stationary - error < 2 / 255)
        displacement = torch.where(stationary_match[..., None], 0, displacement)
        error = torch.where(stationary_match, stationary, error)
    confidence = 1 - smoothstep(6 / 255, 16 / 255, error)
    if stationary_override:
        confidence = torch.where(stationary <= 6.5 / 255, 1, confidence)
    return torch.cat((displacement, confidence[..., None], error[..., None]), -1).permute(0, 3, 1, 2)


def taa(source, history, raw_current, raw_history, field, settings=StageSettings()):
    if history is None:
        return source
    n, _, height, width = source.shape
    padded = F.pad(source, (1, 1, 1, 1), mode='replicate')
    mean = F.avg_pool2d(padded, 3, 1)
    variance = (F.avg_pool2d(padded.square(), 3, 1) - mean.square()).clamp_min(0)
    # Safe derivative at zero variance; exactly zero sigma in the forward pass.
    sigma = torch.where(variance > 0, variance.clamp_min(1e-12).sqrt(), 0)
    lo, hi = mean - settings.gamma * sigma, mean + settings.gamma * sigma
    y, x = torch.meshgrid(torch.arange(height, device=source.device),
                          torch.arange(width, device=source.device), indexing='ij')
    motion = field.repeat_interleave(8, -2).repeat_interleave(8, -1)[..., :height, :width]
    xx = x[None] + (motion[:, 0] if settings.motion else 0)
    yy = y[None] + (motion[:, 1] if settings.motion else 0)
    grid = torch.stack(((xx + .5) * 2 / width - 1, (yy + .5) * 2 / height - 1), -1).expand(n, -1, -1, -1)
    warp = lambda image: F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    previous = warp(history)
    inside = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))[:, None]
    confidence = (inside * motion[:, 2:3] * (1 - smoothstep(8 / 255, 24 / 255,
                  (raw_current - warp(raw_history)).abs()))) if settings.motion else 1
    clipped = previous.maximum(lo.minimum(source)).minimum(hi.maximum(source))
    difference = (source - clipped).abs() / source.maximum(clipped).clamp_min(.2)
    keep = max(0, settings.feedback - .05) + min(.05, settings.feedback) * (1 - difference).square()
    return source + (clipped - source) * keep * confidence


def rgb_to_planes(rgb):
    r, g, b = rgb.split(1, 1)
    y = .2126 * r + .7152 * g + .0722 * b
    uv = torch.cat(((b - y) / 1.8556, (r - y) / 1.5748), 1)
    return quantize8((16 + 219 * y) / 255), quantize8((128 + 224 * F.avg_pool2d(uv, 2)) / 255)


def planes_to_rgb(y, uv):
    y = (y * 255 - 16) / 219
    cb, cr = ((F.interpolate(uv, size=y.shape[-2:], mode='bilinear', align_corners=False) * 255 - 128) / 224).split(1, 1)
    return quantize8(torch.cat((y + 1.5748 * cr, y - .1873 * cb - .4681 * cr, y + 1.8556 * cb), 1))


def preprocess_sequence(frames, settings=StageSettings(), first_frame=1, *, motion_policy='taa'):
    """Process BxTx3xHxW frames with reset history and preserve gradients.

    Reset at each training sequence; temporal history is kept in FP16 just as
    the native texture is. Quantize every UNORM8 intermediate plane. Crop-edge
    context and RGB/420 resampling remain explicit proxy limitations.
    Legacy recipes retain their default; current-app training must explicitly
    request motion_policy='search'.
    """
    if motion_policy not in ('taa', 'search'):
        raise ValueError('source motion policy must be taa or search')
    if frames.ndim != 5 or frames.shape[2] != 3 or any(s % 2 for s in frames.shape[-2:]):
        raise ValueError('BTCHW RGB sequence with even spatial size required')
    history = raw_history = None
    outputs = []
    chroma_settings = StageSettings(settings.threshold * 1.5, settings.guard * 1.5,
                                    settings.radius * .5, settings.iterations,
                                    settings.feedback, settings.gamma, settings.motion)
    for t in range(frames.shape[1]):
        raw, uv = rgb_to_planes(frames[:, t])
        cleaned = quantize8(deband(raw, first_frame + t, settings))
        if history is not None:
            field = (motion_blocks(raw, raw_history) if motion_policy == 'taa'
                     else motion_blocks(raw, raw_history, stationary_override=False)).half().float()
            cleaned = taa(cleaned, history, raw, raw_history, field, settings)
        history = cleaned.half().float()
        raw_history = raw
        uv = quantize8(deband(uv, first_frame + t, chroma_settings))
        outputs.append(planes_to_rgb(quantize8(cleaned), uv))
    return torch.stack(outputs, 1)
