"""Causal confidence control in the actual pre-SR/post-SR stage order.

Current confidence can control sharpening. Only the previous frame's aligned
confidence can control this frame's debanding without another SR inference.
Unknown/cut/out-of-bounds history falls back to the fixed shipping strength.
This is a training/evaluation proxy, not native playback integration.
"""
import torch
from torch.nn import functional as F
from architectures.confidence_span import confidence_from_log_variance
from native_stages import (StageSettings, deband, motion_blocks, taa, quantize8,
                            rgb_to_planes, planes_to_rgb, smoothstep)
from native_output_stages import sharpen_luma, grade_luma, grain_scale


def aligned_deband_confidence(previous_confidence, current, previous, field):
    if previous_confidence is None or previous is None:
        return torch.ones_like(current)
    n, _, height, width = current.shape
    motion = field.repeat_interleave(8, -2).repeat_interleave(8, -1)[..., :height, :width]
    y, x = torch.meshgrid(torch.arange(height, device=current.device),
                          torch.arange(width, device=current.device), indexing='ij')
    xx, yy = x[None] + motion[:, 0], y[None] + motion[:, 1]
    grid = torch.stack(((xx + .5) * 2 / width - 1, (yy + .5) * 2 / height - 1), -1)
    warp = lambda a: F.grid_sample(a, grid, padding_mode='border', align_corners=False)
    inside = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))[:, None]
    match = inside * motion[:, 2:3] * (1 - smoothstep(8 / 255, 24 / 255, (current - warp(previous)).abs()))
    return 1 + (warp(previous_confidence).clamp(0, 1) - 1) * match


def presented_with_confidence(output, decoded, confidence):
    y, uv = rgb_to_planes(quantize8(output))
    raw, _ = rgb_to_planes(decoded)
    sharpened = quantize8(sharpen_luma(y, .2 * confidence))
    return planes_to_rgb(quantize8(grade_luma(sharpened, grain_scale(raw))), uv)


@torch.no_grad()
def confidence_sequence(model, frames, policy, settings=StageSettings()):
    """BTCHW -> presented RGB, variance and the actual per-frame deband controls."""
    if policy not in ('fixed', 'constant', 'post', 'causal'):
        raise ValueError('fixed, constant, post or causal confidence policy required')
    if frames.ndim != 5 or frames.shape[2] != 3 or any(s % 2 for s in frames.shape[-2:]):
        raise ValueError('even BTCHW RGB sequence required')
    history = previous_raw = previous_confidence = None
    outputs, variances, controls, raw_predictions = [], [], [], []
    chroma = StageSettings(settings.threshold * 1.5, settings.guard * 1.5,
                           settings.radius * .5, settings.iterations, settings.feedback,
                           settings.gamma, settings.motion)
    for t in range(frames.shape[1]):
        raw, uv = rgb_to_planes(frames[:, t])
        field = motion_blocks(raw, previous_raw).half().float() if previous_raw is not None else None
        blend = (aligned_deband_confidence(previous_confidence, raw, previous_raw, field)
                 if policy == 'causal' else torch.full_like(raw, .5 if policy == 'constant' else 1))
        debanded = deband(raw, t + 1, settings)
        clean = quantize8(torch.where(blend == 1, debanded, raw + blend * (debanded - raw)))
        clean = taa(clean, history, raw, previous_raw, field, settings)
        history, previous_raw = clean.half().float(), raw
        uv_blend = F.avg_pool2d(blend, 2)
        debanded_uv = deband(uv, t + 1, chroma)
        uv = quantize8(torch.where(uv_blend == 1, debanded_uv, uv + uv_blend * (debanded_uv - uv)))
        source = planes_to_rgb(quantize8(clean), uv)
        output, log_variance = model.predict_with_uncertainty(source)
        confidence = confidence_from_log_variance(log_variance)
        if policy in ('fixed', 'constant'):
            confidence = torch.full_like(confidence, .5 if policy == 'constant' else 1)
        previous_confidence = F.avg_pool2d(quantize8(confidence), 2)
        outputs.append(presented_with_confidence(output, frames[:, t], confidence))
        variances.append(log_variance)
        controls.append(blend)
        raw_predictions.append(quantize8(output))
    return tuple(torch.stack(rows, 1) for rows in (outputs, variances, controls, raw_predictions))
