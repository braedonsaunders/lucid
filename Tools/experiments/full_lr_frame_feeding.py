"""Full-LR motion/confidence for feature memory, with an explicit packed ablation.

Confidence and cut rules match recurrent_frame_feeding.warp_previous. The field
is retained at LR resolution and no unused generated-RGB warp is computed.
"""
import torch
from torch.nn import functional as F
from native_stages import preprocess_sequence, rgb_to_planes, smoothstep
from recurrent_frame_feeding import subpixel_motion


@torch.no_grad()
def full_lr_correspondence(current_rgb, previous_rgb):
    if current_rgb.ndim != 4 or current_rgb.shape[1] != 3 or current_rgb.shape != previous_rgb.shape:
        raise ValueError('matched BCHW decoded frames required')
    n, _, h, w = current_rgb.shape
    current, _ = rgb_to_planes(current_rgb)
    previous, _ = rgb_to_planes(previous_rgb)
    field = subpixel_motion(current, previous, motion_seed='search').half().float()
    motion = field.repeat_interleave(8, -2).repeat_interleave(8, -1)[..., :h, :w]
    yy, xx = torch.meshgrid(torch.arange(h, device=current.device),
                            torch.arange(w, device=current.device), indexing='ij')
    x, y = xx[None] + motion[:, 0], yy[None] + motion[:, 1]
    grid = torch.stack(((x + .5) * 2 / w - 1, (y + .5) * 2 / h - 1), -1)
    aligned = F.grid_sample(previous, grid, padding_mode='border', align_corners=False)
    inside = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))[:, None]
    confidence = motion[:, 2:3] * inside * (1 - smoothstep(8 / 255, 24 / 255, (current - aligned).abs()))
    cut = ((current - previous).abs().mean((1, 2, 3)) > .1) & (confidence.mean((1, 2, 3)) < .1)
    confidence = torch.where(cut[:, None, None, None], 0, confidence)
    return confidence, cut, grid


def align_full_lr_state(state, grid, *, state_warp='full_lr'):
    if state_warp == 'full_lr':
        return F.grid_sample(state, grid, padding_mode='zeros', align_corners=False)
    if state_warp != 'packed_half_lr':
        raise ValueError('state warp must be full_lr or packed_half_lr')
    # Deliberate matched alignment ablation: same features and capacity, but
    # warp packed phases as if they were ordinary half-resolution channels.
    packed = F.pixel_unshuffle(state, 2)
    packed_grid = F.avg_pool2d(grid.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    aligned = F.grid_sample(packed, packed_grid, padding_mode='zeros', align_corners=False)
    return F.pixel_shuffle(aligned, 2)


def run_full_lr_sequence(model, decoded, *, use_history=True, source_motion_policy='raw',
                         state_warp='full_lr', return_inputs=False):
    if source_motion_policy not in ('raw', 'taa', 'search'):
        raise ValueError('source motion policy must be raw, taa or search')
    if state_warp not in ('full_lr', 'packed_half_lr'):
        raise ValueError('state warp must be full_lr or packed_half_lr')
    inputs = (decoded if source_motion_policy == 'raw' else
              preprocess_sequence(decoded, motion_policy=source_motion_policy))
    state, outputs = None, []
    for t in range(decoded.shape[1]):
        current = inputs[:, t]
        confidence = current.new_zeros(current.shape[0], 1, *current.shape[-2:])
        if t and use_history:
            confidence, cut, grid = full_lr_correspondence(decoded[:, t], decoded[:, t - 1])
            state = align_full_lr_state(state, grid, state_warp=state_warp)
            state = torch.where(cut[:, None, None, None], 0, state)
        output, state = model(current, decoded[:, t], confidence, state if use_history else None)
        outputs.append(output.clamp(0, 1))
    result = torch.stack(outputs, 1)
    return (result, inputs) if return_inputs else result
