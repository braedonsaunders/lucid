"""Input-derived correspondence and explicit lifetime for previous SR outputs."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from native_stages import motion_blocks, quantize8, smoothstep, rgb_to_planes


def load_recurrent_checkpoint(path, device):
    from architectures.recurrent_span import RecurrentSPAN
    from train_span import Unshuffled
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('architecture') != 'recurrent_span2x' or state.get('scale') != 2:
        raise ValueError('explicit recurrent 2x checkpoint required')
    history_source = state.get('history_source', 'sr')
    declared = state.get('experiment', {}).get('args', {}).get('history_source', history_source)
    if declared != history_source:
        raise ValueError('checkpoint history source differs from training declaration')
    # Historical checkpoints used the TAA-overridden integer seed. Preserve
    # their inference semantics instead of silently changing archived results.
    motion_seed = state.get('motion_seed', 'taa')
    declared_motion = state.get('experiment', {}).get('args', {}).get('motion_seed', motion_seed)
    if declared_motion != motion_seed:
        raise ValueError('checkpoint motion seed differs from training declaration')
    model = RecurrentSPAN(Unshuffled(state['channels'], scale=2, version=state['version']),
                          history_source=history_source, motion_seed=motion_seed)
    model.load_state_dict(state['model'])
    return model.eval().to(device), state


@torch.no_grad()
def subpixel_motion(current, previous, *, motion_seed='search'):
    """Refine native integer blocks over a half-pixel neighborhood.

    This experimental correspondence is not the production Metal kernel. It lets
    the temporal probe test fractional samples rather than only integer denoising.
    """
    if motion_seed not in ('taa', 'search'):
        raise ValueError('motion seed must be taa or search')
    # TAA can deliberately suppress low-error motion, but SR refinement needs
    # the actual integer match. Otherwise a four-pixel match can be discarded
    # and become unreachable inside the remaining +/-0.5-pixel search.
    field = motion_blocks(current, previous, stationary_override=motion_seed == 'taa')
    n, _, height, width = current.shape
    bh, bw = field.shape[-2:]
    cy, cx = torch.meshgrid(torch.arange(bh, device=current.device) * 8 + 4,
                            torch.arange(bw, device=current.device) * 8 + 4, indexing='ij')
    oy, ox = torch.meshgrid(torch.arange(-4, 5, 2, device=current.device),
                            torch.arange(-4, 5, 2, device=current.device), indexing='ij')
    sx = (cx.clamp_max(width - 1)[..., None] + ox.flatten()).clamp(0, width - 1)
    sy = (cy.clamp_max(height - 1)[..., None] + oy.flatten()).clamp(0, height - 1)
    targets = current[:, 0].flatten(1)[:, (sy * width + sx).flatten()].reshape(n, bh, bw, 25)
    origin = field[:, :2].permute(0, 2, 3, 1)
    best = field[:, 3] + .00015 * origin.square().sum(-1)
    chosen, error = origin.clone(), field[:, 3].clone()
    # Also refine zero: an integer search can choose an unrelated block when
    # every true match lies between pixels. Do not inherit TAA's small-motion
    # stationary override, which would suppress the fractional evidence here.
    for center in (origin, torch.zeros_like(origin)):
        for dy in (-.5, 0., .5):
            for dx in (-.5, 0., .5):
                delta = center + center.new_tensor([dx, dy])
                xx, yy = sx[None] + delta[..., 0, None], sy[None] + delta[..., 1, None]
                grid = torch.stack(((xx + .5) * 2 / width - 1, (yy + .5) * 2 / height - 1), -1)
                samples = F.grid_sample(previous, grid.reshape(n, bh * bw, 25, 2),
                                    padding_mode='border', align_corners=False).reshape_as(targets)
                residual = (samples - targets).abs().mean(-1)
                cost = residual + .00015 * delta.square().sum(-1)
                better = cost < best
                best, error = torch.where(better, cost, best), torch.where(better, residual, error)
                chosen = torch.where(better[..., None], delta, chosen)
    confidence = 1 - smoothstep(6 / 255, 16 / 255, error)
    return torch.cat((chosen, confidence[..., None], error[..., None]), -1).permute(0, 3, 1, 2)


def warp_previous(current_rgb, previous_rgb, previous_output, *, subpixel=True, motion_seed='search',
                  return_state_grid=False):
    """Warp a 2x reconstruction with LR block motion; reject cuts and occlusions.

    No output/reference target participates in motion. Optional half-pixel
    refinement remains a prototype requiring native cost and quality checks.
    """
    n, _, height, width = current_rgb.shape
    if previous_rgb.shape != current_rgb.shape or previous_output.shape != (n, 3, height * 2, width * 2):
        raise ValueError('matched LR history and exact 2x reconstruction required')
    if motion_seed not in ('taa', 'search'):
        raise ValueError('motion seed must be taa or search')
    with torch.no_grad():
        current, _ = rgb_to_planes(current_rgb)
        previous, _ = rgb_to_planes(previous_rgb)
        field = (subpixel_motion(current, previous, motion_seed=motion_seed) if subpixel else
                 motion_blocks(current, previous, stationary_override=motion_seed == 'taa')).half().float()
        motion = field.repeat_interleave(8, -2).repeat_interleave(8, -1)[..., :height, :width]
        y, x = torch.meshgrid(torch.arange(height, device=current.device),
                              torch.arange(width, device=current.device), indexing='ij')
        xx, yy = x[None] + motion[:, 0], y[None] + motion[:, 1]
        grid = torch.stack(((xx + .5) * 2 / width - 1, (yy + .5) * 2 / height - 1), -1)
        raw_warp = F.grid_sample(previous, grid, padding_mode='border', align_corners=False)
        inside = ((xx >= 0) & (xx <= width - 1) & (yy >= 0) & (yy <= height - 1))[:, None]
        confidence = motion[:, 2:3] * inside * (1 - smoothstep(8 / 255, 24 / 255, (current - raw_warp).abs()))
        # A conservative automatic reset supplements explicit seek/stream resets.
        cut = ((current - previous).abs().mean((1, 2, 3)) > .1) & (confidence.mean((1, 2, 3)) < .1)
        confidence = torch.where(cut[:, None, None, None], 0, confidence)
        motion_hr = motion[:, :2].repeat_interleave(2, -2).repeat_interleave(2, -1)
        hy, hx = torch.meshgrid(torch.arange(height * 2, device=current.device),
                                torch.arange(width * 2, device=current.device), indexing='ij')
        hxx, hyy = hx[None] + motion_hr[:, 0] * 2, hy[None] + motion_hr[:, 1] * 2
        hr_grid = torch.stack(((hxx + .5) / width - 1, (hyy + .5) / height - 1), -1)
        valid_hr = confidence.repeat_interleave(2, -2).repeat_interleave(2, -1)
        if return_state_grid:
            # SPAN trunk coordinates are half LR resolution. Average the two
            # LR samples per trunk cell, then convert displacement to trunk units.
            sh, sw = height // 2, width // 2
            state_motion = F.avg_pool2d(motion[:, :2], 2) / 2
            sy, sx = torch.meshgrid(torch.arange(sh, device=current.device),
                                    torch.arange(sw, device=current.device), indexing='ij')
            state_grid = torch.stack(((sx[None] + state_motion[:, 0] + .5) * 2 / sw - 1,
                                      (sy[None] + state_motion[:, 1] + .5) * 2 / sh - 1), -1)
    aligned = F.grid_sample(previous_output, hr_grid, padding_mode='border', align_corners=False)
    if return_state_grid:
        return aligned, valid_hr, cut, state_grid
    return aligned, valid_hr, cut


@dataclass
class FrameState:
    source: torch.Tensor
    output: torch.Tensor
    stream: str
    index: int


def recurrent_step(model, current, raw_current, state, *, stream, index, reset=False,
                   detach_history=False, use_history=True):
    """Reset on first frame, seek/gap, stream switch, geometry/device or explicit reset.

State stores decoded observations and quantized pre-detail SR output. The
decoded ablation consumes only the previous observed frame, not generated SR;
the history branch therefore retains one observed previous frame rather than
recursively accumulated SR. Upstream native TAA can still carry earlier frames.
Pass raw decoded RGB separately when the SR input has been preprocessed.
"""
    if getattr(model, 'requires_feature_state', False):
        raise ValueError('feature-state model requires an explicit feature-state sequence driver')
    if current.shape != raw_current.shape or current.ndim != 4 or current.shape[1] != 3 or any(s % 2 for s in current.shape[-2:]):
        raise ValueError('matched even BCHW RGB inputs required')
    discontinuity = (state is None or reset or not use_history or state.stream != stream or
                     state.index + 1 != index or state.source.shape != raw_current.shape or
                     state.source.device != raw_current.device or state.source.dtype != raw_current.dtype)
    if discontinuity:
        aligned = current.new_zeros(current.shape[0], 3, current.shape[-2] * 2, current.shape[-1] * 2)
        confidence = aligned[:, :1]
        cut = torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
    else:
        if model.history_source == 'decoded':
            # Lift observed LR into the same 2x history interface before warping.
            # Interpolation preserves the observation input, not new HR evidence.
            previous = F.interpolate(state.source.detach(), scale_factor=2,
                                     mode='bicubic', align_corners=False).clamp(0, 1)
        else:
            previous = state.output.detach() if detach_history else state.output
        aligned, confidence, cut = warp_previous(raw_current, state.source, previous,
                                                motion_seed=model.motion_seed)
    output = model(current, aligned, confidence).clamp(0, 1)
    stored = quantize8(output)
    if detach_history:
        stored = stored.detach()
    return output, FrameState(raw_current.detach(), stored, stream, index), cut
