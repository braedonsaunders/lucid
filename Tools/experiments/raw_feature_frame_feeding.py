"""Explicit causal driver for decoded-observation feature memory experiments."""
import torch
from torch.nn import functional as F

from native_stages import preprocess_sequence, quantize8
from recurrent_frame_feeding import warp_previous


def run_raw_feature_sequence(model, decoded, *, use_history=True,
                             source_motion_policy='raw', return_inputs=False):
    if source_motion_policy not in ('raw', 'taa', 'search'):
        raise ValueError('source motion policy must be raw, taa or search')
    inputs = (decoded if source_motion_policy == 'raw' else
              preprocess_sequence(decoded, motion_policy=source_motion_policy))
    state, outputs = None, []
    for t in range(decoded.shape[1]):
        current = inputs[:, t]
        confidence = current.new_zeros(current.shape[0], 1,
                                       current.shape[-2] * 2, current.shape[-1] * 2)
        if t and use_history:
            with torch.no_grad():
                # Reuse validated correspondence. Its RGB warp is computed but
                # not consumed by this model; it is not the feature source.
                _, confidence, cut, grid = warp_previous(
                    decoded[:, t], decoded[:, t - 1], previous_output,
                    motion_seed='search', return_state_grid=True)
            state = F.grid_sample(state, grid, padding_mode='zeros', align_corners=False)
            state = torch.where(cut[:, None, None, None], 0, state)
        output, state = model(current, decoded[:, t], confidence,
                              state if use_history else None)
        output = output.clamp(0, 1)
        outputs.append(output)
        previous_output = quantize8(output.detach())
    result = torch.stack(outputs, 1)
    return (result, inputs) if return_inputs else result
