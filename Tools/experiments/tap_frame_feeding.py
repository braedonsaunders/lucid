"""Same-frame tap sequence feeding: accumulated input plus raw decoded tap.

No cross-frame alignment, no state, no recurrence. Each timestep feeds the
model the preprocessed (debanded, TAA-accumulated) frame and the raw decoded
current frame side by side.
"""
import torch
from native_stages import preprocess_sequence


def run_tap_sequence(model, decoded, *, source_motion_policy='search', return_inputs=False):
    if source_motion_policy not in ('raw', 'taa', 'search'):
        raise ValueError('source motion policy must be raw, taa or search')
    inputs = (decoded if source_motion_policy == 'raw' else
              preprocess_sequence(decoded, motion_policy=source_motion_policy))
    outputs = []
    for t in range(decoded.shape[1]):
        outputs.append(model(inputs[:, t], decoded[:, t]).clamp(0, 1))
    result = torch.stack(outputs, 1)
    return (result, inputs) if return_inputs else result
