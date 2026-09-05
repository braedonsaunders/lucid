"""Reference-checked, training-only output distillation; no inference modules."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_teacher_cache(directory, bank_manifest_hash, data, digest):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['bank_sha256'] != bank_manifest_hash:
        raise ValueError('teacher cache belongs to another bank')
    expected = {identity: hr.shape for _, hr, identity in data['train']}
    restored = {}
    for row in manifest['sequences']:
        identity = row['id']
        if identity not in expected or identity in restored:
            raise ValueError('teacher cache has unknown or duplicate training sequence')
        path = directory / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('teacher sequence changed')
        with np.load(path, allow_pickle=False) as pair:
            pixels = pair['teacher'].copy()
        if pixels.dtype != np.uint8 or pixels.shape != expected[identity]:
            raise ValueError('teacher dimensions differ from training reference')
        restored[identity] = pixels
    if set(restored) != set(expected):
        raise ValueError('teacher cache is incomplete')
    return restored, manifest


def reference_checked_loss(output, reference, teacher, floor):
    """Use teacher only where its local RGB error is lower than the fixed floor.

    The mask uses known training references, never evaluation references. Its
    mean is not normalized away: unhelpful teacher regions contribute no loss.
    Original reference reconstruction objectives remain active everywhere.
    """
    if not (output.shape == reference.shape == teacher.shape == floor.shape):
        raise ValueError('distillation tensors must have identical shapes')
    with torch.no_grad():
        def local_error(image):
            error = (image - reference).abs().mean(dim=1, keepdim=True)
            return F.avg_pool2d(F.pad(error, (4, 4, 4, 4), mode='replicate'), 9, 1)
        teacher_error, floor_error = local_error(teacher), local_error(floor)
        # Confidence measures improvement, rather than merely accepting a tie.
        confidence = ((floor_error - teacher_error) / (floor_error + 1e-3)).clamp(0, 1)
    loss = ((output - teacher.detach()).abs() * confidence).mean()
    return loss, confidence.mean()
