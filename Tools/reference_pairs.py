"""Reference selection shared by every tuning entry point."""
from pathlib import Path
KNOWN = {'crowdrun-360p-350k.mp4': 'crowdrun-1080p.mp4', 'dinner-360p-350k.mp4': 'dinner-1080p.mp4'}
def resolve_reference(clip, explicit=None):
    clip = Path(clip)
    if not clip.is_file():
        raise ValueError(f'No clip at {clip}')
    if not explicit and clip.name not in KNOWN:
        raise ValueError(f'{clip.name} has no registered clean reference. Supply --reference from the same source and timeline; the compressed BBB reference is not an evaluation master.')
    reference = Path(explicit) if explicit else clip.parent / KNOWN[clip.name]
    if not reference.is_file():
        raise ValueError(f'No reference at {reference}')
    if reference.resolve() == clip.resolve():
        raise ValueError('Input and reference must be different files')
    return str(reference)
