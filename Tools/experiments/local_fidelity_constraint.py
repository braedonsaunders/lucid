"""Training-only reference-relative color/edge constraints, independent of eval metrics."""
import torch
from torch.nn import functional as F


def signed_edges(x):
    kernel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
    kernels = torch.stack((kernel, kernel.t()))[:, None]
    n, c, h, w = x.shape
    return F.conv2d(x.reshape(n*c, 1, h, w), kernels).reshape(n, c*2, h-2, w-2)


def violations(output, reference, baseline):
    """Positive local squared-error excess over cached shipping, in two domains.

    Signed gradients retain phase and color. No LPIPS, DISTS, Pearson correlation,
    validation frames, or reference-guided inference enter this objective.
    """
    if output.shape != reference.shape or output.shape != baseline.shape:
        raise ValueError('aligned output/reference/baseline shapes required')
    if output.ndim != 4 or min(output.shape[-2:]) < 10:
        raise ValueError('NCHW images of at least 10 pixels required')
    reference, baseline = reference.detach(), baseline.detach()
    def pool(x):
        return F.avg_pool2d(x.mean(1, keepdim=True), 8, ceil_mode=True, count_include_pad=False)
    results = []
    for transform in (lambda x: x, signed_edges):
        actual, target, control = map(transform, (output, reference, baseline))
        anchor_error = pool((control-target).square())
        excess = pool((actual-target).square())-anchor_error
        # Stable on flat patches; independent of candidate error and detached.
        energy = pool(target.square())+anchor_error+1e-4
        results.append((excess/energy).clamp_min(0).mean())
    return torch.stack(results)


class AdaptiveFidelityPenalty:
    """Nonnegative multiplier ascent on local violations; no convergence guarantee."""
    def __init__(self, device):
        self.multipliers = torch.ones(2, device=device)
        self.rate = .05
        self.rho = 10.

    def loss(self, values):
        return (self.multipliers.detach()*values+.5*self.rho*values.square()).sum()

    @torch.no_grad()
    def update(self, values):
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('finite nonnegative violations required')
        self.multipliers.add_(self.rate*values.detach())

