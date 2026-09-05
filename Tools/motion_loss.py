"""Occlusion-gated temporal consistency with bounded correspondence on CUDA.

Alignment is training-only. A single-frame Core ML student still takes one
RGB image, so this objective adds no neural inference cost on the Mac.
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def correspondence(current, previous, radius=8, block=8):
    gray = current.mean(1, keepdim=True)
    old = previous.mean(1, keepdim=True)
    n, _, h, w = gray.shape
    padded = F.pad(old, (radius, radius, radius, radius), mode='replicate')
    offsets = [(x, y) for y in range(-radius, radius + 1, 2) for x in range(-radius, radius + 1, 2)]
    costs = []
    for x, y in offsets:
        shifted = padded[:, :, radius + y:radius + y + h, radius + x:radius + x + w]
        costs.append(F.avg_pool2d((gray - shifted).abs(), block, ceil_mode=True) + .00015 * (x*x + y*y))
    choice = torch.cat(costs, dim=1).argmin(1)
    vectors = torch.tensor(offsets, device=current.device, dtype=current.dtype)[choice].permute(0, 3, 1, 2)
    flow = F.interpolate(vectors, size=(h, w), mode='nearest')
    aligned = warp(previous, flow)
    confidence = (1 - (current - aligned).abs().mean(1, keepdim=True) / .04).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(h, device=current.device), torch.arange(w, device=current.device), indexing='ij')
    inside = ((xx + flow[:, 0] >= 0) & (xx + flow[:, 0] < w) & (yy + flow[:, 1] >= 0) & (yy + flow[:, 1] < h)).unsqueeze(1)
    return flow, confidence * inside

def warp(image, flow):
    n, _, h, w = image.shape
    fh, fw = flow.shape[-2:]
    flow = F.interpolate(flow, size=(h, w), mode='nearest')
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype), torch.arange(w, device=image.device, dtype=image.dtype), indexing='ij')
    gx = 2 * (xx + .5 + flow[:, 0] * w / fw) / w - 1
    gy = 2 * (yy + .5 + flow[:, 1] * h / fh) / h - 1
    return F.grid_sample(image, torch.stack((gx, gy), -1), mode='bilinear', padding_mode='border', align_corners=False)

def aligned_temporal_loss(current_out, previous_out, current_in, previous_in):
    flow, confidence = correspondence(current_in, previous_in)
    previous = warp(previous_out, flow)
    weight = F.interpolate(confidence, size=current_out.shape[-2:], mode='nearest')
    error = (current_out - previous).abs().mean(1, keepdim=True)
    return (error * weight).sum() / weight.sum().clamp(min=1)
