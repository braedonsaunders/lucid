"""Training-only feature and edge objectives; inference graph is unchanged."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeatures(nn.Module):
    """Perceptual loss on VGG19 features, at the layers and weights Real-ESRGAN
    uses. Deliberately NOT LPIPS: LPIPS is what the result is judged by, and a
    model trained on its own scoreboard tells you nothing."""

    LAYERS = {"2": 0.1, "7": 0.1, "16": 1.0, "25": 1.0, "34": 1.0}

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = features[:35].eval().to(device)
        for p in self.slices.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, a, b):
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        loss = 0.0
        for index, layer in enumerate(self.slices):
            a, b = layer(a), layer(b)
            weight = self.LAYERS.get(str(index))
            if weight:
                loss = loss + weight * F.l1_loss(a, b)
        return loss


def sobel_loss(output, reference):
    """Match signed spatial gradients, rather than rewarding texture energy."""
    kernel = output.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernels = torch.stack((kernel, kernel.t())).unsqueeze(1)
    def edges(image):
        # Separate RGB channels: chroma edges are not interchangeable with luma.
        image = image.reshape(-1, 1, *image.shape[-2:])
        return F.conv2d(image, kernels)
    return F.l1_loss(edges(output), edges(reference))
