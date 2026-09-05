"""Training-only hierarchical feature supervision inspired by PixRestore (Aug 2026).

This adapts its reverse reliability weighting to a causal reconstruction model;
it does not implement PixRestore's diffusion generator, conditioning router or
adversarial objective. No feature encoder is exported to the native application.
"""
import hashlib
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

DINOV2_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINOV2_WEIGHT_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def reverse_reliability(low_quality, reference, temperature=1.0):
    if temperature <= 0 or len(low_quality) != len(reference) or not reference:
        raise ValueError('matching nonempty feature layers and positive temperature required')
    with torch.no_grad():
        similarity = torch.stack([F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=1)
                                  for a, b in zip(low_quality, reference)], dim=-1)
        return torch.softmax((1-similarity)/temperature, dim=-1)


def hierarchical_loss(prediction, reference, weights):
    if len(prediction) != len(reference) or len(reference) != weights.shape[-1]:
        raise ValueError('feature layer count differs from weights')
    per_layer = torch.stack([(1-F.cosine_similarity(a.float(), b.detach().float(), dim=-1)).mean(dim=1)
                             for a, b in zip(prediction, reference)], dim=-1)
    if per_layer.shape != weights.shape:
        raise ValueError('per-example layer weights required')
    return (per_layer*weights.detach()).sum(dim=-1).mean()


class DinoSupervision(nn.Module):
    def __init__(self, repository, checkpoint, size=224, temperature=1.0):
        super().__init__()
        repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if size < 28 or size % 14 or temperature <= 0:
            raise ValueError('DINO size must be divisible by 14; temperature must be positive')
        revision = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True).strip()
        if revision != DINOV2_COMMIT or dirty:
            raise ValueError('use the recorded clean official DINOv2 checkout')
        weight_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if weight_hash != DINOV2_WEIGHT_SHA256:
            raise ValueError('DINOv2 weights do not match the recorded official download')
        self.encoder = torch.hub.load(str(repository), 'dinov2_vits14', source='local', pretrained=False)
        self.encoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.encoder.requires_grad_(False).eval()
        self.layers = (1, 2, 4, 6, 8, 10)
        self.size, self.temperature = size, temperature
        self.register_buffer('mean', torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.metadata = {'inspiration': 'https://github.com/csslc/PixRestore',
            'reviewed_pixrestore_commit': '909ca06e614cf72e218d52b4fa41077c603d872b',
            'encoder_repository': 'https://github.com/facebookresearch/dinov2',
            'encoder_commit': revision, 'encoder_weight_sha256': weight_hash,
            'layers': list(self.layers), 'size': size, 'temperature': temperature,
            'normalization': 'ImageNet RGB; raw intermediate patch tokens; cosine distance',
            'weighting': 'softmax((1-mean_LQ_HQ_cosine)/temperature), detached',
            'inference_cost': 'none; used only for training'}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, image):
        image = F.interpolate(image.float(), (self.size, self.size), mode='bicubic',
                              align_corners=False, antialias=True).clamp(0, 1)
        return self.encoder.get_intermediate_layers((image-self.mean)/self.std,
                                                    n=self.layers, norm=False)

    def forward(self, prediction, reference, low_quality):
        # Teacher targets and reliability never learn to excuse a model error.
        with torch.no_grad():
            target = self.features(reference)
            low = self.features(low_quality)
            weights = reverse_reliability(low, target, self.temperature)
        predicted = self.features(prediction)
        return hierarchical_loss(predicted, target, weights)
