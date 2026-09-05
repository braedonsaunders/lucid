"""Training-only discriminator from pinned August 2026 PixRestore code.

This supplies distributional feedback missing from regression to sampled teacher
images. It is an adaptation of the released discriminator, not a new GAN claim.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

from dino_supervision import DinoSupervision

GAN_SHA256 = '21be7412da65dc94eb7048e3510ce8ec8e6b47ae241ff4679df0eba728f99b59'


class DinoAdversary:
    def __init__(self, repository, dino_repository, dino_checkpoint):
        path = Path(repository) / 'pixrestore/gan.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != GAN_SHA256:
            raise ValueError('use the pinned reviewed PixRestore discriminator')
        spec = importlib.util.spec_from_file_location('lucid_reviewed_pixrestore_gan', path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        self.features = DinoSupervision(dino_repository, dino_checkpoint, size=224).cuda().eval()
        self.discriminator = upstream.MultiLayerDinoDiscriminator(384, 6).cuda().train()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, weight_decay=0)
        self.metadata = {'upstream': 'https://github.com/csslc/PixRestore',
            'gan_sha256': GAN_SHA256, 'feature_encoder': self.features.metadata,
            'real_target': 'known training HQ reference', 'real_label': .8,
            'discriminator_optimizer': 'AdamW, lr 1e-4, weight_decay 0',
            'discriminator_loss': '0.5 * (real BCE + detached fake BCE), gradient clip 1',
            'inference_cost': 'none; feature encoder and discriminator discarded'}

    def generator_loss(self, image):
        self.discriminator.requires_grad_(False)
        features = self.features.features(image)
        return self.discriminator(features, real=True), [f.detach() for f in features]

    def update(self, reference, detached_fake):
        self.discriminator.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            real = self.features.features(reference)
        loss = .5 * (self.discriminator(real, real=True)
                     + self.discriminator([f.detach() for f in detached_fake], real=False))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite discriminator objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())
