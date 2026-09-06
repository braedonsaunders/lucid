"""Input-conditioned, wrong-detail-negative adaptation of PixRestore's critic.

Training-only hypothesis; not a reproduction of DNF-SR or a novelty claim.
"""
import torch
from torch.nn import functional as F

from dino_adversary import DinoAdversary


def wrong_detail(reference):
    """Keep coarse content, displace signed fine detail two pixels horizontally.

    Replicate boundaries rather than wrap unrelated opposite image edges.
    """
    smooth = F.avg_pool2d(F.pad(reference, (1, 1, 1, 1), mode='replicate'), 3, 1)
    detail = reference - smooth
    shifted = F.pad(detail, (2, 0, 0, 0), mode='replicate')[..., :-2]
    return (smooth + shifted).clamp(0, 1)


def paired_features(output, condition):
    if len(output) != len(condition) or not output:
        raise ValueError('matching nonempty feature layers required')
    if any(a.shape != b.shape for a, b in zip(output, condition)):
        raise ValueError('condition and output tokens must align')
    return [torch.cat((a, b.detach()), dim=-1) for a, b in zip(output, condition)]


class PairedDinoAdversary(DinoAdversary):
    def __init__(self, repository, dino_repository, dino_checkpoint):
        super().__init__(repository, dino_repository, dino_checkpoint)
        # Use the already hash-verified upstream class; retain 96 hidden units.
        cls = type(self.discriminator)
        self.discriminator = cls(768, 6, hidden_ratio=.125).cuda().train()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, weight_decay=0)
        self.metadata.update(condition='detached DINO tokens of bicubic LR at matching output coordinates',
            features='concatenate output and condition tokens; 768 inputs, 96 hidden units',
            negatives='half generated output, half HR with signed 3x3 highpass displaced two pixels right',
            discriminator_loss='0.5 real BCE + 0.25 generated BCE + 0.25 wrong-detail BCE; clip 1',
            limitation='combined conditioning/negative recipe; not isolated component attribution')

    def generator_loss(self, image, condition):
        self.discriminator.requires_grad_(False)
        with torch.no_grad():
            low = self.features.features(condition)
        features = self.features.features(image)
        paired = paired_features(features, low)
        return self.discriminator(paired, real=True), ([f.detach() for f in features], low)

    def update(self, reference, detached_fake):
        fake, low = detached_fake
        self.discriminator.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            real = self.features.features(reference)
            negative = self.features.features(wrong_detail(reference))
        loss = (.5 * self.discriminator(paired_features(real, low), real=True)
                + .25 * self.discriminator(paired_features(fake, low), real=False)
                + .25 * self.discriminator(paired_features(negative, low), real=False))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite paired discriminator objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())
