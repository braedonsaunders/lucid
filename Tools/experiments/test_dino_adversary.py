import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dino_adversary import DinoAdversary


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4).requires_grad_(False)

    def features(self, image):
        return [self.encoder(image)]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)

    def forward(self, features, real=True):
        logits = self.head(features[0])
        return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, .8 if real else 0))


class AdversarialGradientTest(unittest.TestCase):
    def test_generator_and_discriminator_gradients_stay_separate(self):
        a = DinoAdversary.__new__(DinoAdversary)
        a.features, a.discriminator = Features(), Discriminator()
        a.optimizer = torch.optim.AdamW(a.discriminator.parameters(), lr=.01)
        image = torch.rand(2, 3, requires_grad=True)
        reference = torch.rand(2, 3, requires_grad=True)
        loss, detached = a.generator_loss(image)
        loss.backward()
        self.assertGreater(float(image.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in a.discriminator.parameters()))
        self.assertTrue(all(p.grad is None for p in a.features.parameters()))
        image_gradient = image.grad.clone()
        before = a.discriminator.head.weight.detach().clone()
        a.update(reference, detached)
        self.assertFalse(torch.equal(before, a.discriminator.head.weight))
        torch.testing.assert_close(image.grad, image_gradient)
        self.assertIsNone(reference.grad)
        self.assertTrue(all(p.grad is None for p in a.features.parameters()))


if __name__ == '__main__':
    unittest.main()
