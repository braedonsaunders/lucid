import unittest

import torch
from torch.nn import functional as F

from train_presented_detail import reconstruction_objective, sobel_loss, fft_loss, bounded_adversarial_scale


class PresentedObjectiveTest(unittest.TestCase):
    def test_gradient_budget_caps_without_amplification_or_second_derivatives(self):
        base = torch.tensor([1., 1., 0., 0.], requires_grad=True)
        adversarial = torch.tensor([10., .01, 1., 0.], requires_grad=True)
        scale = bounded_adversarial_scale(base, adversarial, .15)
        self.assertFalse(scale.requires_grad)
        torch.testing.assert_close(scale, torch.tensor([.015, 1., 0., 0.]))
        self.assertTrue(torch.all(scale * adversarial <= .15 * base + 1e-7))
        for invalid in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                bounded_adversarial_scale(base, adversarial, invalid)

    def test_budget_controls_actual_head_gradient(self):
        head = torch.tensor([1., -1.], requires_grad=True)
        base = head.square().sum()
        gan = 100 * head.sum()
        base_gradient = torch.autograd.grad(base, head, retain_graph=True)[0]
        gan_gradient = torch.autograd.grad(gan, head, retain_graph=True)[0]
        scale = bounded_adversarial_scale(base_gradient.norm(), gan_gradient.norm(), .15)
        combined = torch.autograd.grad(base + scale * gan, head)[0]
        torch.testing.assert_close(combined - base_gradient, gan_gradient * scale)
        self.assertLessEqual(float((combined-base_gradient).norm() / base_gradient.norm()), .150001)

    def test_default_preserves_control_value_and_gradient(self):
        torch.manual_seed(41)
        output = torch.rand(1, 3, 24, 24, requires_grad=True)
        reference, intended = torch.rand_like(output), torch.rand_like(output)
        old = (F.l1_loss(output, intended) + .2 * sobel_loss(output, intended)
               + .05 * fft_loss(output, intended) + .1 * F.l1_loss(output, reference))
        actual = reconstruction_objective(output, reference, intended)
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, output)[0],
                                   torch.autograd.grad(old, output)[0], rtol=0, atol=0)

    def test_reference_details_change_gradient_without_changing_pixel_terms(self):
        torch.manual_seed(42)
        output = torch.rand(1, 3, 24, 24, requires_grad=True)
        reference = torch.ones_like(output) * .5
        intended = torch.rand_like(output)
        difference = (reconstruction_objective(output, reference, intended, 'reference')
                      - reconstruction_objective(output, reference, intended, 'mixture'))
        expected = (.2 * (sobel_loss(output, reference) - sobel_loss(output, intended))
                    + .05 * (fft_loss(output, reference) - fft_loss(output, intended)))
        torch.testing.assert_close(difference, expected)
        gradient = torch.autograd.grad(difference, output)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        with self.assertRaises(ValueError):
            reconstruction_objective(output, reference, intended, 'invalid')


if __name__ == '__main__':
    unittest.main()
