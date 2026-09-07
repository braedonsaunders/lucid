import unittest

import torch
from torch.nn import functional as F

from train_presented_detail import reconstruction_objective, sobel_loss, fft_loss, bounded_adversarial_scale, augment_input_noise, temporal_consistency


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

    def test_reference_intended_is_plain_reference_regression(self):
        torch.manual_seed(43)
        output = torch.rand(1, 3, 24, 24, requires_grad=True)
        reference = torch.rand_like(output)
        expected = (1.1 * F.l1_loss(output, reference) + .2 * sobel_loss(output, reference)
                    + .05 * fft_loss(output, reference))
        torch.testing.assert_close(reconstruction_objective(output, reference, reference, 'reference'), expected)

    def test_input_noise_is_identity_at_zero_and_bounded(self):
        torch.manual_seed(7)
        x = torch.rand(3, 3, 16, 16)
        self.assertTrue(torch.equal(augment_input_noise(x, 0), x))
        g = torch.Generator().manual_seed(1)
        y = augment_input_noise(x, .02, g)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(float(y.min()) >= 0 and float(y.max()) <= 1)
        self.assertGreater(float((y - x).abs().mean()), 0)
        self.assertLess(float((y - x).abs().max()), .2)

    def test_temporal_consistency_only_counts_still_pixels(self):
        ref0 = torch.zeros(1, 3, 16, 16); ref1 = ref0.clone(); ref1[..., :, 8:] = .5   # right half moved
        out0 = torch.zeros(1, 3, 16, 16); out1 = out0.clone(); out1[..., :, :8] = .1; out1[..., :, 8:] = .9
        loss, covered = temporal_consistency(out0, out1, ref0, ref1)
        self.assertTrue(.4 <= covered <= .5)   # 3x3 pooling widens the moving edge by a pixel
        self.assertAlmostEqual(float(loss), .1, places=2)   # only the still half counts
        zero, _ = temporal_consistency(out0, out0, ref0, ref1)
        self.assertEqual(float(zero), 0)


if __name__ == '__main__':
    unittest.main()
