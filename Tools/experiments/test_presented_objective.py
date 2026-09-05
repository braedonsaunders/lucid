import unittest

import torch
from torch.nn import functional as F

from train_presented_detail import reconstruction_objective, sobel_loss, fft_loss


class PresentedObjectiveTest(unittest.TestCase):
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
