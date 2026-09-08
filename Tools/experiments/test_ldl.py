"""Check LDL values AND gradients against the author's mathematical definition."""
import unittest

import torch
from torch.nn import functional as F

from train_presented_detail import ldl_artifact_map, ldl_loss, update_ema


class LDLTest(unittest.TestCase):
    def test_reference_values_and_gradients(self):
        torch.manual_seed(19)
        output = torch.rand(2, 3, 16, 20, dtype=torch.float64, requires_grad=True)
        reference = torch.rand_like(output)
        ema = torch.rand_like(output)
        residual = (reference - output).abs().sum(1, keepdim=True)
        windows = F.pad(residual, (3, 3, 3, 3), mode='reflect').unfold(2, 7, 1).unfold(3, 7, 1)
        expected = residual.var(dim=(1, 2, 3), keepdim=True).pow(.2) * windows.var(dim=(-1, -2))
        expected = torch.where(residual < (reference - ema).abs().sum(1, keepdim=True), 0, expected)
        actual = ldl_artifact_map(output, reference, ema)
        torch.testing.assert_close(actual, expected)
        a = ldl_loss(output, reference, ema)
        b = (expected * (output - reference).abs()).mean()
        torch.testing.assert_close(torch.autograd.grad(a, output)[0],
                                   torch.autograd.grad(b, output)[0])

    def test_perfect_and_uniform_residuals_have_finite_gradients(self):
        for offset in (0., .05):
            out = torch.full((1, 3, 12, 12), .5 + offset, requires_grad=True)
            ref = torch.full_like(out, .5)
            loss = ldl_loss(out, ref, ref)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0)
            self.assertTrue(torch.isfinite(out.grad).all())

    def test_better_than_ema_is_masked_and_ema_has_no_gradient(self):
        out = torch.full((1, 3, 12, 12), .51, requires_grad=True)
        ref = torch.full_like(out, .5)
        ema = torch.full_like(out, .7, requires_grad=True)
        loss = ldl_loss(out, ref, ema)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertIsNone(ema.grad)

    def test_legacy_map_is_detached_and_differs_at_borders(self):
        torch.manual_seed(3)
        out = torch.rand(1, 3, 12, 12, requires_grad=True)
        ref = torch.zeros_like(out)
        legacy = ldl_artifact_map(out, ref, ref, formulation='legacy')
        actual = ldl_artifact_map(out, ref, ref)
        self.assertFalse(legacy.requires_grad)
        self.assertTrue(actual.requires_grad)
        self.assertFalse(torch.equal(actual, legacy))

    def test_ema_updates_parameters_and_buffers_without_tracking_gradients(self):
        model, ema = torch.nn.BatchNorm2d(2), torch.nn.BatchNorm2d(2)
        with torch.no_grad():
            model.weight.fill_(3)
            model.running_mean.fill_(4)
            model.num_batches_tracked.fill_(7)
        update_ema(ema, model, .5)
        torch.testing.assert_close(ema.weight, torch.full((2,), 2.))
        torch.testing.assert_close(ema.running_mean, torch.full((2,), 2.))
        self.assertEqual(int(ema.num_batches_tracked), 7)


if __name__ == '__main__':
    unittest.main()
