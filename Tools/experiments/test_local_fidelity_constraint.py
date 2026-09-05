import unittest
import torch
from local_fidelity_constraint import violations, AdaptiveFidelityPenalty


class FidelityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(817)

    def test_identity_and_reference_improvement_cost_nothing(self):
        reference = torch.rand(2, 3, 32, 40)
        baseline = (reference+.05*torch.randn_like(reference)).clamp(0, 1)
        self.assertTrue(torch.equal(violations(baseline, reference, baseline), torch.zeros(2)))
        self.assertTrue(torch.equal(violations(reference, reference, baseline), torch.zeros(2)))

    def test_phase_shift_with_equal_energy_has_real_gradient(self):
        reference = torch.zeros(1, 3, 32, 40)
        reference[:, :, 8:24, 12:28] = 1
        output = torch.roll(reference, 2, -1).requires_grad_()
        self.assertEqual(float(output.detach().square().sum()), float(reference.square().sum()))
        values = violations(output, reference, reference)
        self.assertTrue((values > 0).all())
        values.sum().backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0)

    def test_flat_inputs_and_multiplier_update(self):
        x = torch.zeros(1, 3, 17, 19, requires_grad=True)
        values = violations(x, x.detach(), x.detach())
        penalty = AdaptiveFidelityPenalty('cpu')
        penalty.loss(values).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        penalty.update(values)
        self.assertTrue(torch.equal(penalty.multipliers, torch.ones(2)))
        penalty.update(torch.tensor([.2, .4]))
        self.assertTrue(torch.allclose(penalty.multipliers, torch.tensor([1.01, 1.02])))
        with self.assertRaises(ValueError):
            penalty.update(torch.tensor([float('nan'), 0.]))

    def test_reference_and_baseline_are_not_optimized(self):
        reference = torch.rand(1, 3, 20, 20, requires_grad=True)
        baseline = reference.detach().clone().requires_grad_()
        output = torch.rand_like(reference, requires_grad=True)
        violations(output, reference, baseline).sum().backward()
        self.assertIsNone(reference.grad)
        self.assertIsNone(baseline.grad)
        self.assertGreater(float(output.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
