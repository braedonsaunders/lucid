"""CPU regressions for the SSM paired-DINO adversary wiring (train_ssm_span).

Covers training_objective: zero weight is an
exact reconstruction no-op, positive weight conditions the critic on bicubic-2x
raw decoded LR (cropped 8), invalid weights/critics/geometries are rejected,
and the critic condition remains detached from the decoded observation.
"""
import unittest

import torch
from torch.nn import functional as F


LR = 24  # decoded LR size; bicubic 2x minus 8px borders -> 32px condition
HR = LR * 2 - 16


def _frames(batch=1):
    torch.manual_seed(0)
    output = (torch.rand(batch, 3, HR, HR) * 0.8 + 0.1).requires_grad_(True)
    reference = torch.zeros(batch, 3, HR, HR)
    decoded = (torch.rand(batch, 3, LR, LR) * 0.8 + 0.1)
    return output, reference, decoded


def _expected_condition(decoded):
    return F.interpolate(decoded.detach(), scale_factor=2, mode='bicubic',
                         align_corners=False, antialias=True).clamp(0, 1)[..., 8:-8, 8:-8]


class _StrictCritic:
    """Raises if touched; proves zero weight never calls the critic."""

    def __init__(self):
        self.calls = 0

    def generator_loss(self, image, condition):
        self.calls += 1
        raise AssertionError('critic must not be called at zero weight')


class _RecordingCritic:
    """Sum-gan mock with real-critic detach semantics for fake features."""

    def __init__(self):
        self.calls = []
        self.fake = (torch.zeros(1).detach(), torch.zeros(1).detach())

    def generator_loss(self, image, condition):
        self.calls.append((image, condition))
        return image.sum(), self.fake


class ZeroWeightTests(unittest.TestCase):
    def test_zero_weight_exactly_preserves_value_and_gradient(self):
        from train_ssm_span import training_objective
        from train_presented_detail import reconstruction_objective
        output, reference, decoded = _frames()
        critic = _StrictCritic()
        loss, fake, metrics = training_objective(
            output, reference, decoded, adversary=critic, gan_weight=0)
        expected = reconstruction_objective(
            output.detach().clone().requires_grad_(True), reference, reference, 'reference')
        torch.testing.assert_close(loss.detach(), expected.detach(), rtol=0, atol=0)
        self.assertIsNone(fake)
        self.assertEqual(set(metrics), {'reconstruction'})
        # Same graph, so the output gradient must match reconstruction exactly.
        grad_output = torch.autograd.grad(loss, output)[0]
        probe = output.detach().clone().requires_grad_(True)
        probe_recon = reconstruction_objective(probe, reference, reference, 'reference')
        torch.testing.assert_close(
            grad_output, torch.autograd.grad(probe_recon, probe)[0], rtol=0, atol=0)

    def test_zero_weight_never_calls_provided_critic(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames()
        critic = _StrictCritic()
        training_objective(output, reference, decoded, adversary=critic, gan_weight=0)
        self.assertEqual(critic.calls, 0)

    def test_zero_weight_without_critic_is_valid(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames()
        loss, fake, metrics = training_objective(
            output, reference, decoded, gan_weight=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(fake)


class PositiveWeightTests(unittest.TestCase):
    def test_condition_is_bicubic_raw_decoded_cropped8(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames()
        critic = _RecordingCritic()
        training_objective(output, reference, decoded,
                           adversary=critic, gan_weight=0.05)
        self.assertEqual(len(critic.calls), 1)
        image, condition = critic.calls[0]
        self.assertIs(image, output)
        self.assertEqual(condition.shape, output.shape)
        torch.testing.assert_close(condition, _expected_condition(decoded),
                                   rtol=1e-5, atol=1e-7)
        # Not the reference and not a processed input: conditions built from
        # same-geometry wrong sources (zeros like the reference here, and a
        # rescaled decoded stand-in) must both differ.
        for wrong in (torch.zeros_like(decoded), (decoded * 0.5).clamp(0, 1)):
            other = _expected_condition(wrong)
            self.assertEqual(other.shape, condition.shape)
            self.assertGreater(float((condition - other).abs().max()), 1e-3)

    def test_adversarial_gradient_reaches_output_condition_detached(self):
        from train_ssm_span import training_objective
        from train_presented_detail import reconstruction_objective
        weight = 0.05
        output, reference, decoded = _frames()
        decoded.requires_grad_(True)
        critic = _RecordingCritic()
        loss, fake, metrics = training_objective(
            output, reference, decoded, adversary=critic, gan_weight=weight)
        self.assertIs(fake, critic.fake)
        self.assertIn('gan_unweighted', metrics)
        torch.testing.assert_close(
            torch.tensor(metrics['gan_unweighted']),
            output.detach().sum(), rtol=1e-4, atol=1e-4)
        loss.backward()
        probe = output.detach().clone().requires_grad_(True)
        reconstruction_objective(probe, reference, reference, 'reference').backward()
        # gan = output.sum(), so the total gradient is recon grad + weight.
        torch.testing.assert_close(output.grad, probe.grad + weight,
                                   rtol=1e-4, atol=1e-4)
        self.assertGreater(float((output.grad - probe.grad).abs().max()), 0)
        _, condition = critic.calls[0]
        self.assertFalse(condition.requires_grad)
        self.assertIsNone(condition.grad_fn)
        self.assertIsNone(decoded.grad)


class RejectionTests(unittest.TestCase):
    def test_rejects_negative_and_nonfinite_weights(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames()
        critic = _RecordingCritic()
        for bad in (-1.0, -1e-6, float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                training_objective(output, reference, decoded,
                                   adversary=critic, gan_weight=bad)

    def test_positive_weight_requires_critic(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames()
        with self.assertRaisesRegex(ValueError, 'adversary'):
            training_objective(output, reference, decoded,
                               adversary=None, gan_weight=0.05)

    def test_geometry_mismatch_rejected_before_critic_call(self):
        from train_ssm_span import training_objective
        output, reference, _ = _frames()
        wrong_lr = torch.rand(1, 3, 20, 20)  # 40 - 16 = 24 != 32
        critic = _RecordingCritic()
        with self.assertRaisesRegex(ValueError, 'condition must match'):
            training_objective(output, reference, wrong_lr,
                               adversary=critic, gan_weight=0.05)
        self.assertEqual(len(critic.calls), 0)


if __name__ == '__main__':
    unittest.main()
