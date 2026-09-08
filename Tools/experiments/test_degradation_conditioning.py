import copy
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.degradation_conditioning import (DegradationConditionedSPAN,
    DegradationEstimator, local_curvature, oracle_degradation)
from eval_checkpoint import load
from train_span import Unshuffled
from train_joint_conditioning import joint_objective


class DegradationConditionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)

    def test_clean_texture_is_not_oracle_damage(self):
        texture = torch.rand(1, 3, 24, 32)
        self.assertGreater(float(local_curvature(texture).abs().mean()), .1)
        self.assertEqual(float(oracle_degradation(texture, texture).sum()), 0)
        corrupted = (texture + .03).clamp(0, 1)
        damage = oracle_degradation(corrupted, texture)
        self.assertGreater(float(damage.mean()), .1)
        self.assertTrue(((damage >= 0) & (damage <= 1)).all())

    def test_identity_and_frozen_backbone_in_float_and_autocast(self):
        source = torch.rand(1, 3, 24, 32)
        base = Unshuffled(4, frames=1, scale=2).eval()
        model = DegradationConditionedSPAN(copy.deepcopy(base), estimator_channels=4).train()
        torch.testing.assert_close(model(source), base(source), rtol=0, atol=0)
        state = {k: v.clone() for k, v in model.sr.state_dict().items()}
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model(source)
            expected = model.sr(source)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.float().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
        self.assertGreater(float(model.modulation.weight.grad.abs().sum()), 0)
        for key, value in model.sr.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_checkpoint_round_trip_and_oracle_cannot_be_deployed(self):
        model = DegradationConditionedSPAN(Unshuffled(4, scale=2), estimator_channels=4).eval()
        with torch.no_grad():
            model.modulation.bias.fill_(.1)
        state = {'model': model.state_dict(), 'architecture': 'degradation_conditioned_span2x',
                 'scale': 2, 'frames': 1, 'channels': 4, 'version': 1,
                 'conditioning_mode': 'estimated', 'estimator_channels': 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(state, path)
            restored, _, frames = load(path, 'cpu')
            source = torch.rand(1, 3, 16, 24)
            torch.testing.assert_close(restored(source), model(source), rtol=0, atol=0)
            self.assertEqual(frames, 1)
            state['conditioning_mode'] = 'oracle'
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load(path, 'cpu')
        model.mode = 'oracle'
        with self.assertRaises(ValueError):
            model(source)

    def test_estimator_has_finite_local_input_only_output(self):
        estimator = DegradationEstimator(4)
        x = torch.rand(1, 3, 40, 48)
        full = estimator(x)
        cropped = estimator(x[..., 8:32, 8:40])
        torch.testing.assert_close(full[..., 12:28, 12:36], cropped[..., 4:-4, 4:-4], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(full).all())
        self.assertEqual(full.shape, (1, 2, 40, 48))

    def test_joint_training_updates_backbone_modulation_and_estimator(self):
        model = DegradationConditionedSPAN(Unshuffled(4, scale=2), estimator_channels=4,
                                          train_backbone=True).train()
        source = torch.rand(1, 3, 24, 32)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        condition = model.estimator(source)
        output = model.conditioned(source, condition.detach())
        calibration = (condition - oracle_degradation(source, source * .9)).abs().mean()
        (output.square().mean() + .5 * calibration).backward()
        optimizer.step()
        after = model.state_dict()
        for prefix in ('sr.', 'modulation.', 'estimator.'):
            self.assertTrue(any(not torch.equal(v, after[k]) for k, v in before.items()
                                if k.startswith(prefix)), prefix)

    def test_disabled_control_round_trip_and_metadata_mismatch(self):
        model = DegradationConditionedSPAN(Unshuffled(4, scale=2), estimator_channels=4,
                                          train_backbone=True, use_conditioning=False)
        with torch.no_grad():
            model.modulation.bias.fill_(.5)
        source = torch.rand(1, 3, 24, 32)
        torch.testing.assert_close(model(source), model.sr(source), rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in model.modulation.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.estimator.parameters()))
        state = {'model': model.state_dict(), 'architecture': 'degradation_conditioned_span2x',
                 'scale': 2, 'frames': 1, 'channels': 4, 'version': 1,
                 'conditioning_mode': 'estimated', 'estimator_channels': 4,
                 'use_conditioning': False, 'experiment': {'use_conditioning': False}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(state, path)
            restored, _, _ = load(path, 'cpu')
            torch.testing.assert_close(restored(source), model(source), rtol=0, atol=0)
            state['experiment']['use_conditioning'] = True
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'policies differ'):
                load(path, 'cpu')

    def test_calibration_does_not_change_reconstruction_gradient(self):
        output = torch.rand(1, 3, 48, 64, requires_grad=True)
        reference = torch.rand_like(output)
        condition = torch.rand(1, 2, 24, 32, requires_grad=True)
        oracle = torch.zeros_like(condition)
        zero = torch.autograd.grad(joint_objective(output, reference, condition, oracle, 0),
                                   (output, condition), retain_graph=True)
        calibrated = torch.autograd.grad(joint_objective(output, reference, condition, oracle, .5),
                                         (output, condition))
        torch.testing.assert_close(zero[0], calibrated[0], rtol=0, atol=0)
        self.assertEqual(float(zero[1].abs().sum()), 0)
        self.assertGreater(float(calibrated[1].abs().sum()), 0)
        self.assertEqual(float(calibrated[1][..., :4, :].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
