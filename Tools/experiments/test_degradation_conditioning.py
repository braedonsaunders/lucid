import copy
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.degradation_conditioning import (DegradationConditionedSPAN,
    DegradationEstimator, local_curvature, oracle_degradation)
from eval_checkpoint import load
from train_span import Unshuffled


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


if __name__ == '__main__':
    unittest.main()
