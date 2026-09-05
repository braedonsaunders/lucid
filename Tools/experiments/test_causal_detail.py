import sys
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail import CausalDetail


class CausalDetailTests(unittest.TestCase):
    def test_untrained_output_is_baseline_for_each_scale(self):
        torch.manual_seed(13)
        for scale in (2, 4):
            model = CausalDetail(channels=16, blocks=2, scale=scale).eval()
            frame = torch.rand(1, 3, 24, 32)
            output, state = model(frame, model.initial_state(frame), torch.zeros(1, 1, 1, 1))
            torch.testing.assert_close(output, F.interpolate(frame, scale_factor=scale, mode='bilinear', align_corners=False))
            self.assertEqual(tuple(state.shape), (1, 16, 6, 8))

    def test_reset_erases_prior_clip_and_state_stays_bounded(self):
        torch.manual_seed(27)
        model = CausalDetail(channels=16, blocks=2)
        torch.nn.init.normal_(model.head.weight, std=0.02)
        frame = torch.rand(1, 3, 24, 32)
        zero = model.initial_state(frame)
        with torch.inference_mode():
            a, s = model(frame, zero, torch.zeros(1, 1, 1, 1))
            b, reset = model(frame, torch.rand_like(zero), torch.zeros(1, 1, 1, 1))
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(s, reset)
            for _ in range(100):
                _, s = model(torch.rand_like(frame), s, torch.ones(1, 1, 1, 1))
                self.assertLessEqual(float(s.abs().max()), 1)

    def test_temporal_state_changes_reconstruction_after_head_learns(self):
        torch.manual_seed(18)
        model = CausalDetail(channels=16, blocks=2)
        torch.nn.init.normal_(model.head.weight, std=0.02)
        previous, current = torch.rand(1, 3, 24, 32), torch.rand(1, 3, 24, 32)
        zero = model.initial_state(current)
        _, history = model(previous, zero, torch.zeros(1, 1, 1, 1))
        temporal, _ = model(current, history, torch.ones(1, 1, 1, 1))
        independent, _ = model(current, zero, torch.zeros(1, 1, 1, 1))
        self.assertGreater(float((temporal-independent).abs().max().detach()), 1e-5)
        loss = temporal.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.history.weight.grad).all())
        self.assertGreater(float(model.history.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
