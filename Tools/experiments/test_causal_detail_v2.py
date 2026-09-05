import sys
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.causal_detail_v2 import Lanczos2x, CausalDetailV2


class SpatialEvidenceTests(unittest.TestCase):
    def test_constant_color_and_channel_isolation_at_borders(self):
        floor = Lanczos2x()
        frame = torch.tensor([0.1, 0.5, 0.9]).reshape(1, 3, 1, 1).expand(1, 3, 8, 12)
        torch.testing.assert_close(floor(frame), frame[:, :, :1, :1].expand(1, 3, 16, 24))
        red = torch.zeros(1, 3, 8, 12); red[0, 0, 4, 6] = 1
        self.assertEqual(float(floor(red)[:, 1:].abs().max()), 0)

    def test_bandlimited_detail_is_more_faithful_than_bilinear(self):
        # Continuous signal supplies a reference independent of either kernel.
        def wave(positions):
            return 0.5 + 0.2*torch.sin(2*torch.pi*0.30*positions)
        frame = wave(torch.arange(64)).reshape(1, 1, 1, 64).expand(1, 3, 12, 64)
        target = wave((torch.arange(128)+0.5)/2-0.5)[8:-8]
        actual = Lanczos2x()(frame)[0, 0, 12, 8:-8]
        baseline = F.interpolate(frame, scale_factor=2, mode='bilinear', align_corners=False)[0, 0, 12, 8:-8]
        self.assertLess(float((actual-target).square().mean()), float((baseline-target).square().mean())*0.1)

    def test_new_head_starts_at_floor_and_reset_removes_history(self):
        torch.manual_seed(14)
        model = CausalDetailV2(16, 2)
        frame = torch.rand(1, 3, 16, 20)
        zero = model.initial_state(frame)
        reset = torch.zeros(1, 1, 1, 1)
        output, _ = model(frame, zero, reset)
        torch.testing.assert_close(output, model.floor(frame))
        torch.nn.init.normal_(model.head.weight, std=0.01)
        a, sa = model(frame, zero, reset)
        b, sb = model(frame, torch.rand_like(zero), reset)
        torch.testing.assert_close(a, b); torch.testing.assert_close(sa, sb)
        a.square().mean().backward()
        bypass = model.head.weight.grad[:, model.channels:]
        self.assertGreater(float(bypass.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
