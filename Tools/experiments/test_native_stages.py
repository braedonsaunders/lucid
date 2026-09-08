import unittest

import torch

from native_stages import StageSettings, deband, motion_blocks, preprocess_sequence, taa


class NativeStagesTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_plateau_and_texture_guard(self):
        x = torch.full((1, 1, 16, 24), .5)
        torch.testing.assert_close(deband(x), x)
        x[..., 8, 8] = .51
        self.assertEqual(float(deband(x)[0, 0, 8, 8]), float(x[0, 0, 8, 8]))
        unguarded = deband(x, settings=StageSettings(guard=0, threshold=.02))
        self.assertLess(float(unguarded[0, 0, 8, 8]), float(x[0, 0, 8, 8]))

    def test_stationary_and_known_motion(self):
        torch.manual_seed(27)
        previous = torch.rand(1, 1, 32, 48)
        field = motion_blocks(previous, previous)
        self.assertTrue((field[:, :2] == 0).all())
        self.assertTrue((field[:, 2] == 1).all())
        current = previous.roll(2, -1)
        field = motion_blocks(current, previous)
        torch.testing.assert_close(field[0, 0, 1:-1, 1:-1], torch.full((2, 4), -2.))
        torch.testing.assert_close(field[0, 1, 1:-1, 1:-1], torch.zeros(2, 4))

    def test_temporal_reset_cut_and_clipping(self):
        x = torch.full((1, 1, 16, 24), .4)
        self.assertIs(taa(x, None, x, None, None), x)
        field = torch.zeros(1, 4, 2, 3)
        field[:, 2] = 1
        # A cut in raw luma rejects history even with a confident motion field.
        out = taa(x, torch.ones_like(x), x, torch.ones_like(x), field)
        torch.testing.assert_close(out, x)
        # History beyond the local variance envelope cannot alter a flat plane.
        out = taa(x, torch.ones_like(x), x, x, field)
        torch.testing.assert_close(out, x)

    def test_sequence_forward_backward_and_reset(self):
        torch.manual_seed(4)
        x = (torch.rand(1, 3, 3, 16, 24) * .02 + .45).requires_grad_()
        out = preprocess_sequence(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        out.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        torch.testing.assert_close(preprocess_sequence(x.detach()[:, :1]), out.detach()[:, :1])


if __name__ == '__main__':
    unittest.main()
