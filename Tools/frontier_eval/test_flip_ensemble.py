import unittest

import torch

from score_checkpoint_frames import flip_ensemble


class FlipEnsembleTest(unittest.TestCase):
    def test_equivariant_model_preserves_pixels_and_non_square_geometry(self):
        torch.manual_seed(2)
        x = torch.randint(0, 256, (1, 3, 6, 10)).float() / 255
        model = torch.nn.Upsample(scale_factor=2, mode='nearest')
        torch.testing.assert_close(flip_ensemble(model, x), model(x))

    def test_orientation_bias_is_actually_averaged_over_four_calls(self):
        calls = []
        def model(x):
            calls.append(x.clone())
            return torch.linspace(0, 1, x.shape[-1])[None, None, None].expand_as(x)
        out = flip_ensemble(model, torch.zeros(1, 3, 4, 6))
        self.assertEqual(len(calls), 4)
        torch.testing.assert_close(out, torch.full_like(out, .5))


if __name__ == '__main__':
    unittest.main()
