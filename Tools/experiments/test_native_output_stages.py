import unittest

import torch

from native_output_stages import grain_scale, grade_luma, postprocess_rgb, sharpen_luma


class NativeOutputTest(unittest.TestCase):
    def test_constant_sharpen_and_grain_floor(self):
        x = torch.full((2, 1, 16, 24), .5)
        torch.testing.assert_close(sharpen_luma(x), x)
        torch.testing.assert_close(grain_scale(x), torch.full((2, 1, 1, 1), .25))

    def test_grade_video_range(self):
        x = torch.tensor([16 / 255, 235 / 255]).reshape(1, 1, 1, 2)
        torch.testing.assert_close(grade_luma(x, 0), x)

    def test_presented_output_reaches_model_gradient(self):
        torch.manual_seed(7)
        prediction = (torch.rand(2, 3, 32, 48) * .8 + .1).requires_grad_()
        raw = torch.rand(2, 3, 16, 24)
        y = postprocess_rgb(prediction, raw)
        self.assertEqual(y.shape, prediction.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        self.assertGreaterEqual(float(y.detach().min()), 0)
        self.assertLessEqual(float(y.detach().max()), 1)


if __name__ == '__main__':
    unittest.main()
