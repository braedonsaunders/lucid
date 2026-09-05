import unittest
import numpy as np
from PIL import Image
import torch
from quantized_presentation import bicubic_half


class PresentationTests(unittest.TestCase):
    def test_pillow_rgb_rounding_and_boundary_normalization(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(731)
        for height, width in [(8, 12), (32, 40), (144, 256)]:
            pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            expected = np.asarray(Image.fromarray(pixels).resize((width//2, height//2), Image.Resampling.BICUBIC)).astype(float)
            actual = bicubic_half(torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float())[0].permute(1, 2, 0).numpy()
            delta = np.abs(expected-actual)
            self.assertLessEqual(delta.max(), 1)
            self.assertLess(delta.mean(), .01)

    def test_constant_color_survives_borders(self):
        image = torch.tensor([0., 71., 255.]).reshape(1, 3, 1, 1).expand(1, 3, 32, 40)
        output = bicubic_half(image)
        self.assertTrue(torch.equal(output, image[:, :, ::2, ::2]))


if __name__ == '__main__':
    unittest.main()
