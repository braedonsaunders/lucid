import sys
from pathlib import Path
import unittest
import tempfile

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_span import Unshuffled
from fold_shipping_head import fold_head
from eval_checkpoint import load


class HeadFoldingTest(unittest.TestCase):
    def test_trained_phase_order_and_border_equivalence(self):
        torch.manual_seed(11)
        model = Unshuffled(channels=4).eval()
        folded = fold_head(model)
        image = torch.rand(1, 3, 12, 18)
        with torch.inference_mode():
            expected = F.avg_pool2d(model(image), 2)
            actual = folded(image)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertIsInstance(model.core.upsampler[1], nn.PixelShuffle)
        self.assertEqual(model.core.upsampler[1].upscale_factor, 8)

    def test_wrong_scale_rejected(self):
        with self.assertRaises(ValueError):
            fold_head(Unshuffled(channels=4, scale=2))

    def test_training_and_checkpoint_roundtrip(self):
        model = fold_head(Unshuffled(channels=4)).train()
        image = torch.rand(1, 3, 12, 18)
        loss = model(image).square().mean()
        loss.backward()
        self.assertGreater(float(model.core.upsampler[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.core.block_1.c1_r.conv[0].weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'direct2x.pth'
            torch.save({'model': model.state_dict(), 'channels': 4, 'scale': 2}, path)
            restored, _, _ = load(path, 'cpu')
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), model.eval()(image))


if __name__ == '__main__':
    unittest.main()
