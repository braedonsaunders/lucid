import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_span import Unshuffled
from widen_span import widen


class WidenSpanTests(unittest.TestCase):
    def test_widening_preserves_function_in_both_modes(self):
        torch.manual_seed(1)
        model = Unshuffled(8, scale=2)
        # Random pretrained-like weights, not zeros: attention must be exercised.
        for p in model.parameters():
            p.data.normal_(0, .2)
        for m in model.modules():
            if hasattr(m, 'update_params'):
                m.update_params()
        wide = widen(model, 12)
        x = torch.rand(2, 3, 32, 40)
        with torch.no_grad():
            self.assertLess(float((model.eval()(x) - wide.eval()(x)).abs().max()), 1e-4)
            self.assertLess(float((model.train()(x) - wide.train()(x)).abs().max()), 1e-4)
        self.assertEqual(wide.core.conv_1.eval_conv.out_channels, 12)
        self.assertGreater(sum(p.numel() for p in wide.parameters()), sum(p.numel() for p in model.parameters()))

    def test_duplicates_are_not_identical_parameters(self):
        torch.manual_seed(2)
        model = Unshuffled(8, scale=2)
        for p in model.parameters():
            p.data.normal_(0, .2)
        wide = widen(model, 12)
        w = wide.core.block_1.c1_r.sk.weight
        self.assertFalse(torch.equal(w[:, 0], w[:, 8]))

    def test_rejects_narrowing(self):
        with self.assertRaises(ValueError):
            widen(Unshuffled(8, scale=2), 8)


if __name__ == '__main__':
    unittest.main()
