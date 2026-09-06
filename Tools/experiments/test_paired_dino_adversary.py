import unittest

import torch

from paired_dino_adversary import paired_features, wrong_detail


class PairedCriticTests(unittest.TestCase):
    def test_condition_is_detached_but_prediction_learns(self):
        prediction = torch.randn(2, 16, 4, requires_grad=True)
        condition = torch.randn(2, 16, 4, requires_grad=True)
        result = paired_features([prediction], [condition])[0]
        self.assertTrue(torch.equal(result[..., :4], prediction))
        self.assertTrue(torch.equal(result[..., 4:], condition))
        result.square().sum().backward()
        self.assertIsNone(condition.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_wrong_detail_preserves_flat_and_changes_texture(self):
        flat = torch.full((1, 3, 16, 16), .5)
        self.assertTrue(torch.equal(wrong_detail(flat), flat))
        textured = flat.clone()
        textured[..., 8, 8] = .75
        result = wrong_detail(textured)
        self.assertFalse(torch.equal(result, textured))
        self.assertTrue(torch.equal(result[..., :4], textured[..., :4]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_misaligned_features_rejected(self):
        with self.assertRaises(ValueError):
            paired_features([torch.zeros(1, 2, 3)], [torch.zeros(1, 3, 3)])


if __name__ == '__main__':
    unittest.main()
