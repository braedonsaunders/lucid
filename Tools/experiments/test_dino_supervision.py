import unittest

import torch

from dino_supervision import hierarchical_loss, reverse_reliability


class DinoSupervisionTests(unittest.TestCase):
    def test_unreliable_layer_receives_more_supervision(self):
        reference = [torch.tensor([[[1., 0.]]]), torch.tensor([[[1., 0.]]])]
        low = [reference[0].clone(), torch.tensor([[[0., 1.]]])]
        weights = reverse_reliability(low, reference)
        self.assertGreater(float(weights[0, 1]), float(weights[0, 0]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1))
        self.assertFalse(weights.requires_grad)

    def test_exact_features_have_zero_loss_and_only_prediction_receives_gradient(self):
        reference = [torch.tensor([[[1., 0.]]], requires_grad=True)]
        weights = torch.ones(1, 1, requires_grad=True)
        self.assertEqual(float(hierarchical_loss(reference, reference, weights).detach()), 0)
        prediction = [torch.tensor([[[0., 1.]]], requires_grad=True)]
        loss = hierarchical_loss(prediction, reference, weights)
        loss.backward()
        self.assertGreater(float(prediction[0].grad.abs().sum()), 0)
        self.assertIsNone(reference[0].grad)
        self.assertIsNone(weights.grad)


if __name__ == '__main__':
    unittest.main()
