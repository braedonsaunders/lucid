"""Focused regressions for the evaluation and training objectives.

Run with: .venv-convert/bin/python -m unittest discover -s Tools/frontier_eval -p 'test_*.py'
"""
import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_loss import sobel_loss
from evaluate_sequences import temporal_metrics, summarize


class ObjectiveTests(unittest.TestCase):
    def test_signed_edges_reject_inverted_structure(self):
        torch.manual_seed(6)
        image = torch.rand(1, 3, 16, 16)
        self.assertEqual(float(sobel_loss(image, image)), 0)
        self.assertLess(float(sobel_loss(image + 0.1, image)), 1e-7)
        inverted = (1 - image).requires_grad_()
        loss = sobel_loss(inverted, image)
        self.assertGreater(float(loss.detach()), 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(inverted.grad).all())
        self.assertGreater(float(inverted.grad.abs().sum()), 0)

    def test_exact_motion_has_zero_residual(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = a.copy()
        a[20:40, 20:40] = 128
        b[20:40, 21:41] = 128
        images = [Image.fromarray(a), Image.fromarray(b)]
        metrics = temporal_metrics(images, images)
        self.assertEqual(metrics['temporal_residual_l1'], 0)
        self.assertEqual(metrics['static_flicker'], 0)
        self.assertGreater(metrics['static_coverage'], 0.9)

    def test_static_noise_cannot_change_reference_mask(self):
        reference = [Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))] * 2
        outputs = [Image.fromarray(np.full((64, 64, 3), v, dtype=np.uint8)) for v in (125, 131)]
        metrics = temporal_metrics(outputs, reference)
        self.assertEqual(metrics['static_coverage'], 1)
        self.assertEqual(metrics['static_flicker'], 6)
        self.assertEqual(metrics['temporal_residual_l1'], 6)

    def test_source_balancing_does_not_overweight_more_conditions(self):
        rows = [{'variant': 'a', 'source_id': s, 'metrics': {'lpips': x}}
                for s, x in [('one', 0), ('one', 0), ('two', 1)]]
        self.assertEqual(summarize(rows)['a']['source_balanced']['lpips'], 0.5)


if __name__ == '__main__':
    unittest.main()
