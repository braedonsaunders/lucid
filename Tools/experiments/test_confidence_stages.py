import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.confidence_span import ConfidenceSPAN, gaussian_error_nll, confidence_from_log_variance
from confidence_stages import confidence_sequence, aligned_deband_confidence
from native_stages import preprocess_sequence
from native_output_stages import postprocess_rgb
from train_span import Unshuffled
from eval_checkpoint import load
from train_confidence_head import calibration_bins


class ConfidenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(75)

    def test_variance_head_preserves_mean_and_backbone(self):
        sr = Unshuffled(4, scale=2).eval()
        model = ConfidenceSPAN(copy.deepcopy(sr)).train()
        image = torch.rand(2, 3, 16, 24)
        prediction, variance = model.predict_with_uncertainty(image)
        torch.testing.assert_close(prediction, sr(image), rtol=0, atol=0)
        self.assertEqual(variance.shape, (2, 1, 32, 48))
        gaussian_error_nll(prediction, torch.zeros_like(prediction), variance).backward()
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
        self.assertGreater(float(model.variance_head.weight.grad.abs().sum()), 0)

    def test_nll_calibrates_squared_error_and_does_not_optimize_mean(self):
        prediction = torch.full((1, 3, 8, 8), .1, requires_grad=True)
        log_variance = torch.full((1, 1, 8, 8), math.log(.01), requires_grad=True)
        gaussian_error_nll(prediction, torch.zeros_like(prediction), log_variance).backward()
        self.assertIsNone(prediction.grad)
        self.assertLess(float(log_variance.grad.abs().max()), 1e-8)
        confidence = confidence_from_log_variance(torch.tensor([-12., -6., 0.]))
        self.assertTrue(((confidence > 0) & (confidence < 1)).all())
        self.assertTrue((confidence[:-1] > confidence[1:]).all())

    def test_fixed_policy_matches_existing_source_and_output_proxies(self):
        model = ConfidenceSPAN(Unshuffled(4, scale=2)).eval()
        frames = torch.rand(1, 3, 3, 16, 24) * .004 + .4
        actual, _, controls, _ = confidence_sequence(model, frames, 'fixed')
        sources = preprocess_sequence(frames)
        expected = torch.stack([postprocess_rgb(model.sr(sources[:, t]), frames[:, t]) for t in range(3)], 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((controls == 1).all())

    def test_history_unknown_and_cut_fall_back_to_fixed_strength(self):
        dark = torch.zeros(1, 1, 16, 24)
        light = torch.ones_like(dark)
        field = torch.zeros(1, 4, 2, 3)
        torch.testing.assert_close(aligned_deband_confidence(None, dark, None, None), light)
        torch.testing.assert_close(aligned_deband_confidence(dark, light, dark, field), light)
        field[:, 2] = 1
        torch.testing.assert_close(aligned_deband_confidence(dark, light, dark, field), light)
        torch.testing.assert_close(aligned_deband_confidence(dark, dark, dark, field), dark)

    def test_causal_controls_do_not_use_future_frames(self):
        model = ConfidenceSPAN(Unshuffled(4, scale=2)).eval()
        frames = torch.rand(1, 3, 3, 16, 24)
        first = confidence_sequence(model, frames, 'causal')
        changed = frames.clone()
        changed[:, 2] = 1 - changed[:, 2]
        second = confidence_sequence(model, changed, 'causal')
        for a, b in zip(first, second):
            torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
        self.assertTrue((first[2][:, 0] == 1).all())

    def test_checkpoint_restores_mean_and_uncertainty(self):
        model = ConfidenceSPAN(Unshuffled(4, scale=2)).eval()
        with torch.no_grad():
            model.variance_head.weight.fill_(.03)
        image = torch.rand(1, 3, 16, 24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'confidence.pth'
            torch.save({'model': model.state_dict(), 'architecture': 'confidence_span2x',
                        'scale': 2, 'frames': 1, 'channels': 4, 'version': 1}, path)
            restored, _, _ = load(path, 'cpu')
            for a, b in zip(model.predict_with_uncertainty(image), restored.predict_with_uncertainty(image)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_autocast_preserves_frozen_weights_and_calibration_counts(self):
        model = ConfidenceSPAN(Unshuffled(4, scale=2)).train()
        state = {k: v.clone() for k, v in model.sr.state_dict().items()}
        source = torch.rand(1, 3, 16, 24)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output, variance = model.predict_with_uncertainty(source)
            torch.testing.assert_close(output, model.sr(source), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(variance).all())
        for k, v in model.sr.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)
        prediction = torch.full((1, 2, 3, 8, 8), .1)
        bins = calibration_bins(prediction, torch.zeros_like(prediction),
                                torch.full((1, 2, 1, 8, 8), math.log(.01)))
        self.assertEqual(sum(row['pixels'] for row in bins), 128)
        self.assertAlmostEqual(sum(row['variance_sum'] for row in bins),
                               sum(row['squared_error_sum'] for row in bins), places=5)


if __name__ == '__main__':
    unittest.main()
