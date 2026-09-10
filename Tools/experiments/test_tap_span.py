"""Behavioral CPU tests for the same-frame pre-TAA tap prototype.

Covers Tools/architectures/tap_span.py and
Tools/experiments/tap_frame_feeding.py only. Bounded CPU sizes,
no GPU, no native code, no network. No quality claim: geometry tests
characterize initialization and feeding behavior only.

Parent-owned Tools/experiments/test_tap_recipe.py covers arm
training/serialization; this file covers initialization preservation,
geometry, and checkpoint versioning.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from architectures.tap_span import TapSPAN, load_tap_checkpoint, TAP_MODES
from tap_frame_feeding import run_tap_sequence
from train_span import Unshuffled


class TapGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def _model(self, seed=0, channels=8, tap='full'):
        torch.manual_seed(seed)
        return TapSPAN(Unshuffled(channels, scale=2), tap=tap).eval()

    def test_modes_reject_unknown(self):
        with self.assertRaisesRegex(ValueError, 'none, full or luma'):
            self._model(tap='residual')
        with self.assertRaisesRegex(ValueError, 'raw, taa or search'):
            run_tap_sequence(self._model(), torch.rand(1, 3, 3, 32, 40), source_motion_policy='packed')

    def test_geometry_rejects_mismatch(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, 'matched even'):
            model(torch.rand(1, 3, 32, 40), torch.rand(1, 3, 30, 40))
        with self.assertRaisesRegex(ValueError, 'matched even'):
            model(torch.rand(1, 3, 32, 40), torch.rand(1, 3, 32, 40).repeat(1, 2, 1, 1)[:, :3])

    def test_every_mode_preserves_source_function(self):
        torch.manual_seed(3)
        single = Unshuffled(8, scale=2).eval()
        processed = torch.rand(2, 3, 32, 40)
        raw = torch.rand(2, 3, 32, 40)
        expected = single(processed)
        for tap in TAP_MODES:
            with self.subTest(tap=tap):
                torch.manual_seed(3)
                model = TapSPAN(Unshuffled(8, scale=2), tap=tap).eval()
                with torch.no_grad():
                    self.assertLess(float((model(processed, raw) - expected).abs().max()), 1e-5)

    def test_tap_changes_output_once_learned(self):
        model = self._model().train()
        with torch.no_grad():
            model.tap_conv.weight.fill_(.01)
        processed = torch.rand(1, 3, 32, 40)
        raw = torch.rand(1, 3, 32, 40)
        plain = self._model().eval()(processed, processed)
        tapped = model.eval()(processed, raw)
        self.assertGreater(float((tapped - plain).abs().max()), 1e-4)

    def test_luma_tap_ignores_chroma_only_difference(self):
        torch.manual_seed(5)
        model = TapSPAN(Unshuffled(8, scale=2), tap='luma').eval()
        with torch.no_grad():
            model.tap_conv.weight.fill_(.02)
        processed = torch.rand(1, 3, 32, 40)
        raw = processed.clone()
        raw[:, 1:] = torch.rand(1, 2, 32, 40)
        raw[:, 0] = (processed[:, 0] * .2126 + processed[:, 1] * .7152
                     + processed[:, 2] * .0722 - raw[:, 1] * .7152 - raw[:, 2] * .0722) / .2126
        self.assertLess(float((model(processed, raw) - model(processed, processed)).abs().max()), 1e-5)

    def test_raw_policy_passes_inputs_through(self):
        model = self._model()
        torch.manual_seed(7)
        decoded = torch.rand(1, 4, 3, 32, 40)
        with torch.no_grad():
            output, inputs = run_tap_sequence(model, decoded, source_motion_policy='raw', return_inputs=True)
        self.assertTrue(torch.equal(inputs, decoded))
        self.assertEqual(output.shape, (1, 4, 3, 64, 80))

    def test_taa_policy_smooths_inputs(self):
        model = self._model()
        torch.manual_seed(11)
        decoded = torch.rand(1, 3, 3, 32, 40)
        with torch.no_grad():
            output, inputs = run_tap_sequence(model, decoded, source_motion_policy='taa', return_inputs=True)
        self.assertFalse(torch.equal(inputs, decoded))
        self.assertEqual(output.shape, (1, 3, 3, 64, 80))

    def test_checkpoint_roundtrip_and_version_gates(self):
        torch.manual_seed(13)
        model = self._model(tap='full')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tap.pth'
            torch.save({'model': model.state_dict(), 'architecture': 'tap_span2x', 'scale': 2,
                        'state_representation': 'same_frame_tap_v1', 'feature_source': 'decoded_rgb8',
                        'tap_mode': 'full', 'source_motion_policy': 'search',
                        'channels': 8, 'version': 1, 'step': 1,
                        'experiment': {'args': {'tap_mode': 'full', 'source_motion_policy': 'search'}}}, path)
            loaded, checkpoint = load_tap_checkpoint(path)
            self.assertEqual(checkpoint['tap_mode'], 'full')
            processed = torch.rand(1, 3, 32, 40)
            with torch.no_grad():
                self.assertLess(float((loaded(processed, processed) - model(processed, processed)).abs().max()), 1e-6)
            bad = dict(torch.load(path, map_location='cpu', weights_only=False))
            bad['tap_mode'] = 'luma'
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, 'training declaration'):
                load_tap_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
