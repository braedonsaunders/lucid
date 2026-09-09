"""Behavioral CPU tests for the full-LR observation feature-memory prototype.

Covers Tools/architectures/full_lr_feature_span.py and
Tools/experiments/full_lr_frame_feeding.py only. Bounded CPU sizes,
no GPU, no native code, no network. No quality claim: geometry tests
characterize alignment behavior only.

Parent-owned Tools/experiments/test_full_lr_recipe.py covers sequence
gradients and arm training/serialization; this file covers geometry,
reset/provenance, and checkpoint versioning.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from architectures.full_lr_feature_span import FullLRFeatureSPAN, load_full_lr_checkpoint
from full_lr_frame_feeding import (
    align_full_lr_state,
    full_lr_correspondence,
    run_full_lr_sequence,
)
from train_span import Unshuffled


def _smooth_content(seed=0, height=48, width=64):
    """Deterministic broadband content that block matching can track exactly."""
    generator = torch.Generator().manual_seed(seed)
    content = torch.rand(1, 3, height, width, generator=generator)
    kernel = torch.ones(1, 1, 5, 5) / 25
    content = F.conv2d(content.reshape(-1, 1, height, width), kernel, padding=2)
    content = content.reshape(1, 3, height, width)
    return (content - content.min()) / (content.max() - content.min())


class FullLRFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def _model(self, seed=0, channels=4, state_channels=8):
        torch.manual_seed(seed)
        return FullLRFeatureSPAN(Unshuffled(channels, scale=2),
                                 state_channels=state_channels).eval()

    def _learned(self, model, value=.02):
        with torch.no_grad():
            model.project.weight.fill_(value)
        return model

    def test_full_lr_state_shape_and_zero_projection_match_backbone(self):
        model = self._model()
        self.assertEqual(model.state_channels, 8)
        self.assertEqual(model.raw_encoder.out_channels, 8)
        # Eight LR channels pixel-unshuffled match 32 half-LR channels.
        self.assertEqual(model.project.in_channels, 8 * 4)
        self.assertEqual(float(model.project.weight.abs().sum()), 0)
        current = torch.rand(1, 3, 32, 32)
        raw = 1 - current
        self.assertFalse(torch.equal(raw, current))
        state = torch.rand(1, 8, 32, 32)
        output, new_state = model(current, raw, torch.ones(1, 1, 32, 32), state)
        with torch.no_grad():
            expected = model.sr(current)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(new_state.shape, state.shape)
        self.assertGreater(float((new_state - state).abs().max()), 0)

    def test_first_frame_matches_same_model_current_only_not_backbone(self):
        model = self._learned(self._model())
        frame = torch.rand(1, 3, 16, 24)
        static = torch.stack((frame, frame), 1)
        history = run_full_lr_sequence(model, static, use_history=True)
        control = run_full_lr_sequence(model, static, use_history=False)
        torch.testing.assert_close(history[:, 0], control[:, 0], rtol=0, atol=0)
        with torch.no_grad():
            backbone = model.sr(static[:, 0]).clamp(0, 1)
        self.assertGreater(float((history[:, 0] - backbone).abs().max()), 0)
        self.assertGreater(float((history[:, 1] - control[:, 1]).abs().max()), 0)
        self.assertTrue(torch.isfinite(history).all())
        self.assertTrue(((history >= 0) & (history <= 1)).all())

    def test_zero_confidence_forgets_history_retains_fresh_encoded_input(self):
        model = self._learned(self._model())
        current = torch.rand(1, 3, 32, 32)
        raw = torch.rand(1, 3, 32, 32)
        state = torch.rand(1, 8, 32, 32)
        zeros = torch.zeros(1, 1, 32, 32)
        with_history, _ = model(current, raw, zeros, state)
        current_only, new_state = model(current, raw, zeros, None)
        torch.testing.assert_close(with_history, current_only, rtol=0, atol=0)
        with torch.no_grad():
            observed = model.raw_encoder(raw)
            backbone = model.sr(current)
        torch.testing.assert_close(new_state, observed, rtol=0, atol=0)
        self.assertGreater(float((with_history - backbone).abs().max()), 0)

    def test_per_sample_cut_resets_only_cut_sample(self):
        model = self._learned(self._model(seed=5))
        torch.manual_seed(6)
        frame = torch.rand(1, 3, 16, 24)
        static = torch.stack((frame, frame), 1)
        cut = torch.stack((torch.zeros(1, 3, 16, 24), torch.ones(1, 3, 16, 24)), 1)
        source = torch.cat((static, cut), 0)
        seen = []
        hook = model.register_forward_pre_hook(
            lambda _, args: seen.append(args[3].detach().clone()
                                        if args[3] is not None else None))
        try:
            output = run_full_lr_sequence(model, source, use_history=True)
        finally:
            hook.remove()
        self.assertIsNone(seen[0])
        carried = seen[1]
        self.assertGreater(float(carried[0].abs().sum()), 0)
        torch.testing.assert_close(carried[1], torch.zeros_like(carried[1]),
                                   rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_driver_separates_raw_and_preprocessed_sources(self):
        from native_stages import preprocess_sequence
        model = self._model(seed=3)
        torch.manual_seed(4)
        source = torch.rand(1, 2, 3, 16, 24)
        seen = []
        hook = model.register_forward_pre_hook(
            lambda _, args: seen.append(tuple(
                a.detach().clone() if torch.is_tensor(a) else a for a in args)))
        try:
            raw_out, raw_inputs = run_full_lr_sequence(
                model, source, source_motion_policy='raw', return_inputs=True)
            torch.testing.assert_close(raw_inputs, source, rtol=0, atol=0)
            for call in seen:
                torch.testing.assert_close(call[0], call[1], rtol=0, atol=0)
            seen.clear()
            search_out, search_inputs = run_full_lr_sequence(
                model, source, source_motion_policy='search', return_inputs=True)
            torch.testing.assert_close(
                search_inputs, preprocess_sequence(source, motion_policy='search'),
                rtol=0, atol=0)
            self.assertGreater(float((search_inputs - source).abs().max()), 0)
            for t, call in enumerate(seen):
                torch.testing.assert_close(call[0], search_inputs[:, t], rtol=0, atol=0)
                torch.testing.assert_close(call[1], source[:, t], rtol=0, atol=0)
        finally:
            hook.remove()
        self.assertGreater(float((search_out - raw_out).abs().max()), 0)
        with self.assertRaises(ValueError):
            run_full_lr_sequence(model, source, source_motion_policy='unknown')
        with self.assertRaises(ValueError):
            run_full_lr_sequence(model, source, state_warp='half_lr')
        with self.assertRaises(ValueError):
            align_full_lr_state(torch.rand(1, 8, 16, 16),
                                torch.rand(1, 16, 16, 2), state_warp='half_lr')

    def test_motion_and_confidence_come_from_raw_decoded_frames(self):
        model = self._model(seed=7)
        torch.manual_seed(8)
        source = torch.rand(1, 2, 3, 16, 24)
        seen = []
        hook = model.register_forward_pre_hook(
            lambda _, args: seen.append(tuple(
                a.detach().clone() if torch.is_tensor(a) else a for a in args)))
        try:
            run_full_lr_sequence(model, source, source_motion_policy='search')
        finally:
            hook.remove()
        # Model input is preprocessed, raw tap and confidence are decoded.
        self.assertFalse(torch.equal(seen[1][0], source[:, 1]))
        torch.testing.assert_close(seen[1][1], source[:, 1], rtol=0, atol=0)
        expected, _, _ = full_lr_correspondence(source[:, 1], source[:, 0])
        torch.testing.assert_close(seen[1][2], expected, rtol=0, atol=0)

    def test_correspondence_matches_legacy_warp_on_bounded_frames(self):
        from recurrent_frame_feeding import warp_previous
        previous = _smooth_content(seed=1)
        current = torch.roll(previous, 2, -1)
        confidence, cut, grid = full_lr_correspondence(current, previous)
        self.assertFalse(cut.any())
        static_confidence, static_cut, _ = full_lr_correspondence(previous, previous)
        self.assertFalse(static_cut.any())
        self.assertTrue((static_confidence == 1).all())
        previous_hr = previous.repeat_interleave(2, -2).repeat_interleave(2, -1)
        _, valid_hr, legacy_cut, state_grid = warp_previous(
            current, previous, previous_hr, motion_seed='search', return_state_grid=True)
        torch.testing.assert_close(cut, legacy_cut, rtol=0, atol=0)
        torch.testing.assert_close(F.avg_pool2d(valid_hr, 2), confidence, rtol=0, atol=0)
        pooled = F.avg_pool2d(grid.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        torch.testing.assert_close(pooled, state_grid, rtol=0, atol=1e-5)

    def test_translation_commutes_with_full_lr_but_not_packed_ablation(self):
        torch.manual_seed(9)
        encoder = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False).eval()
        base = _smooth_content(seed=2)
        previous_features = encoder(base)
        for shift in (1, 2):
            current = torch.roll(base, shift, -1)
            confidence, cut, grid = full_lr_correspondence(current, base)
            self.assertFalse(cut.any())
            interior = (slice(None), slice(None), slice(10, -10),
                        slice(10 + shift, -10))
            expected = encoder(current)[interior]
            aligned = align_full_lr_state(previous_features, grid,
                                          state_warp='full_lr')[interior]
            torch.testing.assert_close(aligned, expected, rtol=0, atol=1e-5)
            ablated = align_full_lr_state(previous_features, grid,
                                          state_warp='packed_half_lr')[interior]
            if shift % 2:
                self.assertGreater(float((ablated - expected).abs().max()), 1e-3)
                self.assertGreater(float((ablated - aligned).abs().max()), 1e-3)
            else:
                torch.testing.assert_close(ablated, expected, rtol=0, atol=1e-5)

    def test_checkpoint_roundtrip_binds_warp_policy_and_representation(self):
        from eval_checkpoint import load
        torch.manual_seed(8)
        model = self._model(seed=8)
        with torch.no_grad():
            model.project.weight.normal_(std=.01)
        source = torch.rand(1, 2, 3, 16, 24)
        expected = run_full_lr_sequence(model, source)
        state = dict(architecture='full_lr_feature_span2x', scale=2, channels=4, version=1,
                     state_representation='full_lr_observation_features_v1',
                     feature_source='decoded_rgb8', source_motion_policy='raw',
                     state_warp='full_lr', state_channels=8,
                     experiment={'args': {'source_motion_policy': 'raw',
                                          'state_warp': 'full_lr', 'state_channels': 8}},
                     model=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(state, path)
            restored, _ = load_full_lr_checkpoint(path)
            self.assertEqual(restored.history_source, 'full_lr_features')
            self.assertEqual(restored.motion_seed, 'search')
            torch.testing.assert_close(run_full_lr_sequence(restored, source),
                                       expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'explicit stateful loader'):
                load(path, 'cpu')
            state['architecture'] = 'raw_feature_span2x'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'versioned full-LR decoded-feature'):
                load_full_lr_checkpoint(path)
            state['architecture'] = 'full_lr_feature_span2x'
            state['source_motion_policy'] = 'search'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'source policy'):
                load_full_lr_checkpoint(path)
            state['source_motion_policy'] = 'raw'
            state['state_warp'] = 'packed_half_lr'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'feature alignment'):
                load_full_lr_checkpoint(path)
            state['state_warp'] = 'full_lr'
            state['state_channels'] = 4
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'feature state channels'):
                load_full_lr_checkpoint(path)
            state['state_channels'] = 8
            del state['state_representation']
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'versioned full-LR decoded-feature'):
                load_full_lr_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
