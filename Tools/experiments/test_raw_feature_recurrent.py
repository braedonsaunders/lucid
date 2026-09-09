"""Behavioral CPU tests for the raw-observation feature-memory prototype.

Covers Tools/architectures/raw_feature_span.py and
Tools/experiments/raw_feature_frame_feeding.py only. Bounded CPU sizes,
no GPU, no native code, no network.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from architectures.raw_feature_span import RawFeatureSPAN, load_raw_feature_checkpoint
from raw_feature_frame_feeding import run_raw_feature_sequence
from train_span import Unshuffled


class RawFeatureRecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _model(self, seed=0, channels=4):
        torch.manual_seed(seed)
        return RawFeatureSPAN(Unshuffled(channels, scale=2)).eval()

    def _learned(self, model, value=.02):
        with torch.no_grad():
            model.project.weight.fill_(value)
        return model

    def test_zero_projection_matches_backbone_with_distinct_raw_and_state(self):
        model = self._model()
        current = torch.rand(1, 3, 32, 32)
        raw = 1 - current
        self.assertFalse(torch.equal(raw, current))
        channels = model.project.in_channels
        state = torch.rand(1, channels, 16, 16)
        confidence = torch.ones(1, 1, 64, 64)
        output, new_state = model(current, raw, confidence, state)
        with torch.no_grad():
            expected = model.sr(current)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(new_state.shape, state.shape)
        self.assertGreater(float((new_state - state).abs().max()), 0)

    def test_encoder_is_independent_trainable_copy_with_frozen_backbone(self):
        torch.manual_seed(1)
        model = self._learned(RawFeatureSPAN(Unshuffled(4, scale=2))).train()
        self.assertFalse(model.sr.training)
        for parameter in model.sr.parameters():
            self.assertFalse(parameter.requires_grad)
        for module in (model.raw_encoder, model.decay, model.project):
            for parameter in module.parameters():
                self.assertTrue(parameter.requires_grad)
        current = torch.rand(1, 3, 32, 32)
        raw = torch.rand(1, 3, 32, 32)
        confidence = torch.ones(1, 1, 64, 64)
        state = torch.rand(1, model.project.in_channels, 16, 16)
        before = model(current, raw, confidence, state)[0].detach()
        with torch.no_grad():
            frozen = model.sr.core.conv_1.weight.clone()
            model.raw_encoder.weight.add_(1.)
        torch.testing.assert_close(model.sr.core.conv_1.weight, frozen, rtol=0, atol=0)
        output, _ = model(current, raw, confidence, state)
        self.assertGreater(float((output - before).abs().max()), 0)
        with torch.no_grad():
            backbone = model.sr(current)
        torch.testing.assert_close(model.sr(current), backbone, rtol=0, atol=0)
        model.zero_grad()
        output.sum().backward()
        for module in (model.raw_encoder, model.decay, model.project):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))

    def test_zero_confidence_removes_past_but_retains_fresh_raw(self):
        model = self._learned(self._model())
        current = torch.rand(1, 3, 32, 32)
        raw = torch.rand(1, 3, 32, 32)
        channels = model.project.in_channels
        state = torch.rand(1, channels, 16, 16)
        zeros = torch.zeros(1, 1, 64, 64)
        with_history, _ = model(current, raw, zeros, state)
        current_only, new_state = model(current, raw, zeros, None)
        torch.testing.assert_close(with_history, current_only, rtol=0, atol=0)
        with torch.no_grad():
            observed = model.raw_encoder(model.sr.unshuffle(raw))
            backbone = model.sr(current)
        torch.testing.assert_close(new_state, observed, rtol=0, atol=0)
        self.assertGreater(float((with_history - backbone).abs().max()), 0)

    def test_first_frame_matches_same_model_current_only_not_backbone(self):
        model = self._learned(self._model())
        frame = torch.rand(1, 3, 16, 24)
        static = torch.stack((frame, frame), 1)
        history = run_raw_feature_sequence(model, static, use_history=True)
        control = run_raw_feature_sequence(model, static, use_history=False)
        torch.testing.assert_close(history[:, 0], control[:, 0], rtol=0, atol=0)
        with torch.no_grad():
            backbone = model.sr(static[:, 0]).clamp(0, 1)
        self.assertGreater(float((history[:, 0] - backbone).abs().max()), 0)
        self.assertGreater(float((history[:, 1] - control[:, 1]).abs().max()), 0)
        self.assertTrue(((history >= 0) & (history <= 1)).all())

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
            raw_out, raw_inputs = run_raw_feature_sequence(
                model, source, source_motion_policy='raw', return_inputs=True)
            torch.testing.assert_close(raw_inputs, source, rtol=0, atol=0)
            for call in seen:
                torch.testing.assert_close(call[0], call[1], rtol=0, atol=0)
            seen.clear()
            search_out, search_inputs = run_raw_feature_sequence(
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
            run_raw_feature_sequence(model, source, source_motion_policy='unknown')

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
            output = run_raw_feature_sequence(model, source, use_history=True)
        finally:
            hook.remove()
        self.assertIsNone(seen[0])
        carried = seen[1]
        self.assertGreater(float(carried[0].abs().sum()), 0)
        torch.testing.assert_close(carried[1], torch.zeros_like(carried[1]),
                                   rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_motion_grid_translates_state_and_preserves_gradients(self):
        from recurrent_frame_feeding import warp_previous
        torch.manual_seed(54)
        previous = torch.rand(1, 3, 48, 64)
        current = torch.roll(previous, 4, -1)
        rgb = previous.repeat_interleave(2, -2).repeat_interleave(2, -1)
        _, _, cut, grid = warp_previous(current, previous, rgb, return_state_grid=True)
        self.assertFalse(cut.any())
        state = torch.rand(1, 8, 24, 32, requires_grad=True)
        aligned = F.grid_sample(state, grid, padding_mode='zeros', align_corners=False)
        expected = torch.roll(state, 2, -1)
        torch.testing.assert_close(aligned[..., 8:-8, 8:-8], expected[..., 8:-8, 8:-8],
                                   atol=4e-6, rtol=0)
        aligned.sum().backward()
        self.assertGreater(float(state.grad.abs().sum()), 0)
        self.assertFalse(grid.requires_grad)

    def test_carried_state_gradients_reach_state_tensor(self):
        torch.manual_seed(7)
        model = self._learned(RawFeatureSPAN(Unshuffled(4, scale=2))).train()
        state = torch.rand(1, model.project.in_channels, 16, 16, requires_grad=True)
        output, _ = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32),
                          torch.ones(1, 1, 64, 64), state)
        output.sum().backward()
        self.assertGreater(float(state.grad.abs().sum()), 0)

    def test_checkpoint_roundtrip_binds_feature_source_and_motion_policy(self):
        from eval_checkpoint import load
        torch.manual_seed(8)
        model = self._model(seed=8)
        with torch.no_grad():
            model.project.weight.normal_(std=.01)
        source = torch.rand(1, 2, 3, 16, 24)
        expected = run_raw_feature_sequence(model, source)
        state = dict(architecture='raw_feature_span2x', scale=2, channels=4, version=1,
                     state_representation='aligned_raw_features_v1',
                     feature_source='decoded_rgb8', source_motion_policy='raw',
                     experiment={'args': {'source_motion_policy': 'raw'}},
                     model=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save(state, path)
            restored, _ = load_raw_feature_checkpoint(path)
            self.assertEqual(restored.history_source, 'raw_features')
            self.assertEqual(restored.motion_seed, 'search')
            torch.testing.assert_close(run_raw_feature_sequence(restored, source),
                                       expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'explicit stateful loader'):
                load(path, 'cpu')
            state['architecture'] = 'recurrent_span2x'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'versioned decoded-feature'):
                load_raw_feature_checkpoint(path)
            state['architecture'] = 'raw_feature_span2x'
            state['source_motion_policy'] = 'search'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'source policy'):
                load_raw_feature_checkpoint(path)
            state['source_motion_policy'] = 'raw'
            del state['state_representation']
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'versioned decoded-feature'):
                load_raw_feature_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
