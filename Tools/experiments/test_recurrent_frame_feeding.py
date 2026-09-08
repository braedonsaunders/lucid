import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch

from architectures.recurrent_span import RecurrentSPAN
from recurrent_frame_feeding import recurrent_step, warp_previous, subpixel_motion, load_recurrent_checkpoint
from train_recurrent_span import run_sequence
from eval_checkpoint import load
from train_span import Unshuffled
from native_stages import motion_blocks, preprocess_sequence


class RecurrentFrameTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(40)

    def test_identity_initialization_and_history_gradients(self):
        source = torch.rand(1, 3, 16, 24)
        base = Unshuffled(4, scale=2).eval()
        model = RecurrentSPAN(copy.deepcopy(base)).train()
        history = torch.rand(1, 3, 32, 48)
        output = model(source, history, torch.ones_like(history[:, :1]))
        torch.testing.assert_close(output, base(source), rtol=0, atol=0)
        output.mean().backward()
        self.assertGreater(float(model.history.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))

    def test_static_and_translated_correspondence_uses_decoded_inputs(self):
        previous = torch.rand(1, 3, 40, 48)
        previous_hr = previous.repeat_interleave(2, -2).repeat_interleave(2, -1)
        aligned, confidence, cut = warp_previous(previous, previous, previous_hr)
        torch.testing.assert_close(aligned, previous_hr, atol=4e-6, rtol=0)
        self.assertTrue((confidence == 1).all())
        self.assertFalse(cut.any())
        current = torch.roll(previous, 4, -1)
        aligned, confidence, cut = warp_previous(current, previous, previous_hr)
        target = current.repeat_interleave(2, -2).repeat_interleave(2, -1)
        selected = confidence[..., 24:-24, 24:-24] > .99
        self.assertTrue(selected.any())
        error = (aligned - target).abs()[..., 24:-24, 24:-24]
        self.assertLess(float(error.masked_select(selected.expand_as(error)).max()), 5e-6)

    def test_scene_cut_rejects_old_pixels(self):
        dark, light = torch.zeros(1, 3, 16, 24), torch.ones(1, 3, 16, 24)
        _, confidence, cut = warp_previous(light, dark, dark.repeat_interleave(2, -2).repeat_interleave(2, -1))
        self.assertTrue(cut.all())
        self.assertEqual(float(confidence.sum()), 0)

    def test_half_pixel_motion_recovers_fractional_samples(self):
        previous = torch.rand(1, 1, 40, 48)
        current = (previous + torch.roll(previous, -1, -1)) / 2
        # Ignore wrapped boundary blocks; interior is an exact half-pixel shift.
        field = subpixel_motion(current, previous)[..., 1:-1, 1:-1]
        self.assertTrue((field[:, 0] == .5).all())
        self.assertTrue((field[:, 1] == 0).all())
        self.assertLess(float(field[:, 3].max()), 2e-6)

    def test_low_contrast_translation_keeps_integer_seed_without_changing_native_taa(self):
        previous = ((.1 + .03 * torch.rand(1, 1, 48, 56)) * 255).round() / 255
        current = torch.roll(previous, 4, -1)
        native = motion_blocks(current, previous)[..., 1:-1, 1:-1]
        legacy = subpixel_motion(current, previous, motion_seed='taa')[..., 1:-1, 1:-1]
        corrected = subpixel_motion(current, previous)[..., 1:-1, 1:-1]
        # Native TAA deliberately calls this low-error motion stationary.
        self.assertTrue((native[:, :2] == 0).all())
        self.assertTrue((legacy[:, 0].abs() <= .5).all())
        self.assertTrue((corrected[:, 0] == -4).all())
        self.assertTrue((corrected[:, 1] == 0).all())
        self.assertLess(float(corrected[:, 3].max()), 2e-6)

    def test_explicit_reset_seek_stream_and_resize_use_current_frame_only(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2)).eval()
        with torch.no_grad():
            model.history.weight.fill_(.01)
        source = torch.rand(1, 3, 16, 24)
        _, state, _ = recurrent_step(model, source, source, None, stream='a', index=10)
        cases = [{'stream': 'a', 'index': 11, 'reset': True},
                 {'stream': 'a', 'index': 20}, {'stream': 'b', 'index': 11},
                 {'stream': 'a', 'index': 11, 'use_history': False}]
        for kwargs in cases:
            output, _, reset = recurrent_step(model, source, source, state, **kwargs)
            torch.testing.assert_close(output, model.sr(source).clamp(0, 1), atol=0, rtol=0)
            self.assertTrue(reset.all())
        resized = torch.rand(1, 3, 24, 32)
        output, new_state, reset = recurrent_step(model, resized, resized, state, stream='a', index=11)
        torch.testing.assert_close(output, model.sr(resized).clamp(0, 1), atol=0, rtol=0)
        self.assertEqual(new_state.output.shape, (1, 3, 48, 64))
        self.assertTrue(reset.all())

    def test_stored_state_is_quantized_pre_detail_output_and_detachable(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2)).eval()
        source = torch.rand(1, 3, 16, 24)
        output, state, _ = recurrent_step(model, source, source, None, stream='a', index=0, detach_history=True)
        torch.testing.assert_close(state.output, (output.detach() * 255).round() / 255, rtol=0, atol=0)
        self.assertFalse(state.output.requires_grad)
        self.assertFalse(state.source.requires_grad)
        self.assertTrue(((state.output >= 0) & (state.output <= 1)).all())

    def test_sequence_backprop_and_stateful_checkpoint_round_trip(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), motion_seed='taa').train()
        source = torch.rand(1, 1, 3, 16, 24).repeat(1, 3, 1, 1, 1)
        frozen = {k: v.clone() for k, v in model.sr.state_dict().items()}
        output, _ = run_sequence(model, source)
        output[:, -1].square().mean().backward()
        self.assertGreater(float(model.history.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
        for k, v in model.sr.state_dict().items():
            torch.testing.assert_close(v, frozen[k], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save({'architecture': 'recurrent_span2x', 'scale': 2, 'channels': 4,
                        'version': 1, 'model': model.state_dict()}, path)
            restored, _ = load_recurrent_checkpoint(path, 'cpu')
            self.assertEqual(restored.history_source, 'sr')
            self.assertEqual(restored.motion_seed, 'taa')
            actual, _ = run_sequence(restored, source)
            torch.testing.assert_close(actual, output, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                load(path, 'cpu')

    def test_joint_backbone_and_history_receive_finite_gradients(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), train_backbone=True).train()
        source = torch.rand(1, 1, 3, 16, 24).repeat(1, 3, 1, 1, 1)
        before = model.sr.core.conv_1.weight.detach().clone()
        output, _ = run_sequence(model, source)
        loss = (output[:, -1] - .4).square().mean()
        loss.backward()
        for parameter in (model.sr.core.conv_1.weight, model.history.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        torch.optim.SGD(model.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(before, model.sr.core.conv_1.weight))
        self.assertGreater(float(model.history.weight.detach().abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'joint.pth'
            torch.save({'architecture': 'recurrent_span2x', 'scale': 2, 'channels': 4,
                        'version': 1, 'model': model.state_dict(), 'motion_seed': model.motion_seed}, path)
            restored, _ = load_recurrent_checkpoint(path, 'cpu')
            actual, _ = run_sequence(restored, source)
            expected, _ = run_sequence(model, source)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_joint_no_history_control_is_independent_of_prior_frames(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), train_backbone=True).train()
        with torch.no_grad():
            model.history.weight.fill_(.02)
        source = torch.rand(1, 3, 3, 16, 24)
        actual, _ = run_sequence(model, source, use_history=False)
        changed = source.clone()
        changed[:, :2] = 1 - changed[:, :2]
        other, _ = run_sequence(model, changed, use_history=False)
        torch.testing.assert_close(actual[:, -1], other[:, -1], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, -1], model.sr(source[:, -1]).clamp(0, 1), rtol=0, atol=0)

    def test_source_policy_preserves_legacy_and_raw_but_selects_full_search(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), history_source='decoded').eval()
        previous = ((.1 + .03 * torch.rand(1, 3, 48, 56)) * 255).round() / 255
        source = torch.stack((previous, torch.roll(previous, 4, -1)), 1)
        with torch.no_grad():
            legacy, legacy_inputs = run_sequence(model, source, True)
            explicit, _ = run_sequence(model, source, True, source_motion_policy='taa')
            actual, inputs = run_sequence(model, source, True, source_motion_policy='search')
            raw, raw_inputs = run_sequence(model, source, False, source_motion_policy='search')
        torch.testing.assert_close(legacy, explicit, rtol=0, atol=0)
        torch.testing.assert_close(inputs, preprocess_sequence(source, motion_policy='search'), rtol=0, atol=0)
        torch.testing.assert_close(legacy_inputs[:, 0], inputs[:, 0], rtol=0, atol=0)
        self.assertFalse(torch.equal(legacy_inputs[:, -1], inputs[:, -1]))
        torch.testing.assert_close(raw_inputs, source, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 0], legacy[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(raw[:, -1], model.sr(source[:, -1]).clamp(0, 1), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            run_sequence(model, source, source_motion_policy='unknown')

    def test_decoded_history_uses_observations_not_generated_or_preprocessed_pixels(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), train_backbone=True,
                              history_source='decoded').train()
        raw = torch.rand(1, 3, 16, 24) * .5 + .2
        processed = raw * .8
        _, state, _ = recurrent_step(model, processed, raw, None, stream='a', index=0)
        bogus_output = torch.rand_like(state.output).requires_grad_()
        state = replace(state, output=bogus_output)
        seen = []
        handle = model.register_forward_pre_hook(lambda _, args: seen.append(args))
        output, _, _ = recurrent_step(model, processed, raw, state, stream='a', index=1)
        handle.remove()
        expected = torch.nn.functional.interpolate(raw, scale_factor=2, mode='bicubic',
                                                   align_corners=False).clamp(0, 1)
        torch.testing.assert_close(seen[0][1], expected, rtol=0, atol=4e-6)
        self.assertTrue((seen[0][2] == 1).all())
        output.square().mean().backward()
        self.assertIsNone(bogus_output.grad)
        for parameter in (model.history.weight, model.sr.core.conv_1.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_decoded_history_resets_and_disabled_control_ignore_observations(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), history_source='decoded').eval()
        with torch.no_grad():
            model.history.weight.fill_(.01)
        source = torch.rand(1, 3, 16, 24)
        _, state, _ = recurrent_step(model, source, source, None, stream='a', index=0)
        for options in ({'reset': True}, {'use_history': False}, {'index': 3}):
            kwargs = {'stream': 'a', 'index': 1}
            kwargs.update(options)
            output, _, reset = recurrent_step(model, source, source, state, **kwargs)
            torch.testing.assert_close(output, model.sr(source).clamp(0, 1), rtol=0, atol=0)
            self.assertTrue(reset.all())

    def test_decoded_checkpoint_retains_history_semantics_and_rejects_conflicting_metadata(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), history_source='decoded').eval()
        with torch.no_grad():
            model.history.weight.normal_(std=.001)
        source = torch.rand(1, 1, 3, 16, 24).repeat(1, 3, 1, 1, 1)
        expected, _ = run_sequence(model, source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'decoded.pth'
            state = {'architecture': 'recurrent_span2x', 'scale': 2, 'channels': 4,
                     'version': 1, 'model': model.state_dict(), 'history_source': 'decoded',
                     'motion_seed': 'search',
                     'experiment': {'args': {'history_source': 'decoded', 'motion_seed': 'search'}}}
            torch.save(state, path)
            restored, _ = load_recurrent_checkpoint(path, 'cpu')
            self.assertEqual(restored.history_source, 'decoded')
            self.assertEqual(restored.motion_seed, 'search')
            actual, _ = run_sequence(restored, source)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            del state['history_source']
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'differs from training declaration'):
                load_recurrent_checkpoint(path, 'cpu')
            state['history_source'] = 'decoded'
            state['motion_seed'] = 'taa'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'motion seed differs'):
                load_recurrent_checkpoint(path, 'cpu')
        with self.assertRaises(ValueError):
            RecurrentSPAN(Unshuffled(4, scale=2), history_source='reference')

    def test_unversioned_checkpoint_preserves_legacy_motion_on_moving_low_contrast_frames(self):
        model = RecurrentSPAN(Unshuffled(4, scale=2), history_source='decoded', motion_seed='taa').eval()
        with torch.no_grad():
            model.history.weight.normal_(std=.02)
        previous = ((.1 + .03 * torch.rand(1, 3, 48, 56)) * 255).round() / 255
        source = torch.stack((previous, torch.roll(previous, 4, -1)), 1)
        expected, _ = run_sequence(model, source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy-motion.pth'
            torch.save({'architecture': 'recurrent_span2x', 'scale': 2, 'channels': 4,
                        'version': 1, 'model': model.state_dict(), 'history_source': 'decoded'}, path)
            restored, _ = load_recurrent_checkpoint(path, 'cpu')
            self.assertEqual(restored.motion_seed, 'taa')
            actual, _ = run_sequence(restored, source)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            restored.motion_seed = 'search'
            changed, _ = run_sequence(restored, source)
            self.assertGreater(float((changed[:, -1] - actual[:, -1]).abs().max()), 1e-7)


if __name__ == '__main__':
    unittest.main()
