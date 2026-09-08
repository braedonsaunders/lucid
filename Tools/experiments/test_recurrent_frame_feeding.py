import copy
from pathlib import Path
import tempfile
import unittest
import torch

from architectures.recurrent_span import RecurrentSPAN
from recurrent_frame_feeding import recurrent_step, warp_previous, subpixel_motion, load_recurrent_checkpoint
from train_recurrent_span import run_sequence
from eval_checkpoint import load
from train_span import Unshuffled


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
        model = RecurrentSPAN(Unshuffled(4, scale=2)).train()
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
            actual, _ = run_sequence(restored, source)
            torch.testing.assert_close(actual, output, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                load(path, 'cpu')


if __name__ == '__main__':
    unittest.main()
