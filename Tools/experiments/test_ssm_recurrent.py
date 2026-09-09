import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from architectures.ssm_recurrent_span import SelectiveScanState, SSMRecurrentSPAN, load_ssm_checkpoint


class SelectiveScanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_feature_grid_translates_state_and_preserves_gradients(self):
        from recurrent_frame_feeding import warp_previous
        torch.manual_seed(54)
        previous = torch.rand(1, 3, 48, 64)
        current = torch.roll(previous, 4, -1)
        rgb = previous.repeat_interleave(2, -2).repeat_interleave(2, -1)
        _, _, _, grid = warp_previous(current, previous, rgb, return_state_grid=True)
        state = torch.rand(1, 8, 24, 32, requires_grad=True)
        aligned = F.grid_sample(state, grid, padding_mode='zeros', align_corners=False)
        expected = torch.roll(state, 2, -1)
        torch.testing.assert_close(aligned[..., 8:-8, 8:-8], expected[..., 8:-8, 8:-8], atol=4e-6, rtol=0)
        aligned.sum().backward()
        self.assertGreater(float(state.grad.abs().sum()), 0)
        self.assertFalse(grid.requires_grad)

    def test_sequence_warps_carried_features_and_disabled_scan_ignores_past(self):
        from train_span import Unshuffled
        from train_ssm_span import run_ssm_sequence
        torch.manual_seed(55)
        model = SSMRecurrentSPAN(Unshuffled(8, scale=2)).eval()
        with torch.no_grad():
            model.scan.encode.weight.normal_(std=.01)
        first = torch.rand(1, 3, 48, 64)
        source = torch.stack((first, torch.roll(first, 4, -1), torch.roll(first, 8, -1)), 1)
        states, grids = [], []
        from recurrent_frame_feeding import warp_previous
        def observed_warp(*args, **kwargs):
            result = warp_previous(*args, **kwargs)
            grids.append(result[3])
            return result
        hook = model.register_forward_hook(lambda _, args, output: states.append((args[3], output[1])))
        with patch('recurrent_frame_feeding.warp_previous', side_effect=observed_warp):
            output = run_ssm_sequence(model, source)
        hook.remove()
        self.assertGreater(float(states[1][1].detach().abs().sum()), 0)
        expected = F.grid_sample(states[1][1], grids[1], padding_mode='zeros', align_corners=False)
        torch.testing.assert_close(states[2][0], expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(states[2][0], states[1][1]))
        disabled = run_ssm_sequence(model, source, use_scan=False)
        direct = torch.stack([model.sr(source[:, t]).clamp(0, 1) for t in range(3)], 1)
        torch.testing.assert_close(disabled, direct, rtol=0, atol=0)
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_half_lr_pixel_motion_moves_state_by_quarter_trunk_pixel(self):
        from recurrent_frame_feeding import warp_previous
        torch.manual_seed(57)
        previous = torch.rand(1, 1, 48, 64).repeat(1, 3, 1, 1)
        current = (previous + torch.roll(previous, -1, -1)) / 2
        rgb = previous.repeat_interleave(2, -2).repeat_interleave(2, -1)
        _, _, _, grid = warp_previous(current, previous, rgb, return_state_grid=True)
        state = torch.arange(32).float()[None, None, None].expand(1, 8, 24, 32)
        aligned = F.grid_sample(state, grid, padding_mode='zeros', align_corners=False)
        torch.testing.assert_close(aligned[..., 8:-8, 8:-8], (state + .25)[..., 8:-8, 8:-8], atol=3e-6, rtol=0)

    def test_zero_initialization_matches_backbone(self):
        from train_span import Unshuffled
        torch.manual_seed(0)
        base = Unshuffled(8, scale=2)
        model = SSMRecurrentSPAN(base).eval()
        current = torch.rand(1, 3, 32, 32)
        aligned = torch.rand(1, 3, 64, 64)
        confidence = torch.ones(1, 1, 64, 64)
        # trunk is 16x16 after the 2x unshuffle; aligned is 4x the trunk
        output, _ = model(current, aligned, confidence, None)
        with torch.no_grad():
            expected = model.sr(current)
        torch.testing.assert_close(output, expected, rtol=0, atol=1e-5)

    def test_rgb_only_driver_rejects_feature_state_model(self):
        from train_span import Unshuffled
        from recurrent_frame_feeding import recurrent_step
        model = SSMRecurrentSPAN(Unshuffled(8, scale=2)).eval()
        current = torch.rand(1, 3, 32, 32)
        with self.assertRaisesRegex(ValueError, 'explicit feature-state sequence driver'):
            recurrent_step(model, current, current, None, stream='test', index=0)

    def test_cut_zeroes_history(self):
        scan = SelectiveScanState(8).eval()
        trunk = torch.rand(1, 8, 16, 16)
        aligned = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            scan.encode.weight.normal_(std=.01)
            _, first = scan(trunk, aligned, torch.ones(1, 1, 64, 64), torch.rand_like(trunk))
            self.assertGreater(float(first.abs().sum()), 0)
            _, second = scan(trunk, aligned, torch.zeros(1, 1, 64, 64), first)
            gain = torch.sigmoid(scan.gain(trunk))
            encoded_zero = scan.encode(F.pixel_unshuffle(aligned * 0, 4))
        # Zero confidence kills the carried state; what remains is the gain
        # applied to a zeroed history, independent of the previous state.
        self.assertTrue(torch.isfinite(second).all())
        torch.testing.assert_close(second, gain * encoded_zero, rtol=1e-4, atol=1e-5)

    def test_state_geometry_guard(self):
        scan = SelectiveScanState(8).eval()
        trunk = torch.rand(1, 8, 16, 16)
        aligned = torch.rand(1, 3, 64, 64)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            scan(trunk, aligned, torch.ones(1, 1, 64, 64), torch.zeros(1, 8, 4, 4))

    def test_checkpoint_roundtrip_binds_state_and_input_policy(self):
        from train_span import Unshuffled
        from train_ssm_span import run_ssm_sequence
        model = SSMRecurrentSPAN(Unshuffled(8, scale=2)).eval()
        with torch.no_grad():
            model.scan.encode.weight.normal_(std=.01)
        source = torch.rand(1, 1, 3, 32, 40).repeat(1, 3, 1, 1, 1)
        state = dict(architecture='ssm_recurrent_span2x', scale=2, channels=8, version=1,
                     state_representation='aligned_scan_features_v1', source_motion_policy='raw',
                     experiment={'args': {'source_motion_policy': 'raw'}}, model=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            torch.save(state, path)
            restored, _ = load_ssm_checkpoint(path)
            from eval_checkpoint import load
            with self.assertRaisesRegex(ValueError, 'stateful loader'):
                load(path, 'cpu')
            torch.testing.assert_close(run_ssm_sequence(restored, source), run_ssm_sequence(model, source), rtol=0, atol=0)
            state['source_motion_policy'] = 'search'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'source policy'):
                load_ssm_checkpoint(path)
            state['source_motion_policy'] = 'raw'
            del state['state_representation']
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'versioned'):
                load_ssm_checkpoint(path)

    def test_trace_uses_elementwise_gates_without_attention(self):
        scan = SelectiveScanState(8).eval()
        trunk = torch.rand(1, 8, 16, 16)
        aligned = torch.rand(1, 3, 64, 64)
        confidence = torch.rand(1, 1, 64, 64)
        zero_state = torch.zeros(1, 8, 16, 16)
        traced = torch.jit.trace(scan, (trunk, aligned, confidence, zero_state))
        graph = str(traced.graph)
        for op in ('aten::sigmoid', 'aten::mul', 'aten::add'):
            self.assertIn(op, graph)
        for op in ('aten::softmax', 'aten::matmul'):
            self.assertNotIn(op, graph)
        buf = io.BytesIO()
        torch.jit.save(traced, buf)


if __name__ == '__main__':
    unittest.main()
