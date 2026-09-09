"""CPU regressions for the SSM span training recipe (train_ssm_span).

Covers trainer wiring that the feature tests do not: source-policy
preprocessing vs raw motion inputs, validation-bank selection, per-group
gradient clipping, and per-sample cut resets.
"""
import unittest
from unittest.mock import call, patch

import torch


def _identity_state_grid(batch, height, width):
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    grid = torch.stack(((xs + .5) * 2 / width - 1, (ys + .5) * 2 / height - 1), -1)
    return grid[None].expand(batch, height, width, 2).contiguous()


def _warp_canned(batch, height=8, width=8, cut=None):
    aligned = torch.zeros(batch, 3, height * 2, width * 2)
    confidence = torch.ones(batch, 1, height * 2, width * 2)
    if cut is None:
        cut = torch.zeros(batch, dtype=torch.bool)
    return aligned, confidence, cut, _identity_state_grid(batch, height // 2, width // 2)


class FakeSequenceModel:
    """Callable stub: records SR inputs, carries a constant feature state."""

    def __init__(self, state_fill=3.0):
        self.currents = []
        self.states = []
        self.state_fill = state_fill

    def __call__(self, current, aligned, confidence, state):
        self.currents.append(current.detach().clone())
        self.states.append(None if state is None else state.detach().clone())
        output = torch.zeros(current.shape[0], 3,
                             current.shape[-2] * 2, current.shape[-1] * 2)
        return output, torch.full((current.shape[0], 8,
                                   current.shape[-2] // 2, current.shape[-1] // 2),
                                  self.state_fill)


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_preprocessed_sr_input_but_raw_warp_frames(self):
        from train_ssm_span import run_ssm_sequence
        decoded = torch.rand(1, 2, 3, 8, 8)
        altered = decoded + 10.0  # visibly altered preprocessing stand-in
        warp_calls = []
        canned = _warp_canned(1)

        def observed_warp(*args, **kwargs):
            warp_calls.append((args, kwargs))
            return canned

        model = FakeSequenceModel()
        with patch('train_ssm_span.preprocess_sequence', return_value=altered) as pre, \
                patch('recurrent_frame_feeding.warp_previous', side_effect=observed_warp):
            result, inputs = run_ssm_sequence(model, decoded, source_motion_policy='taa',
                                              return_inputs=True)
        pre.assert_called_once()
        self.assertEqual(pre.call_args[1].get('motion_policy'), 'taa')
        torch.testing.assert_close(inputs, altered, rtol=0, atol=0)
        # SR sees the preprocessed frames at every step.
        self.assertEqual(len(model.currents), 2)
        for t in range(2):
            torch.testing.assert_close(model.currents[t], altered[:, t], rtol=0, atol=0)
        # Motion still sees the raw decoded frames, not the preprocessed ones.
        self.assertEqual(len(warp_calls), 1)
        torch.testing.assert_close(warp_calls[0][0][0], decoded[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(warp_calls[0][0][1], decoded[:, 0], rtol=0, atol=0)
        self.assertGreater(float((warp_calls[0][0][0] - altered[:, 1]).abs().max()), 5)
        self.assertEqual(result.shape, (1, 2, 3, 16, 16))

    def test_mixed_batch_cut_resets_only_cut_sample(self):
        from train_ssm_span import run_ssm_sequence
        decoded = torch.rand(2, 2, 3, 8, 8)
        canned = _warp_canned(2, cut=torch.tensor([True, False]))
        model = FakeSequenceModel()
        with patch('recurrent_frame_feeding.warp_previous', return_value=canned):
            run_ssm_sequence(model, decoded)
        self.assertIsNone(model.states[0])
        reset, kept = model.states[1][0], model.states[1][1]
        self.assertEqual(float(reset.abs().max()), 0)
        torch.testing.assert_close(kept, torch.full_like(kept, 3.0), rtol=0, atol=1e-5)


class SelectValidationTests(unittest.TestCase):
    def test_both_banks_merged_and_duplicate_ids_rejected(self):
        from train_ssm_span import select_validation
        manifest = {'sequences': []}
        first = ([('lr1', 'hr1', 'id1')], {'id1': 'srcA'})
        second = ([('lr2', 'hr2', 'id2')], {'id2': 'srcB'})
        with patch('replay_recurrent_motion.validation_pairs',
                   side_effect=[first, second]) as pairs_fn:
            pairs, sources = select_validation('bank', 'reds_bank', manifest, None)
        self.assertEqual(pairs_fn.call_args_list,
                         [call('bank', manifest), call('reds_bank', manifest, reds=True)])
        self.assertEqual(pairs, first[0] + second[0])
        self.assertEqual(sources, {'id1': 'srcA', 'id2': 'srcB'})

        dup = [(([('lr', 'hr', 'x')], {'x': 'a'}), ([('lr', 'hr', 'x')], {'x': 'b'}))]
        with patch('replay_recurrent_motion.validation_pairs',
                   side_effect=dup[0]):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                select_validation('bank', 'reds_bank', manifest, None)

    def test_no_reds_bank_uses_manifest_validation(self):
        from train_ssm_span import select_validation
        manifest = {'sequences': [{'id': 'v1', 'source_id': 's1'}]}
        pairs, sources = select_validation(None, None, manifest, {'validation': 'V'})
        self.assertEqual(pairs, 'V')
        self.assertEqual(sources, {'v1': 's1'})


class ClipOptimizerGroupsTests(unittest.TestCase):
    def test_huge_scan_gradients_clipped_without_touching_small_sr_gradients(self):
        from train_ssm_span import clip_optimizer_groups
        scan = torch.nn.Parameter(torch.ones(10))
        sr = torch.nn.Parameter(torch.ones(10))
        scan.grad = torch.full((10,), 100.0)
        sr.grad = torch.full((10,), 1e-4)
        optimizer = torch.optim.SGD([{'params': [scan]}, {'params': [sr]}], lr=.1)
        clip_optimizer_groups(optimizer)
        # Independent per-group clipping: the huge group lands at norm 1 ...
        scan_norm = float(scan.grad.norm())
        self.assertLessEqual(scan_norm, 1 + 1e-6)
        self.assertGreater(scan_norm, .99)
        # ... while the small group is untouched. A joint global clip over the
        # combined ~316 norm would have rescaled these by ~300x.
        torch.testing.assert_close(sr.grad, torch.full((10,), 1e-4), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
