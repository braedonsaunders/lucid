"""Bounded CPU retention behavior and real two-step training/loader coverage.

Only CUDA placement, input-bank loading and perceptual scoring are replaced;
recurrence, optimization, checkpoint serialization and loading execute normally.
"""
from contextlib import ExitStack, redirect_stderr
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_full_lr_feature_span as trainer
from architectures.full_lr_feature_span import FullLRFeatureSPAN, load_full_lr_checkpoint
from full_lr_frame_feeding import run_full_lr_sequence
from train_span import Unshuffled


class FullLRRetentionTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        self.previous_rng = torch.get_rng_state()
        torch.set_num_threads(2)
        torch.manual_seed(58)

    def tearDown(self):
        torch.set_rng_state(self.previous_rng)
        torch.set_num_threads(self.previous_threads)

    def test_bias_is_only_initial_difference_and_does_not_consume_rng(self):
        backbone = Unshuffled(8, scale=2).eval()
        models, rng_states, draws = [], [], []
        for bias in (-2., 0.):
            torch.manual_seed(123)
            models.append(FullLRFeatureSPAN(copy.deepcopy(backbone), initial_decay_bias=bias).eval())
            rng_states.append(torch.get_rng_state().clone())
            draws.append(torch.rand(12))
        left, right = (model.state_dict() for model in models)
        self.assertEqual(left.keys(), right.keys())
        self.assertEqual([key for key in left if not torch.equal(left[key], right[key])],
                         ['decay.bias'])
        for model, bias in zip(models, (-2., 0.)):
            torch.testing.assert_close(model.decay.bias, torch.full_like(model.decay.bias, bias),
                                       rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(model.project.weight).item(), 0)
        self.assertTrue(torch.equal(rng_states[0], rng_states[1]))
        self.assertTrue(torch.equal(draws[0], draws[1]))
        current, raw = torch.rand(2, 3, 16, 24), torch.rand(2, 3, 16, 24)
        past = torch.randn(2, 8, 16, 24)
        with torch.no_grad():
            for model in models:
                expected = model.sr(current)
                for state in (None, past):
                    actual, _ = model(current, raw, torch.ones(2, 1, 16, 24), state)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_repeated_observations_follow_retention_geometric_sum(self):
        backbone = Unshuffled(8, scale=2)
        current, raw = torch.rand(1, 3, 16, 24), torch.rand(1, 3, 16, 24)
        confidence = torch.ones(1, 1, 16, 24)
        for bias, approximate in ((-2., .1192), (0., .5)):
            with self.subTest(bias=bias), torch.no_grad():
                model = FullLRFeatureSPAN(copy.deepcopy(backbone), initial_decay_bias=bias).eval()
                retention = 1 / (1 + math.exp(-bias))
                self.assertAlmostEqual(retention, approximate, places=4)
                observed = model.raw_encoder(raw)
                past = torch.full_like(observed, .75)
                state = past
                for step in range(1, 5):
                    _, state = model(current, raw, confidence, state)
                    expected = observed * sum(retention ** i for i in range(step)) + past * retention ** step
                    torch.testing.assert_close(state, expected, rtol=1e-6, atol=2e-7)
                _, first = model(current, raw, confidence, None)
                torch.testing.assert_close(first, observed, rtol=0, atol=0)
                _, second = model(current, raw, confidence, first)
                torch.testing.assert_close(second, observed * (1 + retention), rtol=1e-6, atol=2e-7)

    def test_zero_confidence_discards_past_but_keeps_fresh_observation(self):
        current, raw = torch.rand(1, 3, 16, 24), torch.rand(1, 3, 16, 24)
        for bias in (-2., 0.):
            with self.subTest(bias=bias), torch.no_grad():
                model = FullLRFeatureSPAN(Unshuffled(8, scale=2), initial_decay_bias=bias).eval()
                model.project.weight.fill_(.02)
                observed = model.raw_encoder(raw)
                self.assertGreater(float(observed.abs().sum()), 0)
                confidence = torch.zeros(1, 1, 16, 24)
                output, state = model(current, raw, confidence, torch.full_like(observed, 17))
                fresh_output, fresh_state = model(current, raw, confidence, None)
                torch.testing.assert_close(state, observed, rtol=0, atol=0)
                torch.testing.assert_close(fresh_state, observed, rtol=0, atol=0)
                torch.testing.assert_close(output, fresh_output, rtol=0, atol=0)
                self.assertFalse(torch.equal(output, model.sr(current)))

    def test_constructor_rejects_nonfinite_bias(self):
        backbone = Unshuffled(8, scale=2)
        for bias in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(bias=bias), self.assertRaisesRegex(ValueError, 'finite initial decay bias'):
                FullLRFeatureSPAN(copy.deepcopy(backbone), initial_decay_bias=bias)

    def test_cli_rejects_nonfinite_bias_before_cuda_or_bank_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bias in ('nan', 'inf', '-inf'):
                argv = ['trainer', '--bank', str(root), '--init', str(root / 'init.pth'),
                        '--out', str(root / 'result'), '--initial-decay-bias=' + bias]
                stderr = io.StringIO()
                with self.subTest(bias=bias), patch('sys.argv', argv), redirect_stderr(stderr), \
                        patch.object(torch.cuda, 'is_available') as cuda, \
                        patch.object(trainer, 'load_bank') as bank:
                    with self.assertRaises(SystemExit) as caught:
                        trainer.main()
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn('finite initial decay bias required', stderr.getvalue())
                    cuda.assert_not_called()
                    bank.assert_not_called()
                    self.assertFalse((root / 'result').exists())

    def test_bias_zero_two_step_training_records_initial_hash_and_roundtrips(self):
        initial = Unshuffled(8, scale=2).eval()
        lr = np.random.default_rng(58).integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        constructed, initial_states = [], []

        def construct(*args, **kwargs):
            model = FullLRFeatureSPAN(*args, **kwargs)
            constructed.append(model)
            initial_states.append(copy.deepcopy(model.state_dict()))
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            torch.save(initial.state_dict(), root / 'init.pth')
            out = root / 'result'
            argv = ['trainer', '--bank', str(root), '--init', str(root / 'init.pth'),
                    '--out', str(out), '--steps', '2', '--batch', '1', '--crop', '32',
                    '--seed', '123', '--initial-decay-bias', '0']
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', argv))
                stack.enter_context(patch.object(trainer, 'load_bank', return_value=(manifest, data)))
                stack.enter_context(patch.object(trainer, 'load', side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)))
                stack.enter_context(patch.object(trainer, 'FullLRFeatureSPAN', side_effect=construct))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock CPU integration'))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value))
                stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value))
                scorer = stack.enter_context(patch.object(trainer, 'Scorer')).return_value
                scorer.spatial.side_effect = lambda a, b: {'mse': float(((np.asarray(a).astype(float) - np.asarray(b)) ** 2).mean())}
                trainer.main()
            self.assertEqual(len(constructed), 1)
            trained, start = constructed[0], initial_states[0]
            self.assertTrue(all(p.device.type == 'cpu' for p in trained.parameters()))
            self.assertEqual(torch.count_nonzero(start['decay.bias']).item(), 0)
            self.assertFalse(torch.equal(start['project.weight'], trained.project.weight))
            shared = {k: v for k, v in start.items() if not k.startswith('sr.') and k != 'decay.bias'}
            expected_hash = trainer.state_digest(shared)
            torch.manual_seed(123)
            default = FullLRFeatureSPAN(copy.deepcopy(initial))
            self.assertTrue(torch.equal(default.decay.bias, torch.full_like(default.decay.bias, -2.)))
            self.assertEqual(expected_hash, trainer.state_digest({k: v for k, v in default.state_dict().items()
                             if not k.startswith('sr.') and k != 'decay.bias'}))
            restored, checkpoint = load_full_lr_checkpoint(out / 'step000002.pth')
            self.assertEqual(checkpoint['step'], 2)
            for experiment in (checkpoint['experiment'], json.loads((out / 'experiment.json').read_text())):
                self.assertEqual(experiment['args']['initial_decay_bias'], 0.)
                self.assertEqual(experiment['initial_shared_branch_sha256'], expected_hash)
                self.assertEqual(experiment['initial_branch_sha256'], trainer.state_digest(
                    {k: v for k, v in start.items() if not k.startswith('sr.')}))
            source = torch.from_numpy(lr.copy()).permute(0, 3, 1, 2)[None].float() / 255
            with torch.no_grad():
                torch.testing.assert_close(run_full_lr_sequence(restored, source),
                                           run_full_lr_sequence(trained.eval(), source), rtol=0, atol=0)
            for key, value in trained.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            receipt = json.loads((out / 'complete.json').read_text())
            self.assertTrue(receipt['complete'])
            self.assertTrue(receipt['backbone_unchanged'])
            self.assertEqual(receipt['steps'], 2)


if __name__ == '__main__':
    unittest.main()
