"""CPU tests for the output-resolution refiner head.

CUDA placement and perceptual scoring are replaced; optimization,
sequence feeding, serialization and the versioned loader remain real.
These tests do not claim GPU parity or perceptual quality.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.output_refiner import OutputRefinerSPAN, load_output_refiner_checkpoint
from train_span import Unshuffled


class OutputRefinerTests(unittest.TestCase):
    def test_zero_init_preserves_backbone_exactly(self):
        torch.manual_seed(21)
        backbone = Unshuffled(4, scale=2).eval()
        for width in (8, 16):
            with self.subTest(width=width):
                model = OutputRefinerSPAN(copy.deepcopy(backbone), head_channels=width).eval()
                image = torch.rand(1, 3, 16, 16)
                with torch.no_grad():
                    torch.testing.assert_close(model(image), backbone(image), rtol=0, atol=0)

    def test_head_channels_zero_is_passthrough(self):
        torch.manual_seed(22)
        backbone = Unshuffled(4, scale=2).eval()
        model = OutputRefinerSPAN(copy.deepcopy(backbone), head_channels=0).eval()
        self.assertIsNone(model.head)
        self.assertEqual(sum(p.numel() for p in model.head_parameters()), 0)
        image = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(model(image), backbone(image), rtol=0, atol=0)

    def test_requires_single_frame_2x_unit_range_geometry(self):
        torch.manual_seed(23)
        for kwargs in ({'frames': 3}, {'scale': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OutputRefinerSPAN(Unshuffled(4, **kwargs), head_channels=8)
        with self.assertRaises(ValueError):
            OutputRefinerSPAN(Unshuffled(4, scale=2), head_channels=-1)
        with self.assertRaises(ValueError):
            OutputRefinerSPAN(Unshuffled(4, scale=2), head_channels=8.0)

    def test_frozen_trunk_stays_frozen_but_head_learns(self):
        torch.manual_seed(24)
        torch.set_num_threads(2)
        model = OutputRefinerSPAN(Unshuffled(4, scale=2), head_channels=8).train()
        self.assertTrue(all(not p.requires_grad for p in model.sr.parameters()))
        with torch.no_grad():
            model.head[4].weight.normal_(std=.01)
        fixed = {k: v.detach().clone() for k, v in model.sr.state_dict().items()}
        image = torch.rand(1, 3, 16, 16)
        optimizer = torch.optim.AdamW(model.head_parameters(), lr=1e-3)
        optimizer.zero_grad()
        (model(image) - .5).square().mean().backward()
        self.assertGreater(float(model.head[0].weight.grad.abs().sum()), 0)
        optimizer.step()
        for name, value in model.sr.state_dict().items():
            torch.testing.assert_close(value, fixed[name], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertFalse(model.sr.training)

    def test_joint_mode_trains_trunk_too(self):
        torch.manual_seed(25)
        torch.set_num_threads(2)
        model = OutputRefinerSPAN(Unshuffled(4, scale=2), train_backbone=True,
                                  head_channels=8).train()
        self.assertTrue(any(p.requires_grad for p in model.sr.parameters()))
        image = torch.rand(1, 3, 16, 16)
        optimizer = torch.optim.AdamW(
            list(model.head_parameters()) + list(model.sr.parameters()), lr=1e-3)
        optimizer.zero_grad()
        (model(image) - .5).square().mean().backward()
        # The zero-initialized final convolution blocks first-step gradient
        # to earlier head layers by design; the trunk still receives it.
        trunk_grad = sum(float(p.grad.abs().sum()) for p in model.sr.parameters()
                         if p.grad is not None)
        self.assertGreater(trunk_grad, 0)

    def test_checkpoint_roundtrip_and_loader_gates(self):
        torch.manual_seed(26)
        torch.set_num_threads(2)
        model = OutputRefinerSPAN(Unshuffled(4, scale=2), head_channels=8).eval()
        image = torch.rand(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'refined.pth'
            torch.save({'model': model.state_dict(), 'architecture': 'output_refiner2x',
                        'scale': 2, 'head_channels': 8, 'channels': 4, 'version': 1}, path)
            restored, report = load_output_refiner_checkpoint(path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
            self.assertEqual(report['head_channels'], 8)
            traced = torch.jit.trace(restored, image)
            with torch.no_grad():
                torch.testing.assert_close(traced(image), model(image), rtol=0, atol=1e-6)
            bad = torch.load(path, map_location='cpu', weights_only=False)
            bad['architecture'] = 'full_lr_feature_span2x'
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                load_output_refiner_checkpoint(path, 'cpu')
            bad['architecture'] = 'output_refiner2x'
            bad['head_channels'] = 16
            torch.save(bad, path)
            with self.assertRaises((ValueError, RuntimeError)):
                load_output_refiner_checkpoint(path, 'cpu')


class OutputRefinerRecipeTests(unittest.TestCase):
    def test_training_roundtrip_and_frozen_backbone(self):
        torch.manual_seed(27)
        torch.set_num_threads(2)
        import train_output_refiner as trainer
        initial = Unshuffled(4, scale=2).eval()
        rng = np.random.default_rng(27)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            init_path = root / 'init.pth'
            torch.save(initial.state_dict(), init_path)
            out = root / 'result'
            argv = ['trainer', '--bank', str(root), '--init', str(init_path),
                    '--out', str(out), '--steps', '2', '--batch', '1', '--crop', '32',
                    '--frames', '3', '--seed', '27', '--no-history',
                    '--dino-gan-weight', '0', '--head-channels', '8']
            with self.assertRaisesRegex(ValueError, 'authorized CUDA worker'):
                with patch.object(sys, 'argv', argv), \
                     patch('torch.cuda.is_available', return_value=False):
                    trainer.main()
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', argv))
                stack.enter_context(patch.object(trainer, 'load_bank', return_value=(manifest, data)))
                stack.enter_context(patch.object(trainer, 'load',
                                                 side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock CPU integration'))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value))
                stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value))
                scorer = stack.enter_context(patch.object(trainer, 'Scorer')).return_value
                scorer.spatial.side_effect = lambda a, b: {'mse': float(((np.asarray(a).astype(float) - np.asarray(b)) ** 2).mean())}
                seq = stack.enter_context(patch.object(trainer, 'run_head_sequence',
                                                       wraps=trainer.run_head_sequence))
                objective = stack.enter_context(patch.object(trainer, 'training_objective',
                                                             wraps=trainer.training_objective))
                trainer.main()
            saved = torch.load(out / 'step000002.pth', map_location='cpu', weights_only=False)
            self.assertEqual(saved['architecture'], 'output_refiner2x')
            self.assertEqual(saved['head_channels'], 8)
            self.assertTrue((out / 'complete.json').exists())
            complete = json.loads((out / 'complete.json').read_text())
            self.assertTrue(complete['backbone_unchanged'])
            frozen = json.loads((out / 'experiment.json').read_text())['frozen_backbone_sha256']
            from train_presented_detail import state_digest
            model, _ = load_output_refiner_checkpoint(out / 'step000002.pth', 'cpu')
            self.assertEqual(state_digest(model.sr.state_dict()), frozen)
            self.assertTrue(objective.called)
            self.assertTrue(seq.called)

    def test_no_head_control_requires_joint(self):
        torch.manual_seed(28)
        import train_output_refiner as trainer
        initial = Unshuffled(4, scale=2).eval()
        rng = np.random.default_rng(28)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['trainer', '--bank', str(root), '--init', str(root / 'init.pth'),
                    '--out', str(root / 'result'), '--head-channels', '0',
                    '--no-history', '--dino-gan-weight', '0']
            with self.assertRaisesRegex(SystemExit, '2'):
                with patch.object(sys, 'argv', argv), \
                     patch('torch.cuda.is_available', return_value=True), \
                     patch.object(trainer, 'load_bank', return_value=(manifest, data)), \
                     patch.object(trainer, 'load',
                                  side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)):
                    trainer.main()

    def test_eval_control_preserves_frozen_baseline(self):
        torch.manual_seed(30)
        torch.set_num_threads(2)
        import train_output_refiner as trainer
        import eval_output_refiner as evaluator
        initial = Unshuffled(4, scale=2).eval()
        rng = np.random.default_rng(30)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            init_path = root / 'init.pth'
            torch.save(initial.state_dict(), init_path)
            out = root / 'eval'
            argv = ['evaluator', '--bank', str(root), '--reds-bank', str(root),
                    '--init', str(init_path), '--out', str(out)]
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', argv))
                stack.enter_context(patch.object(evaluator, 'load_bank', return_value=(manifest, data)))
                stack.enter_context(patch.object(evaluator, 'load_any',
                                                 side_effect=lambda *_: (trainer.OutputRefinerSPAN(
                                                     copy.deepcopy(initial), head_channels=0).eval(), {'step': 0})))
                stack.enter_context(patch.object(evaluator, 'select_validation',
                                                 return_value=(data['validation'], {'val': 'heldout'})))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock CPU integration'))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value))
                stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value))
                scorer = stack.enter_context(patch.object(trainer, 'Scorer')).return_value
                scorer.spatial.side_effect = lambda a, b: {'mse': float(((np.asarray(a).astype(float) - np.asarray(b)) ** 2).mean())}
                evaluator.main()
            report = json.loads((out / 'validation.json').read_text())
            self.assertEqual(set(report['summary']), {'base', 'current_only', 'refined', 'initial'})
            self.assertEqual(report['summary']['refined'], report['summary']['base'])
            self.assertTrue((out / 'complete.json').exists())

    def test_recipe_requires_no_history(self):
        torch.manual_seed(29)
        import train_output_refiner as trainer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['trainer', '--bank', str(root), '--init', str(root / 'init.pth'),
                    '--out', str(root / 'result'), '--dino-gan-weight', '0']
            with self.assertRaises(SystemExit):
                with patch.object(sys, 'argv', argv):
                    with patch('torch.cuda.is_available', return_value=True):
                        trainer.main()


if __name__ == '__main__':
    unittest.main()
