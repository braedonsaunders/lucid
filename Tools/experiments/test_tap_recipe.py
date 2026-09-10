"""CPU integration of tap arm training, validation variants and checkpoint loading.

CUDA placement and perceptual scoring are replaced; optimization, sequence
feeding, serialization and the versioned loader remain real. These tests do not
claim GPU parity or perceptual quality.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_tap_span as trainer
from architectures.tap_span import TapSPAN, load_tap_checkpoint
from tap_frame_feeding import run_tap_sequence
from train_span import Unshuffled


class TapRecipeTests(unittest.TestCase):
    def test_control_arm_needs_joint(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        initial = Unshuffled(8, scale=2).eval()
        rng = np.random.default_rng(58)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            init_path = root / 'init.pth'
            torch.save(initial.state_dict(), init_path)
            with ExitStack() as stack:
                stack.enter_context(patch('sys.argv', ['trainer', '--bank', str(root),
                    '--init', str(init_path), '--out', str(root / 'result'),
                    '--tap-mode', 'none', '--steps', '2', '--batch', '1', '--crop', '32']))
                stack.enter_context(patch.object(trainer, 'load_bank', return_value=(manifest, data)))
                stack.enter_context(patch.object(trainer, 'load', side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock CPU integration'))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value))
                stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value))
                with self.assertRaisesRegex(SystemExit, '2'):
                    trainer.main()

    def test_tap_arms_training_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        initial = Unshuffled(8, scale=2).eval()
        rng = np.random.default_rng(58)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        for joint, tap in ((True, 'none'), (False, 'full'), (False, 'luma')):
            with self.subTest(joint=joint, tap=tap), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'manifest.json').write_text(json.dumps(manifest))
                init_path = root / 'init.pth'
                torch.save(initial.state_dict(), init_path)
                out = root / 'result'
                argv = ['trainer', '--bank', str(root), '--init', str(init_path),
                        '--out', str(out), '--tap-mode', tap, '--steps', '2', '--batch', '1', '--crop', '32']
                if joint:
                    argv.append('--joint')
                with ExitStack() as stack:
                    stack.enter_context(patch('sys.argv', argv))
                    stack.enter_context(patch.object(trainer, 'load_bank', return_value=(manifest, data)))
                    stack.enter_context(patch.object(trainer, 'load', side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)))
                    stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                    stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock CPU integration'))
                    stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value))
                    stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **k: value))
                    scorer = stack.enter_context(patch.object(trainer, 'Scorer')).return_value
                    scorer.spatial.side_effect = lambda a, b: {'mse': float(((np.asarray(a).astype(float) - np.asarray(b)) ** 2).mean())}
                    trainer.main()
                with self.assertRaisesRegex(ValueError, 'explicit stateful loader'):
                    trainer.load(out / 'step000002.pth', 'cpu')
                loaded, checkpoint = load_tap_checkpoint(out / 'step000002.pth')
                self.assertEqual(checkpoint['step'], 2)
                self.assertEqual(checkpoint['tap_mode'], tap)
                (complete := json.loads((out / 'complete.json').read_text()))
                self.assertTrue(complete['complete'])
                if not joint:
                    self.assertTrue(complete['backbone_unchanged'])
                experiment = json.loads((out / 'experiment.json').read_text())
                self.assertEqual(experiment['tap_mode'], tap)
                self.assertIn('first_input_sha256', experiment)

    def test_sequence_gradient_reaches_raw_tap(self):
        torch.set_num_threads(2)
        torch.manual_seed(59)
        model = TapSPAN(Unshuffled(8, scale=2), tap='full')
        with torch.no_grad():
            model.tap_conv.weight.normal_(std=.01)
        source = torch.rand(1, 3, 3, 32, 40)
        inputs = (source, source + .01)
        pair = torch.stack([inputs[0][:, 2], inputs[1][:, 2]], 1).requires_grad_()
        output = model(pair[:, 0], pair[:, 1])
        gradient = torch.autograd.grad(output.square().mean(), pair)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[:, 1].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
