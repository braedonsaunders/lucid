"""CPU integration of arm training, validation variants and checkpoint loading.

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

import train_full_lr_feature_span as trainer
from architectures.full_lr_feature_span import FullLRFeatureSPAN, load_full_lr_checkpoint
from full_lr_frame_feeding import run_full_lr_sequence
from train_span import Unshuffled


class FullLRRecipeTests(unittest.TestCase):
    def test_sequence_gradient_reaches_past_only_with_memory(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        model = FullLRFeatureSPAN(Unshuffled(8, scale=2))
        with torch.no_grad():
            model.project.weight.normal_(std=.01)
        source = torch.rand(1, 1, 3, 32, 40).repeat(1, 3, 1, 1, 1).requires_grad_()
        for use_history in (True, False):
            output = run_full_lr_sequence(model, source, use_history=use_history)
            gradient = torch.autograd.grad(output[:, -1].square().mean(), source)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[:, -1].abs().sum()), 0)
            if use_history:
                self.assertGreater(float(gradient[:, 0].abs().sum()), 0)
            else:
                self.assertEqual(float(gradient[:, :-1].abs().sum()), 0)

    def test_alignment_arms_training_roundtrip_and_capacity_control(self):
        torch.set_num_threads(2)
        torch.manual_seed(58)
        initial = Unshuffled(8, scale=2).eval()
        rng = np.random.default_rng(58)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'val')]}
        manifest = {'frames': 3, 'sequences': [{'id': 'val', 'source_id': 'heldout'}]}
        first = None
        for joint in (False, True):
            for current_only, state_warp in ((True, 'full_lr'), (False, 'packed_half_lr'), (False, 'full_lr')):
                with self.subTest(joint=joint, current_only=current_only, state_warp=state_warp), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / 'manifest.json').write_text(json.dumps(manifest))
                    init_path = root / 'init.pth'
                    torch.save(initial.state_dict(), init_path)
                    out = root / 'result'
                    argv = ['trainer', '--bank', str(root), '--init', str(init_path),
                            '--out', str(out), '--state-warp', state_warp, '--steps', '2', '--batch', '1', '--crop', '32']
                    if joint:
                        argv.append('--joint')
                    if current_only:
                        argv.append('--no-history')
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
                    loaded, checkpoint = load_full_lr_checkpoint(out / 'step000002.pth')
                    self.assertEqual(checkpoint['step'], 2)
                    self.assertEqual(checkpoint['state_warp'], state_warp)
                    self.assertEqual(checkpoint['state_channels'], 8)
                    self.assertEqual(checkpoint['experiment']['use_history'], not current_only)
                    self.assertGreater(float(loaded.project.weight.detach().abs().sum()), 0)
                    receipt = json.loads((out / 'complete.json').read_text())
                    self.assertEqual(receipt['backbone_unchanged'], not joint)
                    report = json.loads((out / 'validation.json').read_text())
                    self.assertEqual(len(report['rows']), 12)
                    variants = {v: {r['frame']: r['metrics'] for r in report['rows'] if r['variant'] == v}
                                for v in ('base', 'initial', 'current_only', 'full_lr_features')}
                    self.assertEqual(variants['full_lr_features'][0], variants['current_only'][0])
                    if current_only:
                        self.assertEqual(variants['full_lr_features'], variants['current_only'])
                    if not joint:
                        self.assertEqual(variants['base'], variants['initial'])
                    experiment = checkpoint['experiment']
                    hashes = tuple(experiment[k] for k in ('initial_branch_sha256',
                        'first_decoded_sequence_sha256', 'first_input_sha256', 'first_output_sha256'))
                    if first is None:
                        first = hashes
                    self.assertEqual(first, hashes)


if __name__ == '__main__':
    unittest.main()
