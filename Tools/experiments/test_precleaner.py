import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from architectures.precleaner import Precleaner, PrecleanedSPAN
from clean_lr_targets import batch_clean_lr, clean_lr_rgb, load_clean_targets
from eval_checkpoint import load
from train_precleaner import cleaner_loss
from train_span import Unshuffled


class PrecleanerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)

    def test_identity_initialization_and_frozen_backbone(self):
        base = Unshuffled(4, frames=1, scale=2).eval()
        x = torch.rand(1, 3, 16, 24)
        with torch.no_grad():
            expected = base(x)
        model = PrecleanedSPAN(copy.deepcopy(base), 4).train()
        self.assertFalse(model.sr.training)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        model(x).mean().backward()
        self.assertTrue(all(p.grad is None for p in model.sr.parameters()))
        self.assertGreater(float(model.cleaner.head.weight.grad.abs().sum()), 0)

    def test_bounded_correction_and_small_parameter_budget(self):
        cleaner = Precleaner()
        self.assertLess(sum(p.numel() for p in cleaner.parameters()), 5000)
        with torch.no_grad():
            cleaner.head.bias.fill_(100)
        x = torch.rand(1, 3, 12, 20)
        output = cleaner(x)
        self.assertLessEqual(float((output - x).abs().max().detach()), .125001)
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_checkpoint_round_trip_preserves_native_input_interface(self):
        model = PrecleanedSPAN(Unshuffled(4, frames=1, scale=2), 4).eval()
        with torch.no_grad():
            model.cleaner.head.bias.fill_(.1)
        x = torch.rand(1, 3, 16, 24)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'model.pth'
            torch.save({'model': model.state_dict(), 'architecture': 'precleaned_span2x',
                        'channels': 4, 'cleaner_channels': 4, 'frames': 1, 'scale': 2,
                        'version': model.version}, p)
            restored, _, frames = load(p, 'cpu')
            self.assertEqual(frames, 1)
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_clean_supervision_reduces_a_known_offset(self):
        model = Precleaner(4)
        source = torch.full((2, 3, 24, 24), .55)
        target = torch.full_like(source, .5)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = cleaner_loss(model(source), target)
        before.backward()
        optimizer.step()
        after = cleaner_loss(model(source), target)
        self.assertLess(float(after.detach()), float(before.detach()))

    def test_joint_sr_loss_reaches_cleaner_and_backbone(self):
        model = PrecleanedSPAN(Unshuffled(4, scale=2), 4,
                              train_backbone=True, fused_sr=True).train()
        x = torch.rand(1, 3, 16, 24)
        before = model.sr.core.conv_1.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model(x).square().mean().backward()
        self.assertGreater(float(model.sr.core.conv_1.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.cleaner.head.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sr.core.conv_1.weight))

    def test_disabled_cleaner_is_exact_and_receives_no_gradients(self):
        model = PrecleanedSPAN(Unshuffled(4, scale=2), 4, train_backbone=True,
                              fused_sr=True, use_cleaner=False).train()
        with torch.no_grad():
            model.cleaner.head.bias.fill_(10)
        x = torch.rand(1, 3, 16, 24)
        torch.testing.assert_close(model(x), model.sr(x), rtol=0, atol=0)
        model(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.cleaner.parameters()))
        self.assertGreater(float(model.sr.core.conv_1.weight.grad.abs().sum()), 0)

    def test_fused_checkpoint_roundtrip_and_policy_binding(self):
        for enabled in (True, False):
            model = PrecleanedSPAN(Unshuffled(4, scale=2), 4, train_backbone=True,
                                  fused_sr=True, use_cleaner=enabled).eval()
            with torch.no_grad():
                model.cleaner.head.bias.fill_(.2)
            x = torch.rand(1, 3, 16, 24)
            state = {'architecture': 'precleaned_span2x', 'model': model.state_dict(),
                     'channels': 4, 'cleaner_channels': 4, 'scale': 2, 'frames': 1,
                     'version': model.version, 'fused_sr': True, 'use_cleaner': enabled,
                     'experiment': {'fused_sr': True, 'use_cleaner': enabled}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pth'
                torch.save(state, path)
                restored, _, _ = load(path, 'cpu')
                self.assertEqual(restored.use_cleaner, enabled)
                torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
                state['experiment']['use_cleaner'] = not enabled
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    load(path, 'cpu')


class CleanTargetTest(unittest.TestCase):
    def test_lanczos_crop_matches_full_frame_interior(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (3, 64, 96, 3), dtype=np.uint8)
        full = clean_lr_rgb(image)
        crop = clean_lr_rgb(image[:, 16:48, 24:72])
        a, b = crop[:, 4:-4, 4:-4], full[:, 12:20, 16:32]
        self.assertLessEqual(int(np.abs(a.astype(int) - b.astype(int)).max()), 1)

    def test_cache_is_split_bound_and_hash_checked(self):
        rng = np.random.default_rng(9)
        lr = rng.integers(0, 256, (3, 16, 24, 3), dtype=np.uint8)
        hr = lr.repeat(2, 1).repeat(2, 2)
        data = {'train': [(lr, hr, 'train-one')], 'validation': [(lr, hr, 'val-two')]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cache'
            target, manifest = load_clean_targets(root, data, 'bank-hash')
            restored, _ = load_clean_targets(root, data, 'bank-hash')
            np.testing.assert_array_equal(target['train-one'], restored['train-one'])
            with self.assertRaises(ValueError):
                load_clean_targets(root, data, 'different-bank')
            path = root / manifest['sequences'][0]['file']
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                load_clean_targets(root, data, 'bank-hash')

    def test_clean_targets_follow_all_crop_time_and_orientation_augmentation(self):
        rng = np.random.default_rng(21)
        lr = rng.integers(0, 256, (6, 24, 32, 3), dtype=np.uint8)
        hr = lr.repeat(2, 1).repeat(2, 2)
        source, clean, reference = batch_clean_lr([(lr, hr, 'one')], {'one': lr.copy()}, rng, 10, 3, 16)
        torch.testing.assert_close(source, clean, rtol=0, atol=0)
        torch.testing.assert_close(source, reference[..., ::2, ::2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
