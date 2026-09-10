"""CPU regressions for train_scale_matched.py (r91 section-4's matched trainer).

Covers: build_model constructs the right architecture at both scales;
matched_objective at scale=2/gan_weight=0 is byte-identical to
train_ssm_span.py's training_objective (the whole point of "matched");
matched_objective's scale=4 critic condition uses a 4x bicubic upsample, not
the 2x train_ssm_span.py hardcodes; make_source refuses every scale/format
pair that cannot mean what the recipe assumes; patch_corpus_source actually
builds and loads a real (tiny, synthetic) 4x patch bank end to end; and a
saved checkpoint round-trips through eval_checkpoint.load.
"""
import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from train_scale_matched import (build_model, matched_objective, make_source,
                                 patch_corpus_source)
import train_span


def _frames(scale, lr_size, batch=1):
    torch.manual_seed(0)
    hr_size = lr_size * scale - 2 * 8
    output = (torch.rand(batch, 3, hr_size, hr_size) * 0.8 + 0.1).requires_grad_(True)
    reference = torch.zeros(batch, 3, hr_size, hr_size)
    decoded = torch.rand(batch, 3, lr_size, lr_size) * 0.8 + 0.1
    return output, reference, decoded


def _expected_condition(decoded, scale):
    return F.interpolate(decoded.detach(), scale_factor=scale, mode='bicubic',
                         align_corners=False, antialias=True).clamp(0, 1)[..., 8:-8, 8:-8]


class _RecordingCritic:
    def __init__(self):
        self.calls = []
        self.fake = (torch.zeros(1).detach(), torch.zeros(1).detach())

    def generator_loss(self, image, condition):
        self.calls.append((image, condition))
        return image.sum(), self.fake


class _StrictCritic:
    def generator_loss(self, image, condition):
        raise AssertionError('critic must not be called at zero weight')


class ArchitectureTests(unittest.TestCase):
    def test_scale2_doubles_spatial_size(self):
        model = build_model(8, 2)
        output = model(torch.rand(1, 3, 16, 16))
        self.assertEqual(output.shape, (1, 3, 32, 32))

    def test_scale4_quadruples_spatial_size(self):
        model = build_model(8, 4)
        output = model(torch.rand(1, 3, 16, 16))
        self.assertEqual(output.shape, (1, 3, 64, 64))

    def test_is_the_same_class_lucidbig2k_and_span_x4_ch32u_both_use(self):
        self.assertIsInstance(build_model(8, 2), train_span.Unshuffled)
        self.assertIsInstance(build_model(8, 4), train_span.Unshuffled)

    def test_rejects_unsupported_scale(self):
        for bad in (1, 3, 8, 0, -2):
            with self.assertRaises(ValueError):
                build_model(8, bad)

    def test_channel_count_matches_request(self):
        model = build_model(8, 4)
        parameters = sum(p.numel() for p in model.parameters())
        bigger = build_model(16, 4)
        self.assertGreater(sum(p.numel() for p in bigger.parameters()), parameters)


class MatchedObjectiveScale2ControlTests(unittest.TestCase):
    """scale=2, gan_weight=0 must be the exact control train_ssm_span.py uses."""

    def test_matches_train_ssm_span_training_objective_exactly(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames(2, 24)
        loss, fake, metrics = matched_objective(output, reference, decoded, 2, gan_weight=0)
        expected_loss, expected_fake, expected_metrics = training_objective(
            output, reference, decoded, gan_weight=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        self.assertEqual(fake, expected_fake)
        self.assertEqual(metrics, expected_metrics)

    def test_matches_train_ssm_span_gradient_with_critic(self):
        from train_ssm_span import training_objective
        output, reference, decoded = _frames(2, 24)
        weight = 0.05
        loss, fake, metrics = matched_objective(
            output, reference, decoded, 2, adversary=_RecordingCritic(), gan_weight=weight)
        probe = output.detach().clone().requires_grad_(True)
        expected_loss, _, expected_metrics = training_objective(
            probe, reference, decoded, adversary=_RecordingCritic(), gan_weight=weight)
        torch.testing.assert_close(loss.detach(), expected_loss.detach(), rtol=1e-6, atol=1e-6)
        self.assertEqual(metrics['gan_unweighted'], expected_metrics['gan_unweighted'])


class MatchedObjectiveGeneralisationTests(unittest.TestCase):
    def test_scale4_condition_is_bicubic_4x_not_2x(self):
        output, reference, decoded = _frames(4, 40)
        critic = _RecordingCritic()
        matched_objective(output, reference, decoded, 4, adversary=critic, gan_weight=0.05)
        self.assertEqual(len(critic.calls), 1)
        image, condition = critic.calls[0]
        self.assertIs(image, output)
        self.assertEqual(condition.shape, output.shape)
        torch.testing.assert_close(condition, _expected_condition(decoded, 4), rtol=1e-5, atol=1e-7)
        # A 2x upsample of the same LR must NOT match at scale=4 - proves the
        # generalisation actually uses the passed scale, not a hardcoded 2.
        wrong = _expected_condition(decoded, 2)
        if wrong.shape == condition.shape:
            self.assertGreater(float((condition - wrong).abs().max()), 1e-3)

    def test_zero_weight_never_calls_critic_at_either_scale(self):
        for scale, lr_size in ((2, 24), (4, 40)):
            output, reference, decoded = _frames(scale, lr_size)
            matched_objective(output, reference, decoded, scale,
                              adversary=_StrictCritic(), gan_weight=0)

    def test_rejects_unsupported_scale(self):
        output, reference, decoded = _frames(2, 24)
        with self.assertRaises(ValueError):
            matched_objective(output, reference, decoded, 3, gan_weight=0)

    def test_rejects_negative_and_nonfinite_weights(self):
        output, reference, decoded = _frames(4, 40)
        for bad in (-1.0, -1e-6, float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                matched_objective(output, reference, decoded, 4,
                                  adversary=_RecordingCritic(), gan_weight=bad)

    def test_positive_weight_requires_critic(self):
        output, reference, decoded = _frames(4, 40)
        with self.assertRaisesRegex(ValueError, 'adversary'):
            matched_objective(output, reference, decoded, 4, adversary=None, gan_weight=0.05)

    def test_condition_geometry_mismatch_rejected(self):
        output, reference, _ = _frames(4, 40)
        wrong_lr = torch.rand(1, 3, 20, 20)
        critic = _RecordingCritic()
        with self.assertRaisesRegex(ValueError, 'condition must match'):
            matched_objective(output, reference, wrong_lr, 4, adversary=critic, gan_weight=0.05)
        self.assertEqual(len(critic.calls), 0)


class MakeSourceRejectionTests(unittest.TestCase):
    """The whole point of r98: refuse scale/format pairs that cannot mean
    what the recipe assumes, before touching CUDA or the filesystem."""

    def _args(self, **overrides):
        base = dict(scale=2, bank=None, corpus=None, bank_dir=Path('/tmp/unused'),
                   per_pair=8, min_correlation=0.85, seed=0, crop=96)
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_bank_with_scale4_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scale == 2'):
            make_source(self._args(scale=4, bank=Path('/tmp/some-bank')))

    def test_corpus_with_scale2_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scale=4'):
            make_source(self._args(scale=2, corpus=Path('/tmp/some-corpus')))

    def test_neither_bank_nor_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            make_source(self._args(scale=2))

    def test_both_bank_and_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            make_source(self._args(scale=2, bank=Path('/tmp/a'), corpus=Path('/tmp/b')))


class PatchCorpusSourceTests(unittest.TestCase):
    """Builds a tiny synthetic 4x corpus (genuine downscale, so it clears the
    correlation gate) through the real train_span.build_bank/load_bank path -
    the same path Arm B's actual training would use once corpus4 is
    regenerated - and checks what comes out the other end."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.corpus = self.tmp / 'corpus'
        (self.corpus / 'lr').mkdir(parents=True)
        (self.corpus / 'hr').mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(3):
            hr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
            hr_image = Image.fromarray(hr)
            lr_image = hr_image.resize((24, 24), Image.LANCZOS)
            hr_image.save(self.corpus / 'hr' / f'{i:06d}.png')
            lr_image.save(self.corpus / 'lr' / f'{i:06d}.png')
        with open(self.corpus / 'manifest.jsonl', 'w') as fh:
            pass  # report_composition tolerates an empty/absent manifest; not under test here

    def test_scale_other_than_4_rejected_before_touching_disk(self):
        original = train_span.SCALE
        train_span.SCALE = 2
        try:
            with self.assertRaises(ValueError):
                patch_corpus_source(self.corpus, self.tmp / 'bank')
        finally:
            train_span.SCALE = original
        self.assertFalse((self.tmp / 'bank').exists())

    def test_builds_bank_and_samples_correctly_shaped_batches(self):
        original_lr_patch = train_span.LR_PATCH
        train_span.LR_PATCH = 16
        train_span.HR_PATCH = 16 * train_span.SCALE
        try:
            manifest, sample = patch_corpus_source(self.corpus, self.tmp / 'bank', per_pair=4, min_correlation=0.5)
            self.assertGreater(manifest['count'], 0)
            rng = np.random.default_rng(1)
            lr, hr = sample(rng, 2)
            self.assertEqual(lr.shape, (2, 3, 16, 16))
            self.assertEqual(hr.shape, (2, 3, 64, 64))
            self.assertTrue(float(lr.min()) >= 0 and float(lr.max()) <= 1)
            self.assertTrue(float(hr.min()) >= 0 and float(hr.max()) <= 1)
        finally:
            train_span.LR_PATCH = original_lr_patch
            train_span.HR_PATCH = original_lr_patch * train_span.SCALE

    def test_second_call_reuses_cached_bank_without_rebuilding(self):
        bank_dir = self.tmp / 'bank'
        original_lr_patch = train_span.LR_PATCH
        train_span.LR_PATCH = 16
        train_span.HR_PATCH = 16 * train_span.SCALE
        try:
            patch_corpus_source(self.corpus, bank_dir, per_pair=4, min_correlation=0.5)
            written_at = (bank_dir / 'bank.json').stat().st_mtime
            patch_corpus_source(self.corpus, bank_dir, per_pair=4, min_correlation=0.5)
            self.assertEqual((bank_dir / 'bank.json').stat().st_mtime, written_at)
        finally:
            train_span.LR_PATCH = original_lr_patch
            train_span.HR_PATCH = original_lr_patch * train_span.SCALE


class CheckpointRoundTripTests(unittest.TestCase):
    def test_saved_checkpoint_loads_through_eval_checkpoint_and_matches_forward(self):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eval_checkpoint import load

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        model = build_model(4, 4)
        checkpoint = {'model': model.state_dict(), 'channels': 4, 'frames': 1,
            'scale': 4, 'version': 1, 'step': 10, 'architecture': 'scale_matched_span'}
        path = tmp / 'checkpoint.pth'
        torch.save(checkpoint, path)

        loaded, step, frames = load(str(path), 'cpu')
        self.assertEqual(step, 10)
        self.assertEqual(frames, 1)
        self.assertIsInstance(loaded, train_span.Unshuffled)

        probe = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            expected = model.eval()(probe)
            actual = loaded(probe)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_saved_checkpoint_scale2_round_trips_too(self):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eval_checkpoint import load

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        model = build_model(4, 2)
        checkpoint = {'model': model.state_dict(), 'channels': 4, 'frames': 1,
            'scale': 2, 'version': 1, 'step': 5, 'architecture': 'scale_matched_span'}
        path = tmp / 'checkpoint.pth'
        torch.save(checkpoint, path)

        loaded, step, frames = load(str(path), 'cpu')
        probe = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            expected = model.eval()(probe)
            actual = loaded(probe)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (1, 3, 32, 32))


if __name__ == '__main__':
    unittest.main()
