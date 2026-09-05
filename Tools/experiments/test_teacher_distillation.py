import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
import torch

from train_causal_detail import batch, digest
from teacher_distillation import reference_checked_loss, load_teacher_cache


class TeacherDistillationTests(unittest.TestCase):
    def test_cache_rejects_other_bank_and_validation_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            lr = np.zeros((2, 8, 8, 3), dtype=np.uint8)
            hr = np.zeros((2, 16, 16, 3), dtype=np.uint8)
            data = {'train': [(lr, hr, 'train')], 'validation': [(lr, hr, 'held_out')]}
            np.savez_compressed(path / 'target.npz', teacher=hr)
            record = {'id': 'train', 'file': 'target.npz', 'sha256': digest(path / 'target.npz')}
            def manifest(bank='right', identity='train'):
                (path / 'manifest.json').write_text(json.dumps({'bank_sha256': bank,
                    'sequences': [{**record, 'id': identity}]}))
            manifest()
            loaded, _ = load_teacher_cache(path, 'right', data, digest)
            np.testing.assert_array_equal(loaded['train'], hr)
            manifest(bank='different')
            with self.assertRaisesRegex(ValueError, 'another bank'):
                load_teacher_cache(path, 'right', data, digest)
            manifest(identity='held_out')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                load_teacher_cache(path, 'right', data, digest)
            manifest()
            (path / 'target.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_teacher_cache(path, 'right', data, digest)

    def test_bad_teacher_is_rejected_without_suppressing_reference_objective(self):
        reference = torch.zeros(1, 3, 16, 16)
        output = torch.full_like(reference, .25, requires_grad=True)
        bad = torch.ones_like(reference)
        loss, confidence = reference_checked_loss(output, reference, bad, reference)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(float(confidence), 0)
        (loss + (output - reference).abs().mean()).backward()
        self.assertTrue(torch.all(output.grad > 0))

    def test_better_teacher_receives_gradient_only_through_student(self):
        reference = torch.zeros(1, 3, 16, 16, requires_grad=True)
        teacher = torch.zeros_like(reference, requires_grad=True)
        output = torch.full_like(reference, .25, requires_grad=True)
        floor = torch.ones_like(reference, requires_grad=True)
        loss, confidence = reference_checked_loss(output, reference, teacher, floor)
        loss.backward()
        self.assertGreater(float(confidence), .99)
        self.assertTrue(torch.all(output.grad > 0))
        for frozen in (reference, teacher, floor):
            self.assertIsNone(frozen.grad)

    def test_cached_teacher_follows_same_crop_flip_transpose_and_time(self):
        rng = np.random.default_rng(3)
        lr = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
        hr = lr.repeat(2, 1).repeat(2, 2)
        data = [(lr, hr, 'source')]
        x, y, teacher = batch(data, np.random.default_rng(17), 8, 7, 16, {'source': hr})
        old_x, old_y = batch(data, np.random.default_rng(17), 8, 7, 16)
        torch.testing.assert_close(x, old_x, rtol=0, atol=0)
        torch.testing.assert_close(y, old_y, rtol=0, atol=0)
        torch.testing.assert_close(teacher, y, rtol=0, atol=0)
        torch.testing.assert_close(teacher[:, :, :, ::2, ::2], x, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
