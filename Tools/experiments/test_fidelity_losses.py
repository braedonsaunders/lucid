"""Check fidelity semantics, supervision alignment, and RGB downsample parity."""
import shutil
import subprocess
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aesop_fidelity import AutoencodedFidelity
from clean_lr_targets import differentiable_clean_lr, clean_lr_consistency
from train_presented_detail import reconstruction_objective
from reconstruction_loss import sobel_loss
from train_span import fft_loss


class FidelityLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_default_objective_is_bit_exact(self):
        output, reference, intended = [torch.rand(2, 3, 32, 32) for _ in range(3)]
        for mode in ('mixture', 'reference'):
            detail = reference if mode == 'reference' else intended
            expected = (F.l1_loss(output, intended) + .2 * sobel_loss(output, detail)
                        + .05 * fft_loss(output, detail) + .1 * F.l1_loss(output, reference))
            self.assertTrue(torch.equal(expected, reconstruction_objective(output, reference, intended, mode)))

    def test_aesop_frozen_teacher_and_detached_target(self):
        teacher = nn.Conv2d(3, 3, 1, bias=False)
        fidelity = AutoencodedFidelity(teacher).train()
        output = torch.rand(1, 3, 16, 20, requires_grad=True)
        reference = torch.rand_like(output, requires_grad=True)
        loss = fidelity(output, reference)
        self.assertTrue(torch.equal(loss, F.l1_loss(teacher(output), teacher(reference))))
        loss.backward()
        self.assertGreater(float(output.grad.abs().sum()), 0)
        self.assertIsNone(reference.grad)
        self.assertIsNone(teacher.weight.grad)
        self.assertFalse(teacher.training)

    def test_aesop_replaces_all_pixel_l1_once(self):
        output, reference = [torch.rand(1, 3, 16, 16) for _ in range(2)]
        calls = []
        def fidelity(x, y):
            calls.append(1)
            return x.new_tensor(2.)
        expected = 2.2 + .2 * sobel_loss(output, reference) + .05 * fft_loss(output, reference)
        self.assertTrue(torch.equal(expected, reconstruction_objective(output, reference, reference, 'reference', fidelity)))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            reconstruction_objective(output, reference, output, 'reference', fidelity)

    def test_bottleneck_comparison_rejected(self):
        with self.assertRaises(ValueError):
            AutoencodedFidelity(nn.AvgPool2d(4))(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))

    def test_clean_lr_identity_gradient_and_border(self):
        reference = torch.rand(1, 3, 48, 64, requires_grad=True)
        target = differentiable_clean_lr(reference)
        self.assertEqual(float(clean_lr_consistency(reference, target)), 0.)
        output = torch.rand_like(reference, requires_grad=True)
        loss = clean_lr_consistency(output, target)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0)
        bad_border = target.detach().clone()
        bad_border[..., :4, :] = 100
        self.assertEqual(float(clean_lr_consistency(reference, bad_border)), 0.)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required for full-chroma parity')
    def test_clean_lr_full_chroma_ffmpeg_interior(self):
        for shape in ((64, 96), (96, 64), (48, 48)):
            pixels = np.random.default_rng(43).integers(0, 256, (1, *shape, 3), dtype=np.uint8)
            h, w = shape
            command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{w}x{h}', '-i', 'pipe:0', '-frames:v', '1', '-vf',
                       f'scale={w//2}:{h//2}:flags=lanczos+full_chroma_inp', '-threads', '1',
                       '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
            expected = np.frombuffer(subprocess.check_output(command, input=pixels.tobytes()), np.uint8).reshape(1, h//2, w//2, 3)
            actual = differentiable_clean_lr(torch.from_numpy(pixels).permute(0, 3, 1, 2).float()/255).permute(0, 2, 3, 1).numpy()*255
            delta = abs(expected[:, 4:-4, 4:-4].astype(float) - actual[:, 4:-4, 4:-4])
            self.assertLessEqual(float(delta.max()), 1.001)
            self.assertLess(float(delta.mean()), .1)


if __name__ == '__main__':
    unittest.main()
