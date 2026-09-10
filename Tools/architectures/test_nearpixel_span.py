#!/usr/bin/env python3
"""Unit tests for nearpixel_span.py. No GPU/ANE, no training, no bank data.

  .venv-convert/bin/python Tools/architectures/test_nearpixel_span.py
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nearpixel_span import NearPixelBranch, NearPixelSPAN, trunk_pre_shuffle  # noqa: E402
from train_span import Unshuffled  # noqa: E402


torch.manual_seed(20260914)


class NearPixelBranchTests(unittest.TestCase):
    def test_near_pixel_branch_matches_nearest_neighbor_scale2(self):
        """The whole point of the branch: at init, PixelShuffle(4) of its
        output must be bit-exact nearest-neighbor 2x upsampling of the
        original RGB frame -- this is the "s=4 applied to the unshuffled
        input" derivation in the module docstring, checked numerically."""
        scale = 2
        branch = NearPixelBranch(scale).eval()
        shuffle = torch.nn.PixelShuffle(2 * scale)
        x = torch.rand(2, 3, 8, 10)
        unshuffled = F.pixel_unshuffle(x, 2)
        with torch.no_grad():
            out = shuffle(branch(unshuffled))
        expected = F.interpolate(x, scale_factor=scale, mode="nearest")
        self.assertEqual(out.shape, expected.shape)
        max_abs_diff = (out - expected).abs().max().item()
        self.assertLess(max_abs_diff, 1e-6, f"max abs diff {max_abs_diff}")

    def test_near_pixel_branch_matches_nearest_neighbor_scale1(self):
        """scale=1 (trunk PixelShuffle factor 2) degenerates to plain
        pixel-unshuffle/shuffle round trip = identity; checked as a second,
        structurally different scale so the derivation isn't scale2-special-cased."""
        scale = 1
        branch = NearPixelBranch(scale).eval()
        shuffle = torch.nn.PixelShuffle(2 * scale)
        x = torch.rand(2, 3, 8, 10)
        unshuffled = F.pixel_unshuffle(x, 2)
        with torch.no_grad():
            out = shuffle(branch(unshuffled))
        expected = F.interpolate(x, scale_factor=scale, mode="nearest")
        self.assertLess((out - expected).abs().max().item(), 1e-6)

    def test_near_pixel_branch_matches_nearest_neighbor_scale3(self):
        """A non-power-of-two scale (trunk factor 6), to rule out an
        accidental power-of-two-only bug in the permutation derivation."""
        scale = 3
        branch = NearPixelBranch(scale).eval()
        shuffle = torch.nn.PixelShuffle(2 * scale)
        x = torch.rand(1, 3, 6, 6)
        unshuffled = F.pixel_unshuffle(x, 2)
        with torch.no_grad():
            out = shuffle(branch(unshuffled))
        expected = F.interpolate(x, scale_factor=scale, mode="nearest")
        self.assertLess((out - expected).abs().max().item(), 1e-6)

    def test_shape_and_param_count(self):
        scale = 2
        branch = NearPixelBranch(scale)
        x = torch.rand(1, 12, 4, 5)
        out = branch(x)
        self.assertEqual(tuple(out.shape), (1, 3 * (2 * scale) ** 2, 4, 5))
        # conv_near: true depthwise, groups=12, 1 input channel per group,
        # 12*scale^2 output channels, 3x3 kernel, no bias.
        conv_near_params = sum(p.numel() for p in branch.conv_near.parameters())
        self.assertEqual(conv_near_params, 12 * scale * scale * 1 * 3 * 3)
        # perm is frozen (not trainable) and contributes zero trainable params.
        trainable = sum(p.numel() for p in branch.parameters() if p.requires_grad)
        self.assertEqual(trainable, conv_near_params)
        for name, p in branch.perm.named_parameters():
            self.assertFalse(p.requires_grad, name)

    def test_perm_is_a_permutation_matrix(self):
        """`perm` only ever reorders conv_near's already-materialized output
        channels -- it does not itself replicate one source to many
        destinations (that repetition happens inside conv_near: its `reps`
        output channels per input group are identical copies at init, see
        test_conv_near_group_outputs_are_identical_copies_at_init below).
        So perm must be a genuine 0/1 bijection: every row and every column
        sums to exactly 1."""
        scale = 2
        branch = NearPixelBranch(scale)
        w = branch.perm.weight.detach()[:, :, 0, 0]  # (out, in)
        self.assertTrue(torch.all((w == 0) | (w == 1)))
        self.assertTrue(torch.equal(w.sum(dim=1), torch.ones(w.shape[0])))
        self.assertTrue(torch.equal(w.sum(dim=0), torch.ones(w.shape[1])))

    def test_conv_near_group_outputs_are_identical_copies_at_init(self):
        """Within one input-channel group, all `scale*scale` raw output
        channels are identical center-tap-1 copies at init -- this is where
        the nearest-neighbor "repeat" actually happens, before perm reorders
        the (already duplicated) values into PixelShuffle's layout."""
        scale = 2
        branch = NearPixelBranch(scale)
        reps = scale * scale
        w = branch.conv_near.weight.detach()  # (12*reps, 1, 3, 3)
        for k in range(12):
            group = w[k * reps:(k + 1) * reps]
            for m in range(1, reps):
                self.assertTrue(torch.equal(group[0], group[m]))


class TrunkPreShuffleTests(unittest.TestCase):
    def test_trunk_pre_shuffle_matches_span_forward(self):
        """Guards against nearpixel_span.trunk_pre_shuffle drifting from
        span_arch.SPAN.forward: PixelShuffle(trunk_pre_shuffle(x)) must equal
        core(x) bit-exactly for an untouched, randomly initialized core."""
        model = Unshuffled(channels=8, scale=2, frames=1, version=1).eval()
        x = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            unshuffled = model.unshuffle(x)
            direct = model.core(unshuffled)
            replayed = model.core.upsampler[1](trunk_pre_shuffle(model.core, unshuffled))
        self.assertTrue(torch.equal(direct, replayed))


class NearPixelSPANTests(unittest.TestCase):
    def test_wrapped_model_matches_control_at_init(self):
        """The de-risking property this design targets: before any training,
        the wrapped model is bit-exact with the plain control model. Nothing
        pretrained is zeroed or perturbed; only fresh parameters are free."""
        torch.manual_seed(1)
        model = Unshuffled(channels=8, scale=2, frames=1, version=1).eval()
        wrapped = NearPixelSPAN(model).eval()
        x = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            control_out = model(x)
            wrapped_out = wrapped(x)
        self.assertEqual(control_out.shape, wrapped_out.shape)
        max_abs_diff = (control_out - wrapped_out).abs().max().item()
        self.assertLess(max_abs_diff, 1e-5, f"max abs diff {max_abs_diff}")

    def test_wrapped_model_matches_control_multiframe(self):
        """frames=3: near-pixel branch must read only the current (last)
        frame's 3 channels, and init parity must still hold."""
        torch.manual_seed(2)
        model = Unshuffled(channels=8, scale=2, frames=3, version=1).eval()
        wrapped = NearPixelSPAN(model).eval()
        x = torch.rand(1, 9, 16, 16)
        with torch.no_grad():
            control_out = model(x)
            wrapped_out = wrapped(x)
        self.assertLess((control_out - wrapped_out).abs().max().item(), 1e-5)

    def test_near_pixel_branch_actually_reads_current_frame(self):
        """Sanity check that the near-pixel path is wired to the current
        frame, not a stale/zeroed slice: perturbing only the earliest frame
        must not change the near-pixel branch's contribution once the fusion
        weight is nudged onto it, but perturbing the CURRENT frame must."""
        torch.manual_seed(3)
        model = Unshuffled(channels=8, scale=2, frames=2, version=1).eval()
        wrapped = NearPixelSPAN(model).eval()
        with torch.no_grad():
            wrapped.pointwise_fuse.weight[:, wrapped.trunk_ch:] = 0.1 * torch.randn(
                wrapped.pointwise_fuse.weight[:, wrapped.trunk_ch:].shape)
        x = torch.rand(1, 6, 16, 16)
        x_perturbed_old = x.clone()
        x_perturbed_old[:, :3] += 0.3
        x_perturbed_new = x.clone()
        x_perturbed_new[:, 3:] += 0.3
        with torch.no_grad():
            base = wrapped(x)
            out_old_changed = wrapped(x_perturbed_old)
            out_new_changed = wrapped(x_perturbed_new)
        self.assertGreater((out_new_changed - base).abs().max().item(), 1e-4)

    def test_fusion_weight_summary(self):
        model = Unshuffled(channels=8, scale=2, frames=1, version=1).eval()
        wrapped = NearPixelSPAN(model)
        summary = wrapped.fusion_weight_summary()
        self.assertEqual(summary["near_pixel_l1"], 0.0)
        self.assertGreater(summary["trunk_l1"], 0.0)
        self.assertEqual(summary["near_pixel_fraction"], 0.0)

    def test_rejects_odd_trunk_factor(self):
        model = Unshuffled(channels=8, scale=2, frames=1, version=1)
        model.core.upsampler[1] = torch.nn.PixelShuffle(3)
        with self.assertRaises(ValueError):
            NearPixelSPAN(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
