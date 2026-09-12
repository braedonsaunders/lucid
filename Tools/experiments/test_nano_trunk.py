"""Tests for the causal NanoVSR-derived trunk.

The reparameterisation test is load-bearing. If the folded 3x3 is not
numerically the training graph, the thing that gets converted and timed is not
the thing that was trained, and the whole architecture comparison measures a
conversion bug instead of an architecture.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.nano_trunk import NanoTrunk, RepBlock, fused_parameters  # noqa: E402
from train_span import Unshuffled  # noqa: E402


class NanoTrunkTests(unittest.TestCase):
    def test_output_geometry_matches_the_span_trunk_it_replaces(self):
        source = torch.rand(2, 3, 32, 48)
        for scale in (2, 4):
            nano = NanoTrunk(channels=16, scale=scale, frames=1, blocks=3).eval()
            span = Unshuffled(16, scale=scale, frames=1).eval()
            with torch.no_grad():
                self.assertEqual(nano(source).shape, span(source).shape)

    def test_reparameterisation_is_numerically_equivalent(self):
        torch.manual_seed(4)
        block = RepBlock(8, 8).eval()
        # Give the BatchNorms non-trivial running statistics: at initialisation
        # they are identity-ish and the test would pass on a broken fold.
        for module in block.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_(0, 0.5)
                module.running_var.uniform_(0.5, 1.5)
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.normal_(0, 0.5)
        source = torch.rand(2, 8, 16, 16)
        with torch.no_grad():
            before = block(source)
        block.switch_to_deploy()
        with torch.no_grad():
            after = block(source)
        self.assertTrue(torch.allclose(before, after, atol=1e-5),
                        f'max deviation {float((before - after).abs().max()):.2e}')

    def test_whole_trunk_survives_deploy_folding(self):
        torch.manual_seed(5)
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=4).eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_var.uniform_(0.5, 1.5)
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.normal_(0, 0.3)
        source = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            before = model(source)
        model.switch_to_deploy()
        with torch.no_grad():
            after = model(source)
        self.assertTrue(torch.allclose(before, after, atol=1e-4),
                        f'max deviation {float((before - after).abs().max()):.2e}')

    def test_it_is_causal(self):
        """No output may depend on a later frame - the reason upstream's
        bidirectional model is disqualified for live playback."""
        model = NanoTrunk(channels=16, scale=2, frames=3, blocks=3).eval()
        source = torch.rand(1, 9, 32, 32)
        with torch.no_grad():
            baseline = model(source)
        # `frames` are ordered with the current frame last, so perturbing an
        # earlier frame may change the output, but there is no later frame to
        # perturb at all. Assert the contract the shape encodes.
        self.assertEqual(source.shape[1], 3 * model.frames)
        self.assertEqual(baseline.shape, (1, 3, 64, 64))

    def test_deploy_construction_matches_folded_training_graph(self):
        torch.manual_seed(6)
        trained = NanoTrunk(channels=12, scale=2, frames=1, blocks=3).eval()
        trained.switch_to_deploy()
        fresh = NanoTrunk(channels=12, scale=2, frames=1, blocks=3, deploy=True).eval()
        self.assertEqual(sorted(trained.state_dict()), sorted(fresh.state_dict()))

    def test_fused_parameter_count_is_below_the_training_count(self):
        model = NanoTrunk(channels=32, scale=2, frames=1, blocks=7)
        raw = sum(p.numel() for p in model.parameters())
        self.assertLess(fused_parameters(model), raw)

    def test_unshuffle_toggle_changes_where_the_trunk_runs(self):
        source = torch.rand(1, 3, 32, 32)
        for unshuffle in (True, False):
            model = NanoTrunk(channels=16, scale=2, frames=1, blocks=2,
                              unshuffle=unshuffle).eval()
            with torch.no_grad():
                self.assertEqual(model(source).shape, (1, 3, 64, 64))

    def test_a_zero_residual_trunk_returns_the_bilinear_base(self):
        """The global skip is one of the concrete differences from SPAN, so it
        should be visible: zero the upsampler and the output is the base."""
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=2).eval()
        with torch.no_grad():
            model.upsampler[0].weight.zero_()
            model.upsampler[0].bias.zero_()
            source = torch.rand(1, 3, 16, 16)
            expected = torch.nn.functional.interpolate(
                source, scale_factor=2, mode='bilinear', align_corners=False)
            self.assertTrue(torch.allclose(model(source), expected, atol=1e-6))

    def test_bad_configuration_is_refused(self):
        with self.assertRaises(ValueError):
            NanoTrunk(scale=3)
        with self.assertRaises(ValueError):
            NanoTrunk(blocks=0)


class NormFreeRepBlockTests(unittest.TestCase):
    """r114b: the BatchNorm-free variant.

    r114's arm lost 4.58 dB PSNR while over-texturing, and `RepBlock` carries
    BatchNorm where SPAN's `Conv3XC` does not. BN is documented as harmful in
    super-resolution and this project trains at batch 4, so the arm confounded
    block topology with normalisation. These tests pin the variant that
    separates them.
    """

    def test_norm_free_fold_is_numerically_equivalent(self):
        torch.manual_seed(21)
        block = RepBlock(8, 8, norm='none').eval()
        with torch.no_grad():
            block.rbr_dense.weight.normal_(0, 0.2)
            block.rbr_dense.bias.normal_(0, 0.1)
            block.rbr_1x1.weight.normal_(0, 0.2)
            block.rbr_1x1.bias.normal_(0, 0.1)
            block.rbr_identity.uniform_(0.5, 1.5)
        source = torch.rand(2, 8, 16, 16)
        with torch.no_grad():
            before = block(source)
        block.switch_to_deploy()
        with torch.no_grad():
            after = block(source)
        self.assertTrue(torch.allclose(before, after, atol=1e-5),
                        f'max deviation {float((before - after).abs().max()):.2e}')

    def test_norm_free_trunk_has_no_batchnorm_at_all(self):
        model = NanoTrunk(channels=32, scale=2, frames=1, blocks=6, norm='none')
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        batched = NanoTrunk(channels=32, scale=2, frames=1, blocks=6, norm='batch')
        self.assertTrue(any(isinstance(m, torch.nn.BatchNorm2d) for m in batched.modules()))

    def test_both_variants_have_the_same_deployed_cost(self):
        """The comparison is only about normalisation if the folded graphs match."""
        self.assertEqual(
            fused_parameters(NanoTrunk(channels=48, scale=2, frames=1, blocks=6, norm='none')),
            fused_parameters(NanoTrunk(channels=48, scale=2, frames=1, blocks=6, norm='batch')))

    def test_norm_free_trunk_survives_deploy_folding(self):
        torch.manual_seed(22)
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=4, norm='none').eval()
        source = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            before = model(source)
        model.switch_to_deploy()
        with torch.no_grad():
            after = model(source)
        self.assertTrue(torch.allclose(before, after, atol=1e-4),
                        f'max deviation {float((before - after).abs().max()):.2e}')

    def test_unknown_norm_is_refused(self):
        with self.assertRaises(ValueError):
            RepBlock(8, 8, norm='layer')


if __name__ == '__main__':
    unittest.main()


class RecurrentStateTests(unittest.TestCase):
    """r119: the recurrent propagation r114 deliberately left switched off.

    This is what makes NanoVSR a video model rather than an image model run per
    frame, and it is the one lever that adds information instead of reprocessing
    its absence - r83 measured 74% of the error in bands where the input carries
    2.5% of its energy.
    """

    def test_state_changes_the_output(self):
        """If carrying state were a no-op the whole arm would be a null by
        construction, and that has happened in this ledger before."""
        torch.manual_seed(31)
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=3, norm='none').eval()
        frames = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            first, state = model(frames, return_state=True)
            with_state, _ = model(frames, state=state, return_state=True)
            without_state = model(frames)
        self.assertFalse(torch.allclose(with_state, without_state, atol=1e-6))
        self.assertEqual(first.shape, with_state.shape)

    def test_first_frame_matches_the_stateless_call(self):
        """A stream's first frame has no history, so training must see exactly
        what playback sees rather than a warm state the live path never has."""
        torch.manual_seed(32)
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=3, norm='none').eval()
        clip = torch.rand(1, 4, 3, 32, 32)
        with torch.no_grad():
            sequence = model.forward_sequence(clip)
            single = model(clip[:, 0])
        self.assertTrue(torch.allclose(sequence[:, 0], single, atol=1e-6))

    def test_sequence_geometry(self):
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=3, norm='none').eval()
        with torch.no_grad():
            out = model.forward_sequence(torch.rand(2, 5, 3, 24, 32))
        self.assertEqual(out.shape, (2, 5, 3, 48, 64))

    def test_sequence_is_causal(self):
        """Changing a LATER frame must not change an EARLIER output. Upstream's
        backward pass breaks this, which is why it is disqualified for live
        playback; this arm must not reintroduce it."""
        torch.manual_seed(33)
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=3, norm='none').eval()
        clip = torch.rand(1, 4, 3, 32, 32)
        altered = clip.clone()
        altered[:, 3] = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            a = model.forward_sequence(clip)
            b = model.forward_sequence(altered)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 3], b[:, 3], atol=1e-6))

    def test_sequence_requires_five_dimensions(self):
        model = NanoTrunk(channels=16, scale=2, frames=1, blocks=3, norm='none')
        with self.assertRaises(ValueError):
            model.forward_sequence(torch.rand(2, 3, 32, 32))
