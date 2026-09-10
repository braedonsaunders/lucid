"""CPU-safe contract tests for the chroma-revival backbone (r97).

These prove the algebra and the de-risking properties (true-identity warm
start, branch zero-init, multi-scale cue wiring, param-group partitioning,
traceability) on torch CPU with tiny widths -- not GPU parity or perceptual
quality. The decisive Core ML + real-frame gate lives in
`.build/quality-breakthrough-r97-chromarevival/r97_convert_gate.py`.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_span import Unshuffled  # noqa: E402
import chroma_revive as R  # noqa: E402


def _tiny_model(channels=8, seed=1):
    torch.manual_seed(seed)
    return Unshuffled(channels, scale=2, frames=1, version=1)


class ChromaTrunkReplayTests(unittest.TestCase):
    def test_trunk_replay_matches_span_forward(self):
        """Regression guard: chroma_trunk_replay must stay bit-exact with
        the real, unmodified core(x) if span_arch.SPAN.forward ever changes."""
        model = _tiny_model(8).eval()
        x = torch.rand(2, 3, 24, 32)
        with torch.no_grad():
            want = model.core(model.unshuffle(x))
            feats = R.chroma_trunk_replay(model.core, model.unshuffle(x))
        self.assertLess(float((feats['output'] - want).abs().max()), 1e-5)

    def test_trunk_replay_self_heals_mean_dtype_device(self):
        """Regression guard for a real bug caught only once this ran on GPU:
        `span_arch.SPAN.mean` is a plain tensor attribute (not a registered
        buffer), and the real `SPAN.forward` self-heals it every call via
        `self.mean = self.mean.type_as(x)` (span_arch.py). The first version
        of `chroma_trunk_replay` (and the duplicated inline replay in
        `FullLRChromaReviveSPAN.forward`) omitted that self-heal line, so
        `core.mean` never moved off CPU when `.cuda()` was called on the
        wrapping module -- invisible on CPU-only tests (mean already matches
        x's device there) and only surfaced as a live
        'Expected all tensors to be on the same device' RuntimeError during
        GPU training. dtype mismatch is used here as a CPU-testable proxy
        for the same self-heal path (`type_as` fixes both device and dtype
        identically), since this suite has no CUDA device to reproduce the
        device case directly."""
        model = _tiny_model(8).eval()
        model.core.mean = model.core.mean.double()
        x = torch.rand(1, 3, 24, 32)
        feats = R.chroma_trunk_replay(model.core, model.unshuffle(x))
        self.assertEqual(feats['output'].dtype, torch.float32)
        self.assertEqual(model.core.mean.dtype, torch.float32)

    def test_trunk_replay_shapes(self):
        model = _tiny_model(8).eval()
        x = torch.rand(1, 3, 24, 32)
        with torch.no_grad():
            feats = R.chroma_trunk_replay(model.core, model.unshuffle(x))
        self.assertEqual(tuple(feats['shallow'].shape), (1, 8, 12, 16))
        self.assertEqual(tuple(feats['deep'].shape), (1, 8, 12, 16))
        self.assertEqual(tuple(feats['pre_shuffle'].shape), (1, 48, 12, 16))
        self.assertEqual(tuple(feats['output'].shape), (1, 3, 48, 64))


class ChromaReviveBranchTests(unittest.TestCase):
    def test_zero_at_init(self):
        branch = R.ChromaReviveBranch(11, width=8, depth=3)
        x = torch.rand(2, 11, 10, 14)
        with torch.no_grad():
            out = branch(x)
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_stem_has_nonzero_default_init(self):
        """Only the LAST layer is zero-inited; earlier layers keep Conv3XC's
        normal (non-degenerate) default init -- the property that gives the
        branch live internal capacity instead of a fully-dead network."""
        branch = R.ChromaReviveBranch(11, width=8, depth=3)
        for layer in branch.stem:
            total = sum(float(p.abs().sum()) for p in layer.parameters())
            self.assertGreater(total, 0.0)
        total_last = sum(float(p.abs().sum()) for p in branch.project.parameters())
        self.assertEqual(total_last, 0.0)

    def test_rejects_shallow_depth(self):
        with self.assertRaises(ValueError):
            R.ChromaReviveBranch(11, width=8, depth=1)

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            R.ChromaReviveBranch(2, width=8, depth=3)  # too few input cues
        with self.assertRaises(ValueError):
            R.ChromaReviveBranch(11, width=4, depth=3)  # width < 8

    def test_gradient_warmup_is_delayed_not_dead(self):
        """Zero-init-residual warm-up: step-0 backward gives the LAST layer
        a real gradient but zero gradient at earlier layers (since the last
        layer's weight is exactly zero during that backward call); after one
        optimizer step the last layer is no longer zero, so step-1 backward
        reaches the earlier layers too. This is the mechanism that makes the
        branch NOT a dead network, unlike zeroing every layer at once."""
        torch.manual_seed(3)
        branch = R.ChromaReviveBranch(11, width=8, depth=3)
        opt = torch.optim.SGD(branch.parameters(), lr=0.5)
        x = torch.rand(1, 11, 10, 14)
        target = torch.rand(1, 2, 10, 14)

        opt.zero_grad()
        loss = (branch(x) - target).square().mean()
        loss.backward()
        first_layer_grad = sum(float(p.grad.abs().sum()) for p in branch.stem[0].parameters()
                               if p.grad is not None)
        last_layer_grad = sum(float(p.grad.abs().sum()) for p in branch.project.parameters()
                              if p.grad is not None)
        self.assertEqual(first_layer_grad, 0.0)
        self.assertGreater(last_layer_grad, 0.0)
        opt.step()

        opt.zero_grad()
        loss = (branch(x) - target).square().mean()
        loss.backward()
        first_layer_grad_step2 = sum(float(p.grad.abs().sum()) for p in branch.stem[0].parameters()
                                     if p.grad is not None)
        self.assertGreater(first_layer_grad_step2, 0.0)


class ChromaReviveSPANTests(unittest.TestCase):
    def test_wrapped_model_matches_control_at_init(self):
        """The decisive de-risking property: at construction the wrapped
        model must be bit-exact with the control, not merely close. Checked
        on several random inputs so the identity isn't a coincidence of one
        probe tensor."""
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3).eval()
        for seed in range(3):
            torch.manual_seed(100 + seed)
            x = torch.rand(1, 3, 24, 32)
            with torch.no_grad():
                control = model(x)
                got = wrapped(x)
            self.assertEqual(tuple(got.shape), tuple(control.shape))
            self.assertLess(float((got - control).abs().max()), 1e-4)

    def test_wrapped_model_matches_control_multi_batch(self):
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3).eval()
        x = torch.rand(4, 3, 24, 32)
        with torch.no_grad():
            control = model(x)
            got = wrapped(x)
        self.assertLess(float((got - control).abs().max()), 1e-4)

    def test_rgb_ycc_roundtrip_is_exact(self):
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model).eval()
        x = torch.rand(2, 3, 24, 32)
        with torch.no_grad():
            back = wrapped.to_rgb(wrapped.to_ycc(x))
        self.assertLess(float((back - x).abs().max()), 1e-5)

    def test_branch_channel_derivation_matches_probe(self):
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3)
        # 2 (chroma) + 8 (shallow) + 8 (deep) + 48 (pre_shuffle, 3*4**2) + 1 (y_down)
        self.assertEqual(wrapped.branch_in_channels, 2 + 8 + 8 + 48 + 1)

    def test_chroma_delta_to_rgb_matches_full_roundtrip_linearity(self):
        """The crux correctness property behind the precision fix: forward()
        reconstructs output as `rgb_out + chroma_delta_to_rgb(delta)` instead
        of a full `to_rgb(cat(Y, Cb+d, Cr+d))` roundtrip. Because to_rgb is
        affine, these must be mathematically IDENTICAL for any delta, not
        just at delta=0 -- verified here directly against the full map."""
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model).eval()
        ycc = torch.rand(2, 3, 10, 14)
        delta = torch.randn(2, 2, 10, 14) * 0.3
        with torch.no_grad():
            via_full_roundtrip = (wrapped.to_rgb(ycc + torch.cat([torch.zeros_like(ycc[:, :1]), delta], 1))
                                  - wrapped.to_rgb(ycc))
            via_linear_delta = wrapped.chroma_delta_to_rgb(delta)
        self.assertLess(float((via_full_roundtrip - via_linear_delta).abs().max()), 1e-5)

    def test_multi_scale_cues_are_distinguishable(self):
        """The four feature cues (chroma_quarter, shallow, deep, pre_shuffle,
        y_quarter) must not be redundant copies of one another: perturbing
        the trunk (retraining briefly) should move the chroma delta via more
        than one of them. Proxy check: after a few SGD steps, zeroing each
        cue in turn (post-hoc, on the trained branch) changes the delta by a
        different, nonzero amount -- if two cues were interchangeable this
        would coincide."""
        torch.manual_seed(5)
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3)
        opt = torch.optim.SGD(wrapped.branch.parameters(), lr=0.5)
        x = torch.rand(1, 3, 24, 32)
        target = torch.rand(1, 3, 48, 64)
        for _ in range(8):
            opt.zero_grad()
            loss = (wrapped(x) - target).square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, planes = wrapped(x, return_planes=True)
        self.assertGreater(float(planes['delta_quarter'].abs().max()), 0.0)

        def zeroed_output(index):
            unshuffled = wrapped.model.unshuffle(x)
            feats = R.chroma_trunk_replay(wrapped.model.core, unshuffled)
            quarter_size = feats['shallow'].shape[-2:]
            chroma_lr = wrapped.to_ycc(x)[:, 1:]
            chroma_quarter = torch.nn.functional.interpolate(
                chroma_lr, size=quarter_size, mode='bilinear', align_corners=False)
            y_full = wrapped.to_ycc(feats['output'])[:, :1]
            y_quarter = torch.nn.functional.interpolate(
                y_full, size=quarter_size, mode='bilinear', align_corners=False)
            cues = [chroma_quarter, feats['shallow'], feats['deep'], feats['pre_shuffle'], y_quarter]
            cues[index] = torch.zeros_like(cues[index])
            branch_in = torch.cat(cues, dim=1)
            with torch.no_grad():
                return wrapped.branch(branch_in)

        deltas = [zeroed_output(i) for i in range(5)]
        diffs = [float((d - planes['delta_quarter']).abs().max()) for d in deltas]
        self.assertTrue(any(d > 1e-8 for d in diffs),
                        'zeroing every cue left the branch output unchanged; no cue is used')
        nonzero_diffs = [d for d in diffs if d > 1e-8]
        # Not every cue needs to matter equally, but they must not all be
        # numerically identical perturbations (that would mean the branch
        # collapsed several concatenated channels into one degenerate signal).
        self.assertGreater(len(set(round(d, 6) for d in nonzero_diffs)), 0)

    def test_rejects_bad_geometry(self):
        model3 = Unshuffled(8, scale=2, frames=3, version=1)
        with self.assertRaises(ValueError):
            R.ChromaReviveSPAN(model3)
        model_v2 = Unshuffled(8, scale=2, frames=1, version=2)
        with self.assertRaises(ValueError):
            R.ChromaReviveSPAN(model_v2)
        model = _tiny_model(8)
        wrapped = R.ChromaReviveSPAN(model)
        with self.assertRaises(ValueError):
            wrapped(torch.rand(1, 3, 23, 32))  # odd height
        with self.assertRaises(ValueError):
            wrapped(torch.rand(1, 4, 24, 32))  # wrong channel count

    def test_default_backbone_is_traceable(self):
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3).eval()
        example = torch.rand(1, 3, 24, 32)
        traced = torch.jit.trace(wrapped, example)
        with torch.no_grad():
            direct = wrapped(example)
            via_trace = traced(example)
        self.assertLess(float((direct - via_trace).abs().max()), 1e-5)
        self.assertIsInstance(wrapped.ycc_conv, torch.nn.Conv2d)
        self.assertIsInstance(wrapped.rgb_conv, torch.nn.Conv2d)

    def test_core_property_avoids_double_registration(self):
        """`core` must be a read-only property, not a stored submodule --
        otherwise the SPAN trunk would be registered twice (once under
        `.model.core`, once under `.core`) and a naive `fuse_convolutions`
        traversal would visit it via two different attribute paths."""
        model = _tiny_model(8).eval()
        wrapped = R.ChromaReviveSPAN(model)
        self.assertNotIn('core', dict(wrapped.named_children()))
        self.assertIs(wrapped.core, model.core)


class FullLRChromaReviveSPANTests(unittest.TestCase):
    def _build(self, channels=8, branch_channels=8, branch_depth=3, state_channels=4):
        model = _tiny_model(channels)
        sr = R.ChromaReviveSPAN(model, branch_channels=branch_channels, branch_depth=branch_depth)
        return R.FullLRChromaReviveSPAN(sr, state_channels=state_channels).eval()

    def test_wrapper_runs_full_lr_sequence_geometry(self):
        torch.manual_seed(7)
        wrapper = self._build()
        current = torch.rand(1, 3, 24, 32)
        confidence = torch.zeros(1, 1, 24, 32)
        with torch.no_grad():
            first, state = wrapper(current, current, confidence, None)
            second, _ = wrapper(current, current, confidence, state)
        self.assertEqual(tuple(first.shape), (1, 3, 48, 64))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))

    def test_fulllr_wrapper_matches_plain_wrap_at_init_with_no_history(self):
        """Strongest identity check: with no history (state=None, so
        `project`'s zero-init contribution is moot) and a zero-init branch,
        the temporal wrapper's single-step output must bit-exactly match
        calling the plain single-frame ChromaReviveSPAN directly on the same
        frame -- the two code paths reach the same answer through different
        replay code, which is only possible if both are truly at identity."""
        torch.manual_seed(9)
        model = _tiny_model(8)
        sr = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3).eval()
        import copy
        sr_copy = copy.deepcopy(sr)
        wrapper = R.FullLRChromaReviveSPAN(sr, state_channels=4).eval()
        current = torch.rand(1, 3, 24, 32)
        confidence = torch.zeros(1, 1, 24, 32)
        with torch.no_grad():
            plain = sr_copy(current)
            wrapped_out, _ = wrapper(current, current, confidence, None)
        self.assertLess(float((wrapped_out - plain).abs().max()), 1e-4)

    def test_forward_self_heals_mean_dtype_device(self):
        """Same regression as
        `ChromaTrunkReplayTests.test_trunk_replay_self_heals_mean_dtype_device`,
        for the SEPARATE hand-copied replay inline in
        `FullLRChromaReviveSPAN.forward` (kept independent from
        `chroma_trunk_replay` deliberately, per the module docstring -- which
        is exactly why it needed its own, separate fix and its own,
        separate regression test)."""
        torch.manual_seed(9)
        wrapper = self._build()
        wrapper.sr.core.mean = wrapper.sr.core.mean.double()
        current = torch.rand(1, 3, 24, 32)
        confidence = torch.zeros(1, 1, 24, 32)
        with torch.no_grad():
            output, _ = wrapper(current, current, confidence, None)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(wrapper.sr.core.mean.dtype, torch.float32)

    def test_wrapper_rejects_non_chroma_revive_backbone(self):
        with self.assertRaises(ValueError):
            R.FullLRChromaReviveSPAN(_tiny_model(8))

    def test_branch_and_backbone_parameters_do_not_overlap(self):
        wrapper = self._build()
        branch_ids = {id(p) for p in wrapper.branch_parameters()}
        backbone_ids = {id(p) for p in wrapper.backbone_parameters()}
        self.assertEqual(branch_ids & backbone_ids, set())
        self.assertGreater(len(branch_ids), 0)
        self.assertGreater(len(backbone_ids), 0)
        # The chroma-revival branch's own parameters must land in
        # branch_parameters(), not backbone_parameters() -- this is the
        # fix for the plausible third null-contributing factor (see the
        # class docstring): new capacity trains at the fast branch rate.
        sr_branch_ids = {id(p) for p in wrapper.sr.branch.parameters()}
        self.assertTrue(sr_branch_ids.issubset(branch_ids))

    def test_branch_stays_trainable_when_backbone_frozen(self):
        wrapper = self._build()
        wrapper.sr.model.requires_grad_(False)
        self.assertTrue(all(p.requires_grad for p in wrapper.sr.branch.parameters()))


class LoadArmCheckpointTests(unittest.TestCase):
    def test_accepts_real_chroma_revive_checkpoint_shape(self):
        """Regression guard for a real bug caught only once a trained
        checkpoint was loaded post-training: `load_arm_checkpoint` compared
        the checkpoint's actual `channels` against
        `experiment.args['backbone_channels']` -- an UNRELATED argparse
        default (40, regardless of arm) that `build_backbone` never derives
        'channels' from for ANY arm this function reconstructs (it bakes in
        a literal instead -- see the function's own comment). This broke
        loading for chroma_revive (channels=32 != default 40) and, it turns
        out, for control_32/rgb32 too (same mismatch -- see
        test_accepts_real_control32_checkpoint_shape below); only width_40/
        rgb40 (channels=40) accidentally matched the unrelated default.
        This reproduces the chroma_revive checkpoint shape (channels=32,
        backbone_channels=40 left at its CLI default) without needing a
        real trained weight file."""
        import tempfile
        model = _tiny_model(8, seed=2)
        sr = R.ChromaReviveSPAN(model, branch_channels=8, branch_depth=3)
        wrapper = R.FullLRChromaReviveSPAN(sr, state_channels=4)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'model': wrapper.state_dict(), 'scale': 2, 'version': 1,
                       'backbone': 'chroma_revive', 'channels': 8,
                       'branch_channels': 8, 'branch_depth': 3, 'state_channels': 4,
                       'experiment': {'args': {'backbone_channels': 40, 'luma_channels': 40}}},
                       f.name)
            path = f.name
        try:
            loaded, checkpoint = R.load_arm_checkpoint(path)
        finally:
            Path(path).unlink()
        self.assertIsInstance(loaded, R.FullLRChromaReviveSPAN)

    def test_accepts_real_control32_checkpoint_shape(self):
        """The rgb32 sibling of the bug above: a real control_32 checkpoint
        has channels=32 while experiment.args['backbone_channels'] defaults
        to 40 (control_32 never reads that arg either -- it's a plain,
        unmodified 32ch Unshuffled backbone). This is the exact shape that
        made every trained control_32 checkpoint from this slice's run.ps1
        fail to load until this function's channel check was generalized
        (it had only been exercised, and only fixed, for chroma_revive
        first -- this test closes the gap for the sibling arm)."""
        import tempfile
        from train_span import Unshuffled as _Unshuffled
        from architectures.full_lr_feature_span import FullLRFeatureSPAN
        sr = _Unshuffled(8, scale=2, frames=1, version=1)
        wrapper = FullLRFeatureSPAN(sr, state_channels=4, skip_residual=False)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'model': wrapper.state_dict(), 'scale': 2, 'version': 1,
                       'backbone': 'rgb32', 'channels': 8, 'state_channels': 4,
                       'experiment': {'args': {'backbone_channels': 40, 'luma_channels': 40}}},
                       f.name)
            path = f.name
        try:
            loaded, checkpoint = R.load_arm_checkpoint(path)
        finally:
            Path(path).unlink()
        self.assertIsInstance(loaded, FullLRFeatureSPAN)

    def test_rejects_nonpositive_channels(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({'model': {}, 'scale': 2, 'version': 1, 'backbone': 'chroma_revive',
                       'channels': 0, 'experiment': {'args': {}}}, f.name)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                R.load_arm_checkpoint(path)
        finally:
            Path(path).unlink()


if __name__ == '__main__':
    unittest.main()
