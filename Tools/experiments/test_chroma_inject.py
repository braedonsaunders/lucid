"""CPU-safe contract tests for the chroma-inject reallocation backbone.

These run on torch CPU with tiny widths; they prove the algebra (YCbCr
roundtrip, zero-init chroma identity, achromatic warm-start equivalence,
luma-injection gradient flow, geometry compatibility, MAC reallocation
ratio), not GPU parity or perceptual quality.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_span import Unshuffled  # noqa: E402
from widen_span import widen  # noqa: E402
import chroma_inject as C  # noqa: E402
import luma_asym as L  # noqa: E402


class ChromaInjectTests(unittest.TestCase):
    def test_rgb_to_ycbcr_roundtrip_is_exact(self):
        model = C.ChromaInject(8, scale=2)
        x = torch.rand(2, 3, 32, 40)
        with torch.no_grad():
            back = model.to_rgb(model.to_ycc(x))
        self.assertLess(float((back - x).abs().max()), 1e-5)

    def test_zero_init_chroma_matches_bicubic_design(self):
        """Fresh ChromaInject == same-trunk LumaAsymmetric on any input: the
        branch is a zero residual on the bicubic path. Both trunks share one
        seed, so different random inits cannot pollute the comparison."""
        torch.manual_seed(11)
        luma_w, chroma_w = 12, 8
        model = C.ChromaInject(luma_w, chroma_w, scale=2, chroma_mode='bicubic').eval()
        torch.manual_seed(11)
        ref = L.LumaAsymmetric(luma_w, scale=2, chroma_mode='bicubic').eval()
        for a, b in zip(model.core.parameters(), ref.core.parameters()):
            b.data.copy_(a.data)
        x = torch.rand(1, 3, 32, 40)
        with torch.no_grad():
            got = model(x)
            want = ref(x)
        self.assertEqual(tuple(got.shape), (1, 3, 64, 80))
        self.assertLess(float((got - want).abs().max()), 1e-5)

    def test_warm_start_reproduces_widened_luma_on_gray(self):
        # Uneven trunk widths fail in luma_asym._project_conv3xc (verified
        # 8->12: dst width 12 vs src width 24 at dim 0). Even widths that
        # widen_span supports are 8->16, so the warm-start contract is
        # exercised there.
        torch.manual_seed(12)
        wide = widen(Unshuffled(8, scale=2), 16)
        model = C.ChromaInject(16, 8, scale=2)
        C.warm_start_chroma_inject(model, wide)
        gray = torch.rand(2, 1, 32, 40).repeat(1, 3, 1, 1)
        weights = torch.tensor(L.LUMA_WEIGHTS).view(1, 3, 1, 1)
        with torch.no_grad():
            luma_wide = (wide.eval()(gray) * weights).sum(1, keepdim=True)
            luma_got = (model.eval()(gray) * weights).sum(1, keepdim=True)
        self.assertLess(float((luma_wide - luma_got).abs().max()), 1e-4)
        with torch.no_grad():
            full = model(gray)
        spread = full - full.mean(1, keepdim=True)
        self.assertLess(float(spread.abs().max()), 2e-2)

    def test_chroma_branch_learns_and_injects_luma(self):
        """The refinement must (a) move away from bicubic under training and
        (b) actually consume luma features: zeroing them changes the output."""
        torch.manual_seed(13)
        model = C.ChromaInject(8, 4, scale=2)
        for p in model.parameters():
            if p.requires_grad:
                pass
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.rand(1, 3, 32, 40)
        with torch.no_grad():
            before = model(x)
        target = torch.rand(1, 3, 64, 80)
        for _ in range(5):
            opt.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            after = model(x)
        self.assertGreater(float((after - before).abs().max()), 1e-6)
        self.assertTrue(bool(torch.isfinite(after).all()))
        # Injection check: luma features flow into chroma.
        with torch.no_grad():
            out, planes = model(x, return_planes=True)
            zero_feat = torch.zeros_like(
                torch.nn.functional.interpolate(
                    model.core.conv_1(model.unshuffle(model.to_ycc(x)[:, :1])),
                    size=planes['chroma_up'].shape[-2:], mode='bilinear',
                    align_corners=False))
            via_zero = model.chroma_from_luma_features(planes['chroma_up'], zero_feat)
        self.assertGreater(float((via_zero - planes['chroma_ref']).abs().max()), 0)

    def test_gradient_reaches_trunk_through_injection(self):
        torch.manual_seed(14)
        model = C.ChromaInject(8, 4, scale=2)
        x = torch.rand(1, 3, 32, 40)
        loss = model(x).square().mean()
        loss.backward()
        trunk_grads = [p.grad for p in model.core.conv_1.parameters()
                       if p.grad is not None]
        self.assertTrue(trunk_grads, 'no gradient reached trunk conv_1')
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in trunk_grads))

    def test_mac_ratio_is_reallocation_not_expansion(self):
        report = C.report_macs()
        self.assertLess(report['ratio_c_over_b'], 1.05)
        self.assertGreater(report['ratio_c_over_b'], 0.85)
        # The chroma branch itself must be small vs the trunk it augments.
        self.assertLess(report['chroma_branch'], 0.05 * report['luma_trunk'])

    def test_wrapper_runs_full_lr_sequence_geometry(self):
        torch.manual_seed(15)
        backbone = C.ChromaInject(12, 8, scale=2)
        model = C.FullLRChromaInjectSPAN(backbone, state_channels=4).eval()
        current = torch.rand(1, 3, 32, 40)
        confidence = torch.zeros(1, 1, 32, 40)
        with torch.no_grad():
            first, state = model(current, current, confidence, None)
            second, _ = model(current, current, confidence, state)
        self.assertEqual(tuple(first.shape), (1, 3, 64, 80))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))
        # Wrapper chroma path must differ from fixed bicubic once trained.
        with torch.no_grad():
            for p in model.sr.chroma_refine.parameters():
                if p.dim() == 4:
                    p.normal_(std=.01)
            trained, _ = model(current, current, confidence, None)
        self.assertGreater(float((trained - first).abs().max()), 0)

    def test_wrapper_rejects_rgb_backbone(self):
        with self.assertRaises(ValueError):
            C.FullLRChromaInjectSPAN(Unshuffled(8, scale=2))

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            C.ChromaInject(8, scale=4)
        with self.assertRaises(ValueError):
            C.ChromaInject(8, version=2)
        with self.assertRaises(ValueError):
            C.ChromaInject(8)(torch.rand(1, 3, 31, 40))
        with self.assertRaises(ValueError):
            C.ChromaInject(8).chroma_from_luma_features(
                torch.rand(1, 2, 8, 8), torch.rand(1, 8, 6, 6))

    def test_default_chroma_mode_is_bilinear_for_convertibility(self):
        """coremltools has no bicubic upsample; the deployable default must
        avoid it. Bicubic stays available only for research comparisons."""
        model = C.ChromaInject(8, 4, scale=2)
        self.assertEqual(model.chroma_mode, 'bilinear')
        with self.assertRaises(ValueError):
            C.ChromaInject(8, 4, scale=2, chroma_mode='nearest')

    def test_default_backbone_is_traceable(self):
        """torch.jit.trace must succeed with no data-dependent control flow
        or shape unpacking in the traced call (a coremltools precondition).
        This does not run coremltools (unavailable on this host); it proves
        the graph shape, not NE placement or numerics post-conversion."""
        model = C.ChromaInject(8, 4, scale=2).eval()
        example = torch.rand(1, 3, 32, 40)
        traced = torch.jit.trace(model, example)
        with torch.no_grad():
            direct = model(example)
            via_trace = traced(example)
        self.assertLess(float((direct - via_trace).abs().max()), 1e-5)
        # Trace-safety regression: to_ycc/to_rgb must be plain 1x1 convs, not
        # einsum+reshape (einsum on a runtime-derived shape has previously
        # broken tracing in this codebase's other color-conversion paths).
        self.assertIsInstance(model.ycc_conv, torch.nn.Conv2d)
        self.assertIsInstance(model.rgb_conv, torch.nn.Conv2d)

    def test_single_frame_output_geometry(self):
        model = C.ChromaInject(8, scale=2).eval()
        with torch.no_grad():
            out = model(torch.rand(1, 3, 32, 40))
        self.assertEqual(tuple(out.shape), (1, 3, 64, 80))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == '__main__':
    unittest.main()
