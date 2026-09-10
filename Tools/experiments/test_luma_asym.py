"""CPU-safe contract tests for the luma-dominant asymmetric backbone.

These run on torch CPU with tiny widths; they prove the algebra (exact
front-end folding, achromatic warm-start equivalence, YCbCr roundtrip, RGB
geometry compatibility), not GPU parity or perceptual quality.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_span import Unshuffled  # noqa: E402
from widen_span import widen  # noqa: E402
import luma_asym as L  # noqa: E402


class LumaAsymTests(unittest.TestCase):
    def test_rgb_to_ycbcr_roundtrip_is_exact(self):
        model = L.LumaAsymmetric(8, scale=2)
        x = torch.rand(2, 3, 32, 40)
        with torch.no_grad():
            back = model.to_rgb(model.to_ycc(x))
        self.assertLess(float((back - x).abs().max()), 1e-5)

    @staticmethod
    def _fused_first(model):
        model.core.conv_1.update_params()
        return model.core.conv_1.eval_conv.weight.data, model.core.conv_1.eval_conv.bias.data

    def test_front_end_folds_exactly_into_first_conv(self):
        torch.manual_seed(3)
        model = L.LumaAsymmetric(8, scale=2).eval()
        x = torch.rand(1, 3, 32, 40)
        weight, bias = self._fused_first(model)
        with torch.no_grad():
            luma = model.to_ycc(x)[:, :1]
            direct = model.core.conv_1(model.unshuffle(luma))
            folded = L.folded_rgb_first_weight(weight, L.RGB2YCC[0])
            via_folded = torch.nn.functional.conv2d(
                model.unshuffle(x), folded, bias, padding=1)
        self.assertEqual(tuple(folded.shape), (8, 12, 3, 3))
        self.assertLess(float((direct - via_folded).abs().max()), 1e-5)

    def test_folded_weight_matches_conv_on_multi_frame_input(self):
        torch.manual_seed(4)
        model = L.LumaAsymmetric(8, frames=2, scale=2).eval()
        x = torch.rand(1, 6, 32, 40)
        weight, bias = self._fused_first(model)
        with torch.no_grad():
            ycc = model.to_ycc(x)
            luma = torch.cat([ycc[:, 0:1], ycc[:, 3:4]], 1)
            direct = model.core.conv_1(model.unshuffle(luma))
            folded = L.folded_rgb_first_weight(weight, L.RGB2YCC[0])
            via_folded = torch.nn.functional.conv2d(
                model.unshuffle(x), folded, bias, padding=1)
        self.assertEqual(tuple(folded.shape), (8, 24, 3, 3))
        self.assertLess(float((direct - via_folded).abs().max()), 1e-5)

    def test_warm_start_reproduces_widened_luma_on_gray(self):
        torch.manual_seed(5)
        wide = widen(Unshuffled(8, scale=2), 12)
        model = L.LumaAsymmetric(12, scale=2)
        L.warm_start_from_widened(model, wide)
        gray = torch.rand(2, 1, 32, 40).repeat(1, 3, 1, 1)
        weights = torch.tensor(L.LUMA_WEIGHTS).view(1, 3, 1, 1)
        with torch.no_grad():
            luma_wide = (wide.eval()(gray) * weights).sum(1, keepdim=True)
            luma_asym = (model.eval()(gray) * weights).sum(1, keepdim=True)
            back_wide = wide(gray)
        self.assertLess(float((luma_wide - luma_asym).abs().max()), 1e-4)
        # Warm start must actually move away from random init, not just run.
        fresh = L.LumaAsymmetric(12, scale=2).eval()
        with torch.no_grad():
            luma_fresh = (fresh(gray) * weights).sum(1, keepdim=True)
        self.assertGreater(float((luma_fresh - luma_wide).abs().max()),
                           float((luma_asym - luma_wide).abs().max()))
        # Achromatic input through the warm-started backbone stays achromatic:
        # gray in, gray out (what the chroma path preserves). That is the
        # backbone invariant the bicubic design promises, checked directly.
        with torch.no_grad():
            full = model(gray)
        spread = full - full.mean(1, keepdim=True)
        self.assertLess(float(spread.abs().max()), 2e-2)

    def test_warm_start_rejects_width_mismatch(self):
        with self.assertRaises(ValueError):
            L.warm_start_from_widened(L.LumaAsymmetric(12, scale=2),
                                      widen(Unshuffled(8, scale=2), 16))

    def test_wrapper_runs_full_lr_sequence_geometry(self):
        torch.manual_seed(6)
        backbone = L.LumaAsymmetric(12, scale=2)
        model = L.FullLRLumaSPAN(backbone, state_channels=4).eval()
        with torch.no_grad():
            model.project.weight.normal_(std=.01)
        current = torch.rand(1, 3, 32, 40)
        confidence = torch.zeros(1, 1, 32, 40)
        with torch.no_grad():
            first, state = model(current, current, confidence, None)
            second, _ = model(current, current, confidence, state)
        self.assertEqual(tuple(first.shape), (1, 3, 64, 80))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))
        # Carried state must actually flow: with nonzero confidence, a zeroed
        # state changes the output (zero confidence would mask it by design).
        lived = torch.ones(1, 1, 32, 40)
        with torch.no_grad():
            _, live_state = model(current, current, confidence, None)
            via_live, _ = model(current, current, lived, live_state)
            via_zero, _ = model(current, current, lived, torch.zeros_like(live_state))
        self.assertGreater(float((via_live - via_zero).abs().max()), 0)

    def test_wrapper_rejects_rgb_backbone(self):
        with self.assertRaises(ValueError):
            L.FullLRLumaSPAN(Unshuffled(8, scale=2))

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            L.LumaAsymmetric(8, scale=4)
        with self.assertRaises(ValueError):
            L.LumaAsymmetric(8, scale=2, version=2)
        with self.assertRaises(ValueError):
            L.LumaAsymmetric(8, scale=2, chroma_mode='nearest')
        with self.assertRaises(ValueError):
            L.LumaAsymmetric(8, scale=2)(torch.rand(1, 3, 31, 40))

    def test_bilinear_chroma_matches_bicubic_closely(self):
        # Smooth natural-like input: bicubic ringing on white noise would
        # dominate the comparison and prove nothing about real frames.
        torch.manual_seed(8)
        low = torch.rand(1, 3, 8, 10)
        x = torch.nn.functional.interpolate(low, size=(32, 40), mode='bilinear',
                                            align_corners=False)
        # One construction, mode flipped between forwards: same trunk weights,
        # so the gap isolates the interpolator instead of init noise.
        model = L.LumaAsymmetric(8, scale=2).eval()
        with torch.no_grad():
            model.chroma_mode = 'bicubic'
            bic = model(x)
            model.chroma_mode = 'bilinear'
            bil = model(x)
        # Same trunk, same color maps: only the chroma interpolator differs,
        # so outputs agree to interpolation tolerance, not at random distance.
        gap = float((bic - bil).abs().max())
        spread = float((bic - bic.mean()).abs().max())
        self.assertLess(gap, 0.05 * spread + 1e-3)

    def test_single_frame_output_geometry(self):
        model = L.LumaAsymmetric(8, scale=2).eval()
        with torch.no_grad():
            out = model(torch.rand(1, 3, 32, 40))
        self.assertEqual(tuple(out.shape), (1, 3, 64, 80))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == '__main__':
    unittest.main()
