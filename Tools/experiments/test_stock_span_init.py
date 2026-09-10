"""Focused tests for the explicit stock SPAN 2x loader.

CPU only. No GPU, no network, no training. The zero-shot assertion ties the
loader to the pinned r64 eager fingerprints: stock weights must reproduce
their known behaviour through this loader, and a random anchor must be sane.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import stock_span_init as stock_init  # noqa: E402
from architectures.span_arch import SPAN  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
STOCK = os.path.join(REPO, "Model", "SPAN", "weights", "spanx2_ch48.pth")
FRAMES = os.path.join(REPO, ".build", "quality-breakthrough-r64-realframe", "frames")

# Pinned r64 eager fingerprints (verdict.json): stock raw output through the
# original-architecture forward on 640x360 real frames.
EXPECTED = {
    "crowdrun_t4.png": {"min": -0.05652942880988121, "max": 1.0407981872558594,
                        "frac01": 0.9993200231481482},
    "intotree_t1.png": {"min": -0.02773565985262394, "max": 0.9628430008888245,
                        "frac01": 0.9998911313657407},
}


class StockInitTests(unittest.TestCase):
    def test_loader_pins_hash_params_and_geometry(self):
        model, report = stock_init.load_stock_span2x(STOCK)
        self.assertEqual(report["checkpoint_sha256"], stock_init.STOCK_SHA256)
        self.assertEqual(report["parameters"], 2221140)
        self.assertEqual(report["trunk"], "full_lr")
        self.assertEqual(model.conv_1.eval_conv.in_channels, 3)
        self.assertEqual(model.upsampler[1].upscale_factor, 2)
        self.assertEqual(model.img_range, 255.0)
        self.assertFalse(hasattr(model, "unshuffle"))

    def test_stock_reproduces_pinned_zero_shot_behaviour(self):
        model, _ = stock_init.load_stock_span2x(STOCK)
        for name, pinned in EXPECTED.items():
            img = Image.open(os.path.join(FRAMES, name)).convert("RGB")
            example = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)[None].float() / 255
            with torch.inference_mode():
                raw = model(example)[0].permute(1, 2, 0).numpy()
            self.assertAlmostEqual(float(raw.min()), pinned["min"], places=5)
            self.assertAlmostEqual(float(raw.max()), pinned["max"], places=5)
            self.assertAlmostEqual(float(((raw >= 0) & (raw <= 1)).mean()), pinned["frac01"], places=5)

    def test_random_anchor_is_sane_and_mismatched_file_refused(self):
        anchor = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=2).eval()
        example = torch.rand(1, 3, 32, 32)
        with torch.inference_mode():
            out = anchor(example)
        self.assertEqual(tuple(out.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(out).all())
        with self.assertRaisesRegex(ValueError, "sha256"):
            stock_init.load_stock_span2x(__file__)


if __name__ == "__main__":
    unittest.main()
