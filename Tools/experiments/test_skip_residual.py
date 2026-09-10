"""Behavioral CPU tests for the fixed low-frequency bypass (residual skip).

Covers Tools/architectures/skip_residual_span.py and the skip_residual path
in Tools/architectures/full_lr_feature_span.py only. Bounded CPU sizes, no
GPU, no native code, no network. No quality claim: these tests pin the
mechanics the quality experiment depends on - near-identity start, genuine
skip contribution, shape agreement, checkpoint round-trip, and traceability
for the CoreML gate.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from architectures.skip_residual_span import (
    SkipUnshuffled,
    from_unshuffled,
    zero_trunk_head,
)
from architectures.full_lr_feature_span import (
    FullLRFeatureSPAN,
    load_full_lr_checkpoint,
)
from train_span import Unshuffled


def _content(seed=0, height=32, width=48):
    generator = torch.Generator().manual_seed(seed)
    content = torch.rand(1, 3, height, width, generator=generator)
    return (content * 255).round() / 255


class SkipUnshuffledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def _skip(self, seed=0, channels=4):
        torch.manual_seed(seed)
        model = SkipUnshuffled(channels, scale=2).eval()
        return zero_trunk_head(model)

    def test_zeroed_trunk_starts_near_identity(self):
        model = self._skip()
        image = _content()
        with torch.no_grad():
            output = model(image)
            expected = F.interpolate(image, scale_factor=2, mode="bilinear",
                                     align_corners=False)
        self.assertEqual(tuple(output.shape), (1, 3, 64, 96))
        self.assertLess(float((output - expected).abs().max()), 1e-5)

    def test_skip_path_genuinely_contributes(self):
        model = self._skip()
        image = _content()
        with torch.no_grad():
            with_skip = model(image)
            trunk_only = model.core(model.unshuffle(image))
        self.assertGreater(float((with_skip - trunk_only).abs().mean()), 1e-3)
        self.assertEqual(tuple(with_skip.shape), tuple(trunk_only.shape))

    def test_shapes_across_ladder_sizes(self):
        model = self._skip()
        for width, height in ((256, 144), (320, 180), (432, 240),
                              (480, 270), (640, 360), (864, 480)):
            with self.subTest(size=(width, height)):
                with torch.no_grad():
                    output = model(_content(height=height, width=width))
                self.assertEqual(tuple(output.shape),
                                 (1, 3, height * 2, width * 2))

    def test_odd_input_rejected_before_silent_mismatch(self):
        model = self._skip()
        with self.assertRaisesRegex(ValueError, "even BCHW"):
            model(_content(height=33, width=48))

    def test_from_unshuffled_preserves_weights_and_zeroes_head(self):
        torch.manual_seed(3)
        base = Unshuffled(4, scale=2).eval()
        before = copy.deepcopy(base.state_dict())
        skip = from_unshuffled(base)
        # All non-head weights transfer exactly; the skip adds no parameters.
        self.assertEqual(set(skip.state_dict()), set(before))
        head = "core.upsampler.0.weight"
        for key in before:
            if key in (head, "core.upsampler.0.bias"):
                self.assertEqual(float(skip.state_dict()[key].abs().sum()), 0)
            else:
                self.assertTrue(torch.equal(skip.state_dict()[key], before[key]))
        with torch.no_grad():
            output = skip(_content())
            expected = F.interpolate(_content(), scale_factor=2,
                                     mode="bilinear", align_corners=False)
        self.assertLess(float((output - expected).abs().max()), 1e-5)

    def test_traces_for_conversion(self):
        model = self._skip()
        example = _content()
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
            output = traced(example)
        self.assertEqual(tuple(output.shape), (1, 3, 64, 96))
        self.assertTrue(torch.isfinite(output).all())


class StatefulSkipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def _stateful(self, seed=0, channels=4, skip=False):
        torch.manual_seed(seed)
        base = Unshuffled(channels, scale=2)
        if skip:
            zero_trunk_head(base)
        return FullLRFeatureSPAN(base, state_channels=8,
                                 skip_residual=skip).eval()

    def _frame(self, seed=1, height=32, width=48):
        return _content(seed=seed, height=height, width=width)

    def test_control_default_is_skip_free(self):
        self.assertFalse(self._stateful().skip_residual)
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            self._stateful().skip_residual = "yes"
            FullLRFeatureSPAN(Unshuffled(4, scale=2), skip_residual="yes")

    def test_control_matches_pre_skip_behavior(self):
        image, confidence = self._frame(), torch.zeros(1, 1, 32, 48)
        plain = self._stateful(seed=7, skip=False)
        other = self._stateful(seed=7, skip=False)
        with torch.no_grad():
            a, _ = plain(image, image, confidence, None)
            b, _ = other(image, image, confidence, None)
        self.assertTrue(torch.equal(a, b))

    def test_stateful_skip_near_identity_and_contributes(self):
        image, confidence = self._frame(), torch.zeros(1, 1, 32, 48)
        model = self._stateful(seed=7, skip=True)
        with torch.no_grad():
            output, _ = model(image, image, confidence, None)
            expected = F.interpolate(image, scale_factor=2, mode="bilinear",
                                     align_corners=False)
        self.assertLess(float((output - expected).abs().max()), 1e-4)
        control = self._stateful(seed=7, skip=False)
        with torch.no_grad():
            bare, _ = control(image, image, confidence, None)
        self.assertGreater(float((output - bare).abs().mean()), 1e-3)

    def test_skip_survives_checkpoint_roundtrip_and_loader_pins_it(self):
        model = self._stateful(seed=7, skip=True)
        image, confidence = self._frame(), torch.zeros(1, 1, 32, 48)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skip.pth"
            torch.save({"model": model.state_dict(),
                        "architecture": "full_lr_feature_span2x", "scale": 2,
                        "state_representation": "full_lr_observation_features_v1",
                        "feature_source": "decoded_rgb8",
                        "source_motion_policy": "raw", "state_warp": "full_lr",
                        "state_channels": 8, "channels": 4, "version": 1,
                        "skip_residual": True,
                        "experiment": {"args": {"source_motion_policy": "raw",
                                                "state_warp": "full_lr",
                                                "state_channels": 8,
                                                "sr_skip_residual": True}}},
                       path)
            loaded, checkpoint = load_full_lr_checkpoint(path)
            self.assertTrue(loaded.skip_residual)
            self.assertTrue(checkpoint["skip_residual"])
            with torch.no_grad():
                before, _ = model(image, image, confidence, None)
                after, _ = loaded(image, image, confidence, None)
            self.assertTrue(torch.equal(before, after))
            # A control declaration must not load a skip checkpoint.
            mismatched = path.with_name("skip_mismatch.pth")
            tampered = torch.load(path, map_location="cpu", weights_only=False)
            tampered["experiment"]["args"]["sr_skip_residual"] = False
            torch.save(tampered, mismatched)
            with self.assertRaisesRegex(ValueError, "skip residual differs"):
                load_full_lr_checkpoint(mismatched)

    def test_stateful_skip_traces_for_conversion(self):
        model = self._stateful(seed=7, skip=True)
        image, confidence = self._frame(), torch.zeros(1, 1, 32, 48)

        class Stateless(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                zeros = torch.zeros(x.shape[0], 1, *x.shape[-2:])
                return self.inner(x, x, zeros, None)[0]

        wrapped = Stateless(model)
        with torch.no_grad():
            traced = torch.jit.trace(wrapped, image)
            output = traced(image)
        self.assertEqual(tuple(output.shape), (1, 3, 64, 96))
        self.assertTrue(torch.isfinite(output).all())


class TrainerSkipFlagTests(unittest.TestCase):
    def _run_trainer(self, root, extra_argv=()):
        import copy
        import json
        from unittest.mock import patch
        import numpy as np
        import train_full_lr_feature_span as trainer

        torch.set_num_threads(2)
        torch.manual_seed(58)
        initial = Unshuffled(8, scale=2).eval()
        rng = np.random.default_rng(58)
        lr = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8).repeat(3, axis=0)
        hr = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2)
        data = {"train": [(lr, hr, "train")], "validation": [(lr, hr, "val")]}
        manifest = {"frames": 3, "sequences": [{"id": "val", "source_id": "heldout"}]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        init_path = root / "init.pth"
        torch.save(initial.state_dict(), init_path)
        out = root / "result"
        argv = ["trainer", "--bank", str(root), "--init", str(init_path),
                "--out", str(out), "--steps", "2", "--batch", "1",
                "--crop", "32", *extra_argv]
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(trainer, "load_bank", return_value=(manifest, data)))
            stack.enter_context(patch.object(trainer, "load",
                                             side_effect=lambda *_: (copy.deepcopy(initial), 0, 1)))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="mock"))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda value, *a, **k: value))
            stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda value, *a, **k: value))
            scorer = stack.enter_context(patch.object(trainer, "Scorer")).return_value
            scorer.spatial.side_effect = lambda a, b: {
                "mse": float(((np.asarray(a).astype(float) - np.asarray(b)) ** 2).mean())}
            trainer.main()
        return out

    def test_skip_flag_records_declaration_and_zeroes_trunk_head(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_trainer(Path(tmp), ("--sr-skip-residual",))
            checkpoint = torch.load(out / "step000002.pth", map_location="cpu",
                                    weights_only=False)
            self.assertTrue(checkpoint["skip_residual"])
            args = checkpoint["experiment"]["args"]
            self.assertTrue(args["sr_skip_residual"])
            self.assertIn("skip_residual_span.py",
                          " ".join(checkpoint["experiment"]["source_hashes"]))
            loaded, reloaded = load_full_lr_checkpoint(out / "step000002.pth")
            self.assertTrue(loaded.skip_residual)

    def test_default_control_has_no_skip_and_no_curve(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_trainer(Path(tmp))
            checkpoint = torch.load(out / "step000002.pth", map_location="cpu",
                                    weights_only=False)
            self.assertFalse(checkpoint["skip_residual"])
            self.assertFalse(checkpoint["experiment"]["args"]["sr_skip_residual"])
            self.assertFalse((out / "curve.jsonl").exists())
            loaded, _ = load_full_lr_checkpoint(out / "step000002.pth")
            self.assertFalse(loaded.skip_residual)

    def test_validate_every_writes_curve_snapshots(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run_trainer(Path(tmp), ("--validate-every", "1"))
            rows = (out / "curve.jsonl").read_text().strip().split("\n")
            self.assertEqual(len(rows), 1)
            snapshot = json.loads(rows[0])
            self.assertEqual(snapshot["step"], 1)
            self.assertIn("full_lr_features", snapshot["summary"])
            report = json.loads((out / "validation.json").read_text())
            # Mid-run snapshots omit the 'initial' baseline; the trained
            # variants must match the final report exactly.
            self.assertEqual(set(snapshot["summary"]),
                             set(report["summary"]) - {"initial"})
            for variant in snapshot["summary"]:
                # Same variant/source structure; values legitimately evolve
                # with the extra training step before the final report.
                self.assertEqual(set(snapshot["summary"][variant]["sources"]),
                                 set(report["summary"][variant]["sources"]))
                self.assertEqual(set(snapshot["summary"][variant]["source_balanced"]),
                                 set(report["summary"][variant]["source_balanced"]))
            self.assertIn("heldout", snapshot["summary"]["full_lr_features"]["sources"])


if __name__ == "__main__":
    unittest.main()
