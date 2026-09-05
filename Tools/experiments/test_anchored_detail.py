from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.anchored_detail import AnchoredDetail
from train_span import Unshuffled
from eval_checkpoint import load


class AnchoredDetailTests(unittest.TestCase):
    def test_zero_residual_preserves_real_anchor_including_borders(self):
        torch.manual_seed(12)
        anchor = Unshuffled(channels=4, scale=2).eval()
        model = AnchoredDetail(anchor, channels=8, blocks=2).train()
        self.assertFalse(model.anchor.training)
        image = torch.rand(1, 3, 12, 18)
        with torch.no_grad():
            expected, actual = anchor(image), model(image)
        self.assertEqual(tuple(actual.shape), (1, 3, 24, 36))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_learning_updates_detail_without_changing_anchor(self):
        torch.manual_seed(13)
        model = AnchoredDetail(Unshuffled(channels=4, scale=2), channels=8, blocks=2).train()
        image = torch.rand(1, 3, 12, 18)
        target = torch.rand(1, 3, 24, 36)
        model(image)  # Materialize the anchor's inference convolutions first.
        fixed = {k: v.detach().clone() for k, v in model.anchor.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(); (model(image) - target).square().mean().backward(); optimizer.step()
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.input.weight.grad.abs().sum()), 0)
        for key, value in model.anchor.state_dict().items():
            torch.testing.assert_close(value, fixed[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.anchor.parameters()))

    def test_nonzero_graph_trace_and_state_roundtrip(self):
        torch.manual_seed(14)
        model = AnchoredDetail(Unshuffled(channels=4, scale=2), channels=8, blocks=2).eval()
        torch.nn.init.normal_(model.head.weight, std=.001)
        image = torch.rand(1, 3, 12, 18)
        with torch.no_grad():
            expected = model(image)
            self.assertGreater(float((expected - model.anchor(image)).abs().max()), 0)
            restored = AnchoredDetail(Unshuffled(channels=4, scale=2), channels=8, blocks=2).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'detail.pth'
                torch.save({'model': model.state_dict(), 'channels': 4, 'frames': 1, 'scale': 2,
                            'architecture': 'anchored_detail2x', 'detail_channels': 8, 'detail_blocks': 2}, path)
                restored, _, frames = load(path, 'cpu')
                self.assertEqual(frames, 1)
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
            traced = torch.jit.trace(model, image)
            torch.testing.assert_close(traced(image), expected)


if __name__ == '__main__': unittest.main()
