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
    def test_lowpass_branch_learning_and_checkpoint_cannot_lose_its_filter(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontier_eval'))
        from frequency_split_probe import split_residual
        torch.manual_seed(15)
        regular = AnchoredDetail(Unshuffled(channels=4, scale=2), channels=8, blocks=2).eval()
        filtered = AnchoredDetail(Unshuffled(channels=4, scale=2), channels=8, blocks=2,
                                  residual_lowpass=True).eval()
        result = filtered.load_state_dict(regular.state_dict(), strict=False)
        self.assertEqual(result.missing_keys, ['residual_kernel'])
        self.assertEqual(result.unexpected_keys, [])
        image = torch.rand(1,3,12,18)
        with torch.no_grad():
            torch.testing.assert_close(filtered(image), filtered.anchor(image), rtol=0, atol=0)
            torch.nn.init.normal_(regular.head.weight, std=.01)
            filtered.head.load_state_dict(regular.head.state_dict())
            expected = split_residual(regular.anchor(image), regular(image))
            torch.testing.assert_close(filtered(image), expected, rtol=0, atol=1e-6)
        filtered.train()
        fixed = {k:v.detach().clone() for k,v in filtered.anchor.state_dict().items()}
        kernel = filtered.residual_kernel.clone()
        optimizer = torch.optim.AdamW((p for p in filtered.parameters() if p.requires_grad), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            (filtered(image)-.5).square().mean().backward()
            optimizer.step()
        self.assertGreater(float(filtered.input.weight.grad.abs().sum()), 0)
        self.assertGreater(float(filtered.head.weight.grad.abs().sum()), 0)
        for name, value in filtered.anchor.state_dict().items():
            torch.testing.assert_close(value, fixed[name], rtol=0, atol=0)
        torch.testing.assert_close(filtered.residual_kernel, kernel, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            filtered.eval()
            checkpoint = {'model': filtered.state_dict(), 'channels': 4, 'frames': 1, 'scale': 2,
                          'architecture': 'anchored_lowpass2x', 'detail_channels': 8, 'detail_blocks': 2}
            path = Path(directory)/'filtered.pth'; torch.save(checkpoint,path)
            restored, _, _ = load(path,'cpu')
            torch.testing.assert_close(restored(image), filtered(image), rtol=0, atol=0)
            traced = torch.jit.trace(restored,image)
            torch.testing.assert_close(traced(image), filtered(image), rtol=0, atol=1e-6)
            checkpoint['architecture'] = 'anchored_detail2x'; torch.save(checkpoint,path)
            with self.assertRaises(RuntimeError): load(path,'cpu')

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
