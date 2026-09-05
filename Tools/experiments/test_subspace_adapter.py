import copy
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures.subspace_adapter import SubspaceConv, attach_adapters, merge_adapters, adapter_report
from eval_checkpoint import load
from train_span import Unshuffled


class SubspaceAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(905)

    def test_adam_preserves_weight_constraint_and_anchor(self):
        layer = SubspaceConv(nn.Conv2d(4, 8, 3, padding=1), energy=.6)
        anchor = layer.anchor.clone()
        self.assertGreater(layer.coefficients.numel(), 0)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.003)
        x, target = torch.randn(2, 4, 8, 10), torch.randn(2, 8, 8, 10)
        initial = layer(x).detach()
        for _ in range(20):
            optimizer.zero_grad()
            (layer(x) - target).square().mean().backward()
            optimizer.step()
        self.assertTrue(torch.equal(anchor, layer.anchor))
        self.assertGreater((layer(x) - initial).abs().max().item(), .001)
        actual_delta = layer.effective_matrix() - anchor
        self.assertLess((layer.protected_basis.T @ actual_delta).abs().max().item(), 1e-6)
        torch.testing.assert_close(layer(x), layer.merged()(x), rtol=1e-5, atol=1e-6)

    def test_full_coordinate_control_can_change_protected_responses(self):
        original = nn.Conv2d(4, 8, 3, padding=1)
        control = SubspaceConv(original, constrained=False, energy=.6)
        protected = SubspaceConv(original, constrained=True, energy=.6)
        x = torch.randn(1, 4, 8, 10)
        self.assertTrue(torch.equal(control(x), protected(x)))
        with torch.no_grad():
            control.coefficients[0].fill_(.02)
        self.assertGreater((control.protected_basis.T @ control.delta()).abs().max().item(), .01)

    def test_fused_checkpoint_has_no_adapters_and_roundtrips(self):
        base = Unshuffled(channels=8, scale=2).eval()
        adapted = attach_adapters(base, energy=.6)
        x = torch.randn(1, 3, 14, 18)
        with torch.no_grad():
            torch.testing.assert_close(base(x), adapted(x), rtol=1e-5, atol=1e-6)
            for layer in adapted.modules():
                if isinstance(layer, SubspaceConv):layer.coefficients.normal_(std=.0002)
            expected = adapted(x)
        merged = merge_adapters(adapted)
        self.assertFalse(any(isinstance(m, SubspaceConv) for m in merged.modules()))
        self.assertTrue(all(r['protected_update_norm'] < 1e-5 for r in adapter_report(adapted)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'merged.pth'
            payload = {'architecture':'fused_span2x','channels':8,'scale':2,'frames':1,'version':1,'model':merged.state_dict()}
            torch.save(payload, path)
            restored, _, frames = load(path, 'cpu')
            self.assertEqual(frames, 1)
            torch.testing.assert_close(restored(x), expected, rtol=1e-5, atol=1e-6)
            payload['architecture'] = 'shipping_direct2x_area'
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):load(path, 'cpu')

    def test_gradient_projection_is_not_an_adam_weight_constraint(self):
        u = torch.tensor([[1.], [2.]]) / 5**.5
        weights = nn.Parameter(torch.zeros(2, 1))
        optimizer = torch.optim.AdamW([weights], lr=.01, weight_decay=0)
        gradient = torch.tensor([[2.], [-1.]])
        weights.grad = gradient - u @ (u.T @ gradient)
        self.assertLess((u.T @ weights.grad).abs().item(), 1e-6)
        optimizer.step()
        self.assertGreater((u.T @ weights).abs().item(), .004)


if __name__ == '__main__':unittest.main()
