import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_span import Unshuffled
from eval_checkpoint import load
from architectures.subspace_adapter import fuse_convolutions
from architectures.activation_control import ActivationControl


def anchor_digest(model):
    return hashlib.sha256(b''.join(v.detach().contiguous().numpy().tobytes()
        for v in model.state_dict().values())).hexdigest()


class ActivationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(92)
        self.anchor = Unshuffled(32, scale=2).eval()
        fuse_convolutions(self.anchor)
        self.config = {'selection': [[0, 3, 7, 11]]*6, 'rms': [[.3]*32]*6, 'dynamic': True}
        self.model = ActivationControl(self.anchor, **self.config)

    def test_identity_frozen_anchor_and_real_gradient(self):
        x = torch.rand(2, 3, 24, 32)
        expected = self.anchor(x)
        self.assertTrue(torch.equal(self.model(x), expected))
        before = anchor_digest(self.model.anchor)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=.001)
        self.model.train()
        for _ in range(3):
            optimizer.zero_grad()
            (self.model(x)-.5).square().mean().backward()
            optimizer.step()
        self.assertEqual(anchor_digest(self.model.anchor), before)
        self.assertFalse(self.model.anchor.training)
        self.assertFalse(torch.equal(self.model(x), expected))
        self.assertGreater(float(self.model.controller[1].weight.grad.abs().sum()), 0)

    def test_conditioning_and_sparse_bounds(self):
        torch.nn.init.normal_(self.model.controller[-1].weight, std=.1)
        features = torch.randn(2, 32, 8, 12)
        values = self.model.coefficients(features).detach()
        self.assertFalse(torch.equal(values[0], values[1]))
        self.assertLessEqual(float(values.abs().max()), 1)
        for i in range(6):
            scale = 1+.5*(values[:, i, 0] @ self.model.masks[i])
            shift = .2*(values[:, i, 1] @ self.model.masks[i])*self.model.shift_units[i]
            self.assertGreaterEqual(float(scale.min()), .5)
            self.assertLessEqual(float(scale.max()), 1.5)
            excluded = [j for j in range(32) if j not in self.config['selection'][i]]
            self.assertTrue(torch.equal(scale[:, excluded], torch.ones_like(scale[:, excluded])))
            self.assertEqual(float(shift[:, excluded].abs().sum()), 0)
        self.model.dynamic = False
        fixed = self.model.coefficients(features)
        self.assertTrue(torch.equal(fixed[0], fixed[1]))

    def test_zero_controller_preserves_autocast_residual_dtype(self):
        x = torch.rand(2, 3, 24, 32)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            expected = self.anchor(x)
            actual = self.model(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_nonzero_checkpoint_roundtrip(self):
        torch.nn.init.normal_(self.model.controller[-1].weight, std=.01)
        x = torch.rand(1, 3, 24, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'probe.pth'
            torch.save({'model': self.model.state_dict(), 'channels': 32, 'scale': 2, 'frames': 1,
                        'architecture': 'activation_control2x', 'controller_config': self.config}, path)
            restored, _, _ = load(path, 'cpu')
            self.assertTrue(torch.equal(self.model(x), restored(x)))


if __name__ == '__main__':
    unittest.main()
