import sys
import tempfile
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_span import Unshuffled
from eval_checkpoint import load
from architectures.subspace_adapter import fuse_convolutions
from architectures.spatial_activation_control import SpatialActivationControl, CoarseSpatialActivationControl
from test_activation_control import anchor_digest


class SpatialControlTests(unittest.TestCase):
    model_class = SpatialActivationControl
    architecture = 'spatial_activation_control2x'
    margin = 88
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20260905)
        anchor = Unshuffled(32, scale=2).eval()
        fuse_convolutions(anchor)
        self.config = dict(selection=[[0,3,7,11]]*6, rms=[[.3]*32]*6, dynamic=True, window=9)
        self.model = self.model_class(anchor, **self.config)

    def test_identity_real_gradient_and_frozen_anchor(self):
        x = torch.rand(2,3,24,32)
        self.assertTrue(torch.equal(self.model(x), self.model.anchor(x)))
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.equal(self.model(x), self.model.anchor(x)))
        before = anchor_digest(self.model.anchor)
        optimizer = torch.optim.AdamW(self.model.controller.parameters(), lr=.001)
        for _ in range(3):
            optimizer.zero_grad()
            (self.model(x)-.5).square().mean().backward()
            optimizer.step()
        self.assertEqual(anchor_digest(self.model.anchor), before)
        self.assertGreater(float(self.model.controller[1].weight.grad.abs().sum()), 0)

    def test_nonzero_control_has_crop_local_interior(self):
        torch.nn.init.normal_(self.model.controller[-1].weight, std=.1)
        self.model.eval()
        x = torch.rand(1,3,192,192)
        with torch.inference_mode():
            full = self.model(x)[:,:,64:320,64:320]
            cropped = self.model(x[:,:,32:160,32:160].contiguous())
        m = self.margin
        self.assertLess(float((full[:,:,m:-m,m:-m]-cropped[:,:,m:-m,m:-m]).abs().max()), 1e-5)
        self.assertGreater(float((full-cropped).abs().max()), 1e-4)

    def test_nonzero_checkpoint_roundtrip(self):
        torch.nn.init.normal_(self.model.controller[-1].weight, std=.1)
        x = torch.rand(1,3,24,32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'probe.pth'
            torch.save(dict(model=self.model.state_dict(),channels=32,scale=2,frames=1,
                            architecture=self.architecture,controller_config=self.config),path)
            restored,_,_ = load(path,'cpu')
            self.assertTrue(torch.equal(self.model(x),restored(x)))


class CoarseSpatialControlTests(SpatialControlTests):
    model_class = CoarseSpatialActivationControl
    architecture = 'coarse_spatial_activation_control2x'
    margin = 96


if __name__ == '__main__':
    unittest.main()
