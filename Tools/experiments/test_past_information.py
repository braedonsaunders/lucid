import unittest
import numpy as np
import torch

from probe_past_information import aligned_past, features, remap, solve


class PastInformationTests(unittest.TestCase):
    def test_backward_flow_direction_and_visibility(self):
        image=np.arange(8*10*3,dtype=np.float32).reshape(8,10,3)
        flow=np.zeros((8,10,2),np.float32);flow[...,0]=1
        warped,valid=remap(image,flow)
        np.testing.assert_array_equal(warped[:,:-1],image[:,1:])
        self.assertFalse(valid[:,-2:].any())
        self.assertTrue(valid[:-1,:-2].all())

    def test_identity_alignment_and_history_feature_coordinates(self):
        image=np.random.default_rng(20260905).integers(0,256,(64,64,3),dtype=np.uint8)
        from PIL import Image
        expected=np.asarray(Image.fromarray(image).resize((128,128),Image.Resampling.BICUBIC),dtype=np.float32)/255
        aligned,visible=aligned_past(image,image)
        np.testing.assert_allclose(aligned[visible],expected[visible],atol=1e-7)
        a=torch.zeros(1,3,5,6);past=[a+1,a+2];masks=[torch.ones(5,6),torch.zeros(5,6)]
        x=features(a,a,past,masks,'aligned')
        self.assertEqual(tuple(x.shape),(30,111))
        self.assertTrue(torch.equal(x[:,54:81],torch.ones(30,27)))
        self.assertTrue(torch.equal(x[:,81:108],torch.full((30,27),2.)))
        self.assertTrue(torch.equal(x[:,-3:],torch.tensor([1.,0.,1.]).expand(30,-1)))

    def test_fixed_ridge_recovers_known_linear_signal(self):
        generator=torch.Generator().manual_seed(12)
        x=torch.randn(256,12,generator=generator);truth=torch.randn(12,3,generator=generator);y=x@truth
        fitted=solve(x.T@x,x.T@y)
        self.assertLess(float((x@fitted-y).square().mean()/y.square().mean()),1e-5)
        self.assertTrue(torch.equal(fitted,solve(x.T@x,x.T@y)))


if __name__=='__main__':unittest.main()
