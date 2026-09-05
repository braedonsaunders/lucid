import unittest
import numpy as np
import torch
from torch.nn import functional as F

from probe_learned_past_information import align_prediction, features, prediction_channels, remap


class LearnedPastTests(unittest.TestCase):
    def test_prediction_phase_roundtrip_is_lossless_and_ordered(self):
        image=np.random.default_rng(19).integers(0,256,(8,10,3),dtype=np.uint8)
        def model(x):
            result=F.interpolate(x,scale_factor=4,mode='nearest')
            phase=torch.arange(16).reshape(4,4).repeat(8,10)/1024
            return result+phase
        packed=prediction_channels(model,image,'cpu')
        self.assertEqual(packed.shape,(16,20,12))
        reconstructed=F.pixel_shuffle(torch.from_numpy(packed).permute(2,0,1)[None],2)
        reference=model(torch.from_numpy(image.astype(np.float32)/255).permute(2,0,1)[None])
        self.assertTrue(torch.equal(reference,reconstructed))

    def test_identity_flow_preserves_all_twelve_learned_channels(self):
        rng=np.random.default_rng(21)
        image=rng.integers(0,256,(64,64,3),dtype=np.uint8)
        prediction=rng.random((128,128,12),dtype=np.float32)
        aligned,visible=align_prediction(image,image,prediction)
        np.testing.assert_allclose(aligned[visible],prediction[visible],atol=1e-7)
        with self.assertRaises(ValueError):align_prediction(image,image,prediction[:,:,:3])

    def test_current_and_past_features_have_fixed_channel_coordinates(self):
        current=torch.zeros(1,3,5,6);prediction=torch.ones(1,12,5,6)
        history=[prediction+1,prediction+2];masks=[torch.ones(5,6),torch.zeros(5,6)]
        x=features(current,current,prediction,history,masks,'aligned')
        self.assertEqual(tuple(x.shape),(30,93))
        self.assertTrue(torch.equal(x[:,54:78],torch.ones(30,24)))
        self.assertTrue(torch.equal(x[:,78:90],torch.full((30,12),2.)))
        self.assertTrue(torch.equal(x[:,-3:],torch.tensor([1.,0.,1.]).expand(30,-1)))
        self.assertEqual(tuple(features(current,current,prediction,history,masks,'spatial').shape),(30,67))

    def test_backward_flow_moves_every_prediction_channel_together(self):
        prediction=np.arange(8*10*12,dtype=np.float32).reshape(8,10,12)
        flow=np.zeros((8,10,2),np.float32);flow[...,0]=1
        warped,valid=remap(prediction,flow)
        np.testing.assert_array_equal(warped[:,:-1],prediction[:,1:])
        self.assertFalse(valid[:,-2:].any())


if __name__=='__main__':unittest.main()
