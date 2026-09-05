import unittest

import torch

from frequency_split_probe import split_residual


class FrequencySplitTest(unittest.TestCase):
    def test_equal_outputs_and_constant_color_residual(self):
        torch.manual_seed(1)
        base = torch.rand(1,3,32,32)
        self.assertTrue(torch.equal(split_residual(base,base),base))
        shift = torch.tensor([.1,.2,.3]).reshape(1,3,1,1)
        self.assertLess(float((split_residual(base,base+shift)-base-shift).abs().max()), 1e-6)

    def test_high_frequency_change_is_attenuated_without_reference_input(self):
        y,x = torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        checker = ((x+y)%2).float().mul(2).sub(1)[None,None].expand(1,3,32,32)
        base = torch.full_like(checker,.5)
        output = split_residual(base,base+.2*checker)
        self.assertLess(float((output-base)[:,:,3:-3,3:-3].abs().mean()),.001)
        with self.assertRaises(ValueError): split_residual(base,base[:,:,:16])


if __name__ == '__main__': unittest.main()
