import unittest
import torch
from torch.nn import functional as F
from deterministic_resize import bicubic


class ResizeTests(unittest.TestCase):
    def test_forward_and_backward_match_antialias_including_borders(self):
        torch.set_num_threads(2)
        torch.manual_seed(719)
        for source, target in [((17, 23), (28, 32)), ((32, 40), (19, 17)), ((176, 176), (224, 224))]:
            x = torch.rand(2, 3, *source, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            expected = F.interpolate(x, target, mode='bicubic', align_corners=False, antialias=True)
            actual = bicubic(y, target)
            self.assertLess(float((expected-actual).detach().abs().max()), 1e-6)
            gradient = torch.randn_like(actual)
            expected.backward(gradient); actual.backward(gradient)
            self.assertLess(float((x.grad-y.grad).abs().max()), 3e-6)


if __name__ == '__main__':
    unittest.main()
