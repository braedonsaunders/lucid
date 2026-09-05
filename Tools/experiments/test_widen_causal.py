import unittest
import torch
from widen_causal import CausalDetailV2, widen


class WidthExpansionTests(unittest.TestCase):
    def test_reconstruction_preserved_and_new_channels_can_learn(self):
        torch.manual_seed(3)
        original = CausalDetailV2(8, 2).eval()
        torch.nn.init.normal_(original.head.weight, std=.01)
        expanded = widen(original)
        x = torch.rand(1, 3, 16, 20)
        a, b = original.initial_state(x), expanded.initial_state(x)
        for valid in [0, 1, 1, 0]:
            mask = torch.full((1, 1, 1, 1), float(valid))
            y, a = original(x, a, mask)
            z, b = expanded(x, b, mask)
            torch.testing.assert_close(y, z, rtol=1e-5, atol=1e-6)
        z.square().mean().backward()
        gradient = expanded.entry.weight.grad
        self.assertGreater((gradient[:8] - gradient[8:]).abs().max().item(), 1e-8)


if __name__ == '__main__':
    unittest.main()
