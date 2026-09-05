import unittest

import torch

from interpolate_presented_detail import interpolate


class ParameterInterpolationTest(unittest.TestCase):
    def test_exact_endpoints_and_unchanged_sources(self):
        a = {'scale': 2, 'model': {'w': torch.tensor([1e-20, 1e20]), 'count': torch.tensor(3)}}
        b = {'scale': 2, 'model': {'w': torch.tensor([2e-20, -1e20]), 'count': torch.tensor(3)}}
        for alpha, source in [(0, a), (1, b)]:
            out = interpolate(a, b, alpha)
            self.assertTrue(torch.equal(out['w'], source['model']['w']))
            out['w'].zero_()
            self.assertTrue(source['model']['w'].abs().sum() > 0)

    def test_wrong_architecture_and_invalid_parameters_rejected(self):
        a = {'scale': 2, 'model': {'w': torch.ones(2)}}
        for b in [{'scale': 4, 'model': {'w': torch.ones(2)}},
                  {'scale': 2, 'model': {'w': torch.ones(3)}},
                  {'scale': 2, 'model': {'w': torch.tensor([1., float('nan')])}}]:
            with self.assertRaises(ValueError):
                interpolate(a, b, .5)
        with self.assertRaises(ValueError):
            interpolate(a, a, float('nan'))


if __name__ == '__main__':
    unittest.main()
