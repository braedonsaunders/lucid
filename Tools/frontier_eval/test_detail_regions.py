import unittest
import numpy as np
from diagnose_detail_regions import blur_float, region_metrics


class DetailRegionTests(unittest.TestCase):
    def test_float_blur_preserves_constants_and_scaling(self):
        image = np.arange(32*32, dtype=np.float64).reshape(32, 32)/8
        np.testing.assert_allclose(blur_float(image*.37+11), blur_float(image)*.37+11, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(blur_float(np.full((32, 32), 10.5)), 10.5)

    def test_masks_depend_only_on_reference_and_blur_loses_edge_detail(self):
        reference = np.zeros((64, 64)); reference[:, 32:] = 200
        perfect = region_metrics(reference, reference)
        smooth = region_metrics(blur_float(reference), reference)
        for region in perfect['regions']:
            self.assertEqual(perfect['regions'][region]['coverage'], smooth['regions'][region]['coverage'])
        self.assertAlmostEqual(perfect['regions']['edges']['fine_correlation'], 1)
        self.assertGreater(smooth['regions']['edges']['fine_mae'], 0)
        self.assertLess(smooth['regions']['edges']['fine_energy_ratio'], 1)


if __name__ == '__main__':
    unittest.main()
