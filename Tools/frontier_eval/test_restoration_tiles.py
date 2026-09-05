import unittest
import numpy as np
from restoration_tiles import positions, merge_patches


class RestorationTileTests(unittest.TestCase):
    def test_identity_tiles_preserve_every_pixel_and_border(self):
        image = np.random.default_rng(9).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        patches = [(x, y, image[y:y+512, x:x+512]) for y in positions(720) for x in positions(1280)]
        np.testing.assert_array_equal(merge_patches(patches, 1280, 720), image)

    def test_missing_coverage_is_not_filled_with_black(self):
        with self.assertRaisesRegex(ValueError, 'cover'):
            merge_patches([(0, 0, np.zeros((512, 512, 3), dtype=np.uint8))], 720, 720)


if __name__ == '__main__':
    unittest.main()
