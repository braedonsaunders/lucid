import unittest
import numpy as np
from build_causal_bank import validate_sources
from train_causal_detail import batch


class CausalBankTests(unittest.TestCase):
    def test_family_leakage_is_rejected(self):
        records = [{'id':'a','family':'movie','split':'train','sha256':'a'},
                   {'id':'b','family':'movie','split':'validation','sha256':'b'}]
        with self.assertRaisesRegex(ValueError, 'family'):
            validate_sources(records)

    def test_duplicate_content_cannot_cross_split(self):
        records = [{'id':'a','family':'movie_a','split':'train','sha256':'same'},
                   {'id':'b','family':'movie_b','split':'validation','sha256':'same'}]
        with self.assertRaisesRegex(ValueError, 'identical'):
            validate_sources(records)

    def test_batch_augmentation_preserves_space_and_time_alignment(self):
        rng = np.random.default_rng(3)
        lr = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
        hr = lr.repeat(2, axis=1).repeat(2, axis=2)
        x, y = batch([(lr, hr, 'test')], rng, 4, 7, 16)
        self.assertEqual(tuple(x.shape), (4, 7, 3, 16, 16))
        self.assertEqual(tuple(y.shape), (4, 7, 3, 32, 32))
        np.testing.assert_array_equal(y.numpy()[:, :, :, ::2, ::2], x.numpy())


if __name__ == '__main__':
    unittest.main()
