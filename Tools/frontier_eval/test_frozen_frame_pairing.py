import unittest

from score_checkpoint_frames import paired_references


class FrozenFramePairingTest(unittest.TestCase):
    def test_malformed_pairs_cannot_be_scored(self):
        reference = {'side': 'reference', 'sequence_id': 'clip', 'frame': 0, 'source_id': 'original'}
        degraded = {**reference, 'side': 'degraded'}
        self.assertEqual(paired_references([degraded, reference])[('clip', 0)], reference)
        for rows in [[], [reference], [reference, reference, degraded],
                     [reference, {**degraded, 'source_id': 'other'}],
                     [reference, {**degraded, 'frame': 1}]]:
            with self.assertRaises(ValueError):
                paired_references(rows)


if __name__ == '__main__':
    unittest.main()
