from collections import Counter
import unittest
import numpy as np

from source_sampling import SourceBalancedSequences
from train_causal_detail import batch


class SourceSamplingTests(unittest.TestCase):
    def fixtures(self, counts):
        data, rows = [], []
        for source, count in enumerate(counts):
            for sequence in range(count):
                identity = f'{source}-{sequence}'
                data.append((np.full((3, 8, 8, 3), source*20+sequence, np.uint8),
                             np.full((3, 16, 16, 3), source*20+sequence, np.uint8), identity))
                rows.append({'id': identity, 'source_id': str(source), 'split': 'train'})
        return data, rows

    def test_exact_uniform_sources_and_sequences(self):
        data, rows = self.fixtures([2, 3, 4])
        balanced = SourceBalancedSequences(data, rows)
        counts = Counter(balanced[i][2] for i in range(len(balanced)))
        self.assertEqual(len(balanced), 36)
        self.assertEqual([sum(n for identity, n in counts.items() if identity.startswith(f'{s}-')) for s in range(3)], [12]*3)
        self.assertEqual(set(counts[str(s)+'-0'] for s in range(3)), {6, 4, 3})
        self.assertIs(balanced[0][0], data[0][0])

    def test_already_balanced_sampler_preserves_batch_and_rng(self):
        data, rows = self.fixtures([3, 3])
        left, right = np.random.default_rng(72), np.random.default_rng(72)
        a = batch(data, left, 4, 2, 6)
        b = batch(SourceBalancedSequences(data, rows), right, 4, 2, 6)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x.numpy(), y.numpy()))
        self.assertEqual(left.bit_generator.state, right.bit_generator.state)

    def test_bad_split_identity_order_and_indices_rejected(self):
        data, rows = self.fixtures([2, 3])
        bad = [dict(r) for r in rows]
        bad[0]['split'] = 'validation'
        with self.assertRaises(ValueError): SourceBalancedSequences(data, bad)
        bad = [dict(r) for r in rows]; bad[1]['id'] = bad[0]['id']
        with self.assertRaises(ValueError): SourceBalancedSequences(data, bad)
        balanced = SourceBalancedSequences(list(reversed(data)), rows)
        with self.assertRaises(ValueError): balanced[0]
        for index in (-1, len(balanced)):
            with self.assertRaises(IndexError): balanced[index]


if __name__ == '__main__':
    unittest.main()
