"""Disk-backed training must preserve samples, validation splits and byte identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from build_causal_bank import digest
from mmap_training_bank import materialize, MappedSequences
from train_causal_detail import load_bank, batch


class MappedBankTests(unittest.TestCase):
    def make_bank(self, root, name='bank'):
        root = root/name
        root.mkdir()
        rng = np.random.default_rng(8)
        sources, rows = [], []
        for index, split in enumerate(('train', 'validation')):
            source = {'id': 'source'+str(index), 'family': 'family'+str(index), 'split': split, 'sha256': str(index)*64}
            sources.append(source)
            for j in range(2):
                identity = source['id']+'-'+str(j)
                path = root/(identity+'.npz')
                np.savez_compressed(path, lr=rng.integers(0, 256, (4, 16, 20, 3), dtype=np.uint8),
                                    hr=rng.integers(0, 256, (4, 32, 40, 3), dtype=np.uint8))
                rows.append({'id': identity, 'source_id': source['id'], 'source_sha256': source['sha256'],
                             'family': source['family'], 'split': split, 'file': path.name, 'sha256': digest(path)})
        manifest = {'scale': 2, 'frames': 4, 'lr_patch': 16, 'sources': sources, 'sequences': rows}
        (root/'manifest.json').write_text(json.dumps(manifest))
        return root

    def test_identical_sampling_without_resident_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bank = self.make_bank(root); out = root/'mapped'
            materialize([bank], out)
            _, eager = load_bank(bank); _, mapped = load_bank(out)
            self.assertIsInstance(mapped['train'], MappedSequences)
            self.assertIsInstance(mapped['train'][0][0], np.memmap)
            self.assertFalse(mapped['train'][0][0].flags.writeable)
            left, right = np.random.default_rng(9), np.random.default_rng(9)
            teachers = {identity: hr for _, hr, identity in eager['train']}
            for _ in range(10):
                expected = batch(eager['train'], left, 3, 3, 12, teachers)
                actual = batch(mapped['train'], right, 3, 3, 12)
                self.assertTrue(torch.equal(expected[0], actual[0]))
                self.assertTrue(torch.equal(expected[1], actual[1]))
                self.assertTrue(torch.equal(expected[2], actual[1]))
            self.assertEqual(left.bit_generator.state, right.bit_generator.state)
            self.assertEqual([v[2] for v in eager['validation']], [v[2] for v in mapped['validation']])

    def test_changed_array_and_incomplete_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bank = self.make_bank(root); out = root/'mapped'
            manifest = materialize([bank], out)
            path = out/manifest['sequences'][0]['lr_file']
            with path.open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_bank(out)
            manifest['complete'] = False
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'completed'):
                load_bank(out)

    def test_duplicate_and_cross_split_sources_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bank = self.make_bank(root)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                materialize([bank, bank], root/'duplicate')
            other = self.make_bank(root, 'other')
            manifest = json.loads((other/'manifest.json').read_text())
            manifest['sources'][0]['split'] = 'validation'
            manifest['sources'][1]['split'] = 'train'
            (other/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'source identity'):
                materialize([bank, other], root/'leak')

    def test_input_hash_and_scale_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bank = self.make_bank(root)
            path = next(bank.glob('*.npz'))
            with path.open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                materialize([bank], root/'bad')
            self.assertFalse((root/'bad/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
