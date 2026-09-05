import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from export_bank_validation import digest, export


class ValidationExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.bank = self.root / 'bank'; self.bank.mkdir()
        self.lr = np.arange(8 * 4 * 4 * 3, dtype=np.uint8).reshape(8, 4, 4, 3)
        self.hr = np.repeat(np.repeat(self.lr, 2, axis=1), 2, axis=2)
        np.savez(self.bank / 'val.npz', lr=self.lr, hr=self.hr)
        sources = [{'id': s, 'family': s, 'sha256': s, 'split': s} for s in ['train', 'validation']]
        rows = [{'id': s['id'], 'source_id': s['id'], 'source_sha256': s['sha256'], 'family': s['family'],
                 'split': s['split'], 'file': 'val.npz' if s['split'] == 'validation' else 'deliberately-absent-train.npz',
                 'sha256': digest(self.bank / 'val.npz')} for s in sources]
        self.manifest = {'scale': 2, 'frames': 8, 'lr_patch': 4, 'sources': sources, 'sequences': rows}
        self.write_manifest()

    def write_manifest(self):
        (self.bank / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_exact_frames_and_pairing_without_accessing_training_files(self):
        out = self.root / 'out'; result = export(self.bank, out)
        self.assertEqual(len(result['frames']), 4)
        for row in result['frames']:
            self.assertEqual(row['source_id'], 'validation'); self.assertIn(row['frame'], [0, 4])
            with Image.open(out / row['file']) as image:
                np.testing.assert_array_equal(np.asarray(image), (self.hr if row['side'] == 'reference' else self.lr)[row['frame']])
        with self.assertRaisesRegex(ValueError, 'fresh output'): export(self.bank, out)

    def test_changed_bytes_and_shape_fail(self):
        (self.bank / 'val.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): export(self.bank, self.root / 'changed')
        np.savez(self.bank / 'val.npz', lr=self.lr, hr=self.hr[:-1])
        self.manifest['sequences'][1]['sha256'] = digest(self.bank / 'val.npz'); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'geometry'): export(self.bank, self.root / 'shape')

    def test_mislabeled_or_duplicate_sequences_fail(self):
        self.manifest['sequences'][0]['split'] = 'validation'; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'split'): export(self.bank, self.root / 'leak')
        self.manifest['sequences'][0]['split'] = 'train'
        self.manifest['sequences'].append(self.manifest['sequences'][1]); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'identity'): export(self.bank, self.root / 'duplicate')


if __name__ == '__main__': unittest.main()
