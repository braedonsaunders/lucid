import json
from pathlib import Path
import tempfile
import unittest

from build_causal_bank import digest
from merge_training_banks import compose


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bank(self, label):
        bank, cache = self.root / label / 'bank', self.root / label / 'cache'
        bank.mkdir(parents=True); cache.mkdir()
        sources, rows = [], []
        for split in ('train', 'validation'):
            identity = f'{label}_{split}'
            source = {'id': identity, 'family': identity, 'split': split, 'sha256': identity}
            sources.append(source)
            path = bank / f'{identity}.npz'; path.write_bytes(identity.encode())
            rows.append({'id': identity, 'source_id': identity, 'source_sha256': source['sha256'],
                         'family': identity, 'split': split, 'file': path.name, 'sha256': digest(path)})
        self.write(bank / 'manifest.json', {'scale': 2, 'frames': 16, 'lr_patch': 256, 'sources': sources, 'sequences': rows})
        target = cache / 'target.npz'; target.write_bytes(b'teacher')
        self.write(cache / 'manifest.json', {'bank_sha256': digest(bank / 'manifest.json'), 'split': 'train only',
                   'teacher_sha256': 'pinned', 'inference': 'one step', 'baseline_provenance_sha256': 'pinned',
                   'code_sha256': 'pinned', 'seed': 20260913, 'reference_used_for_prediction': False,
                   'sequences': [{'id': rows[0]['id'], 'file': target.name, 'sha256': digest(target)}]})
        return bank, cache

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def mutate_bank(self, pair, mutate):
        bank, cache = pair
        path = bank / 'manifest.json'; value = json.loads(path.read_text()); mutate(value); self.write(path, value)
        path = cache / 'manifest.json'; value = json.loads(path.read_text())
        value['bank_sha256'] = digest(bank / 'manifest.json'); self.write(path, value)

    def test_composes_without_copying_or_relabeling(self):
        first, second = self.bank('one'), self.bank('two')
        bank, targets, receipts = compose([first, second])
        self.assertEqual(len(bank['sources']), 4); self.assertEqual(len(targets), 2)
        self.assertEqual(len(receipts), 2)
        for row in bank['sequences'] + targets:
            self.assertTrue(Path(row['file']).is_absolute())
            self.assertEqual(digest(row['file']), row['sha256'])

    def test_rejects_cross_bank_family_or_content_leakage(self):
        for field in ('family', 'sha256'):
            with self.subTest(field=field):
                first, second = self.bank('a' + field), self.bank('b' + field)
                value = json.loads((first[0] / 'manifest.json').read_text())['sources'][0][field]
                def mutate(m):
                    m['sources'][1][field] = value
                    m['sequences'][1]['source_sha256' if field == 'sha256' else field] = value
                self.mutate_bank(second, mutate)
                with self.assertRaisesRegex(ValueError, 'crosses split'): compose([first, second])

    def test_rejects_changed_data_and_cache(self):
        for kind in ('bank', 'cache'):
            pair = self.bank(kind)
            path = next(pair[0 if kind == 'bank' else 1].glob('*.npz')); path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'): compose([pair])

    def test_rejects_teacher_validation_leakage(self):
        pair = self.bank('leak'); path = pair[1] / 'manifest.json'
        value = json.loads(path.read_text()); value['sequences'][0]['id'] = 'leak_validation'; self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'validation identity'): compose([pair])

    def test_rejects_duplicate_banks_and_incompatible_geometry(self):
        first, second = self.bank('first'), self.bank('second')
        with self.assertRaisesRegex(ValueError, 'duplicate sequence'): compose([first, first])
        self.mutate_bank(second, lambda m: m.update(lr_patch=128))
        with self.assertRaisesRegex(ValueError, 'geometry'): compose([first, second])

    def test_rejects_incomplete_or_foreign_cache(self):
        for missing in (True, False):
            pair = self.bank('missing' if missing else 'foreign'); path = pair[1] / 'manifest.json'
            value = json.loads(path.read_text())
            if missing: value['sequences'] = []
            else: value['bank_sha256'] = 'foreign'
            self.write(path, value)
            with self.assertRaises(ValueError): compose([pair])

    def test_rejects_mixed_teacher_recipes(self):
        first, second = self.bank('old'), self.bank('new')
        path = second[1] / 'manifest.json'; value = json.loads(path.read_text())
        value['seed'] += 1; self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'generation settings'): compose([first, second])


if __name__ == '__main__': unittest.main()
