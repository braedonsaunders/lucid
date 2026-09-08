"""Packaging regressions: alternatives cannot replace or bypass reference identity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import package_models


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.manifest_path = root / 'Models.json'
        self.packages = root / 'Model'
        self.items = []
        for name in ('image', 'tensor'):
            package = self.packages / (name + '.mlpackage')
            package.mkdir(parents=True)
            (package / 'Manifest.json').write_text('{}')
            (package / 'weights').write_bytes(b'frozen weights')
            self.items.append(dict(name=name, width=256, height=144, scale=4,
                                   sha256=package_models.digest(package)))
        self.manifest = dict(models=self.items)
        self.addCleanup(patch.stopall)
        patch.object(package_models, 'ROOT', root).start()
        patch.object(package_models, 'MANIFEST', self.manifest_path).start()

    def verify(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return package_models.verify()

    def test_every_model_is_verified_and_tampering_is_rejected(self):
        self.assertEqual(len(package_models.all_models(self.verify())), 2)
        (self.packages / 'tensor.mlpackage' / 'weights').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.verify()

    def test_duplicate_names_are_rejected(self):
        self.items[1]['name'] = 'image'
        with self.assertRaisesRegex(ValueError, 'Invalid model name'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
