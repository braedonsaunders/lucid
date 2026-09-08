import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from score_native_holdout import reusable_scores, digest


class ScoreReuseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.frame = {'sequence_id': 's', 'source_id': 'source', 'frame': 0,
                      'variant': 'shipping4x', 'image_sha256': 'output', 'reference_sha256': 'reference'}
        self.manifest = {'complete': True, 'rows': [self.frame]}
        self.report = {'complete': True, 'device': 'cpu', 'torch': str(torch.__version__),
                       'scorer_sha256': digest(Path(__file__).with_name('evaluate_sequences.py')),
                       'rows': [{**self.frame, 'metrics': {'lpips': .2, 'dists': .1,
                           'psnr_y': 30., 'detail_energy': .9, 'fine_correlation': .8}}]}

    def write(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        self.report['manifest_sha256'] = digest(self.root / 'manifest.json')
        path = self.root / 'scores.json'
        path.write_text(json.dumps(self.report))
        return path

    def test_reuse_requires_both_image_and_reference_identity(self):
        cache = reusable_scores(self.root, self.write(), 'cpu')
        self.assertEqual(cache[('output', 'reference')]['lpips'], .2)
        self.assertNotIn(('changed-output', 'reference'), cache)
        self.assertNotIn(('output', 'changed-reference'), cache)

    def test_reject_changed_manifest_metric_code_environment_or_incomplete_report(self):
        original = copy.deepcopy(self.report)
        for field, value in [('complete', False), ('scorer_sha256', 'changed'),
                             ('device', 'mps'), ('torch', 'changed'), ('rows', [])]:
            self.report = {**copy.deepcopy(original), field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                reusable_scores(self.root, self.write(), 'cpu')
        self.report = original
        path = self.write()
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest) + '\n')
        with self.assertRaises(ValueError):
            reusable_scores(self.root, path, 'cpu')

    def test_reject_duplicate_or_conflicting_scores(self):
        self.report['rows'].append(copy.deepcopy(self.report['rows'][0]))
        with self.assertRaises(ValueError):
            reusable_scores(self.root, self.write(), 'cpu')
        self.report['rows'].pop()
        self.manifest['rows'].append({**self.frame, 'frame': 1})
        metrics = {**self.report['rows'][0]['metrics'], 'lpips': .3}
        self.report['rows'].append({**self.frame, 'frame': 1, 'metrics': metrics})
        with self.assertRaises(ValueError):
            reusable_scores(self.root, self.write(), 'cpu')


if __name__ == '__main__':
    unittest.main()
