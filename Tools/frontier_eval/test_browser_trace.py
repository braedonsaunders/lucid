import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_browser_trace import analyze


class BrowserTraceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{'at': at, 'session': 'session', 'seq': i, 'latencyMilliseconds': 20}
                     for i, at in enumerate(range(0, 10021, 20))]
        self.report = {'complete': True, 'runs': [{
            'variant': 'shipping', 'installation': {'installed': {'id': 'extension'}},
            'setup': {'status': {'activeSession': 'session'}},
            'measurement': {'startedEpoch': 10, 'endedEpoch': 10010},
            'off': {'enabled': False, 'enhancing': False}}]}

    def save(self):
        raw = gzip.compress(json.dumps({'surfaces': [{
            'url': 'chrome-extension://extension/surface.html#session', 'samples': self.rows}]}).encode())
        (self.root/'trace.gz').write_bytes(raw)
        self.report['runs'][0]['measurement']['presentation_trace'] = {
            'file': 'trace.gz', 'sha256': hashlib.sha256(raw).hexdigest()}
        path = self.root/'report.json'
        path.write_text(json.dumps(self.report))
        return path

    def test_counts_acknowledgments_over_full_window(self):
        result = analyze(self.save(), minimum_seconds=10, minimum_fps=50)
        self.assertTrue(result['pass'])
        self.assertEqual(result['runs'][0]['actual_draws'], 500)
        self.assertEqual(result['runs'][0]['actual_fps'], 50)
        self.assertFalse(analyze(self.save())['pass'])

    def test_pooled_tail_and_off_fail(self):
        for row in self.rows[50:100]:
            row['latencyMilliseconds'] = 80
        self.report['runs'][0]['off']['enabled'] = True
        result = analyze(self.save(), minimum_seconds=10, minimum_fps=50)
        self.assertFalse(result['pass'])
        self.assertEqual(result['runs'][0]['p95_ms'], 80)
        self.assertFalse(result['runs'][0]['gates']['off'])

    def test_rejects_duplicate_foreign_and_nonfinite_samples(self):
        for key, value in [('seq', 0), ('session', 'foreign'), ('latencyMilliseconds', float('nan'))]:
            with self.subTest(key=key):
                old = self.rows[1][key]
                self.rows[1][key] = value
                with self.assertRaises(ValueError):
                    analyze(self.save())
                self.rows[1][key] = old

    def test_rejects_corrupt_or_truncated_evidence(self):
        path = self.save()
        (self.root/'trace.gz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest'):
            analyze(path)
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'bracket'):
            analyze(self.save())
        self.report['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete'):
            analyze(self.save())

    def test_repeated_source_frames_cannot_pass_source_cadence(self):
        for row in self.rows:
            row['sourceTimestamp'] = (row['seq'] // 2) * 40000
        result = analyze(self.save(), minimum_seconds=10, minimum_fps=50, require_source_timestamps=True)
        self.assertTrue(result['runs'][0]['gates']['cadence'])
        self.assertFalse(result['pass'])
        self.assertEqual(result['runs'][0]['source_cadence']['changed_timestamp_fps'], 25)
        self.assertEqual(result['runs'][0]['source_cadence']['repeated_timestamp_draws'], 250)

    def test_looping_source_timestamps_and_missing_evidence(self):
        with self.assertRaisesRegex(ValueError, 'source timestamps'):
            analyze(self.save(), require_source_timestamps=True)
        for row in self.rows:
            row['sourceTimestamp'] = (row['seq'] % 100) * 20000
        result = analyze(self.save(), minimum_seconds=10, minimum_fps=50, require_source_timestamps=True)
        self.assertTrue(result['pass'])
        self.assertEqual(result['runs'][0]['source_cadence']['backward_timestamp_changes'], 5)
        self.rows[8]['sourceTimestamp'] = None
        with self.assertRaisesRegex(ValueError, 'source timestamps'):
            analyze(self.save(), require_source_timestamps=True)


if __name__ == '__main__':
    unittest.main()
