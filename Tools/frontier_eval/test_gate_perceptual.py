import unittest
from gate_perceptual import evaluate


def rows(spec):
    out = []
    for variant, sources in spec.items():
        for source, m in sources.items():
            for frame in (0, 4):
                out.append({'variant': variant, 'source_id': source, 'sequence_id': source + '-h264', 'frame': frame,
                            'metrics': dict(lpips=m[0], dists=m[1], fine_correlation=m[2], detail_energy=m[3], psnr_y=30.0)})
    return out


BASE = {'lanczos': {'a': (0.40, 0.15, 0.30, 0.40), 'b': (0.30, 0.12, 0.60, 0.55)},
        'shipping': {'a': (0.35, 0.14, 0.35, 0.50), 'b': (0.25, 0.11, 0.66, 0.65)}}


class PerceptualGateTests(unittest.TestCase):
    def test_synthesis_above_anchor_passes(self):
        spec = dict(BASE, cand={'a': (0.30, 0.12, 0.33, 0.60), 'b': (0.20, 0.09, 0.62, 0.85)})
        result = evaluate(rows(spec), 'cand')
        self.assertTrue(result['perceptual_gate_pass'], result['failure_reasons'])
        self.assertLess(result['per_source_changes']['a']['fine_correlation_vs_shipping'], 0)

    def test_below_anchor_fails(self):
        spec = dict(BASE, cand={'a': (0.30, 0.12, 0.29, 0.60), 'b': (0.20, 0.09, 0.62, 0.85)})
        result = evaluate(rows(spec), 'cand')
        self.assertIn('a: fine correlation below the lanczos anchor', result['failure_reasons'])

    def test_embellishment_cap_fails(self):
        spec = dict(BASE, cand={'a': (0.30, 0.12, 0.33, 1.2), 'b': (0.20, 0.09, 0.62, 0.85)})
        result = evaluate(rows(spec), 'cand')
        self.assertIn('a: fine-band energy exceeds the embellishment cap', result['failure_reasons'])

    def test_perceptual_regression_and_minimum_fail(self):
        spec = dict(BASE, cand={'a': (0.36, 0.12, 0.33, 0.60), 'b': (0.245, 0.109, 0.62, 0.85)})
        result = evaluate(rows(spec), 'cand')
        self.assertIn('a: lpips regression exceeds limit', result['failure_reasons'])
        self.assertIn('lpips: aggregate improvement below minimum', result['failure_reasons'])

    def test_coverage_mismatch_rejected(self):
        spec = dict(BASE, cand={'a': (0.30, 0.12, 0.33, 0.60)})
        with self.assertRaises(ValueError):
            evaluate(rows(spec), 'cand')


if __name__ == '__main__':
    unittest.main()
