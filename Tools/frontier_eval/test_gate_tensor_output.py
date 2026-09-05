import copy
import unittest

from gate_tensor_output import evaluate, SHIPPING, CORPORA
FRAMES, SEQUENCES, _, _ = CORPORA["regression960"]


class TensorOutputGateTests(unittest.TestCase):
    def setUp(self):
        self.sequences = [{'source_id': str(s), 'id': f'{s}-{c}', 'frames': 300}
                          for s in range(8) for c in range(4)]
        rows = [{'source_id': s['source_id'], 'sequence_id': s['id'], 'frame': f,
                 'variant': label, 'metrics': dict(lpips=.2, dists=.1, fine_correlation=.5)}
                for s in self.sequences for f in range(0, 300, 10)
                for label in ('shipping4x', 'candidate')]
        self.report = dict(complete=True, split='development-presentation', rows=rows,
                           checkpoint_sha256=dict(shipping4x=SHIPPING, candidate=SHIPPING))
        self.manifest = dict(copy.deepcopy(self.report), source_manifest_sha256=SEQUENCES,
                             frozen_frames_manifest_sha256=FRAMES,
                             presentation_corpus='regression960', preserve_display_gain=False, output_storage="tensor4x_fp32")
        self.config = dict(candidate_sha256=SHIPPING, radius=4,
                           preserve_display_gain=False, output_storage="tensor4x_fp32", frames_manifest_sha256=FRAMES, tuning={"sharpness":.75})

    def check(self):
        return evaluate(self.report, self.manifest, self.config, self.sequences)

    def test_identity_passes_without_promoting(self):
        result = self.check()
        self.assertTrue(result['spatial_gate_pass'])
        self.assertFalse(result['promotion_authorized'])

    def test_source_loss_cannot_hide_in_average(self):
        for row in self.report['rows']:
            if row['variant'] == 'candidate':
                row['metrics']['lpips'] = .202 if row['source_id'] == '0' else .1
        self.assertFalse(self.check()['spatial_gate_pass'])

    def test_missing_duplicate_nonfinite_or_wrong_identity_fails(self):
        original = copy.deepcopy(self.report)
        for mutation in ('missing', 'duplicate', 'nonfinite', 'weights'):
            self.report = copy.deepcopy(original)
            if mutation == 'missing': self.report['rows'].pop()
            if mutation == 'duplicate': self.report['rows'].append(self.report['rows'][0])
            if mutation == 'nonfinite': self.report['rows'][0]['metrics']['lpips'] = float('nan')
            if mutation == 'weights': self.report['checkpoint_sha256']['candidate'] = 'wrong'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check()

    def test_wrong_gain_or_corpus_fails(self):
        self.manifest['preserve_display_gain'] = True
        with self.assertRaises(ValueError): self.check()
        self.manifest['preserve_display_gain'] = False
        self.manifest['frozen_frames_manifest_sha256'] = 'wrong'
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__': unittest.main()
