import copy
import unittest
from gate_controller_candidate import evaluate,MANIFESTS,SHIPPING


class ControllerGateTests(unittest.TestCase):
    def setUp(self):
        header=dict(complete=True,split='development-validation',manifest_sha256=MANIFESTS['development'][0],
                    sequence_manifest_sha256='sources',presentation_adapters=['shipping'])
        controls=[dict(source_id=f'source{i//16}',sequence_id=f'sequence{i//16}',frame=i,
                       variant=label,metrics=dict(lpips=1.,dists=1.,fine_correlation=.9))
                  for label in ('shipping','lanczos') for i in range(48)]
        candidates=[dict(r,variant='coarse',metrics=dict(lpips=.7,dists=.7,fine_correlation=.9)) for r in controls[:48]]
        self.control=dict(header,checkpoint_sha256={'shipping':SHIPPING},rows=controls)
        self.report=dict(header,checkpoint_sha256={'shipping':SHIPPING,'coarse':'candidate'},rows=copy.deepcopy(controls)+candidates)

    def test_matched_full_coverage_can_pass_development_only(self):
        result=evaluate(self.report,self.control,'development','candidate')
        self.assertTrue(result['spatial_gate_pass'])
        self.assertFalse(result['promotion_authorized'])

    def test_changed_control_or_checkpoint_is_rejected(self):
        report=copy.deepcopy(self.report);report['rows'][0]['metrics']['lpips']+=.001
        with self.assertRaises(ValueError):evaluate(report,self.control,'development','candidate')
        with self.assertRaises(ValueError):evaluate(self.report,self.control,'development','other')

    def test_duplicate_candidate_is_rejected(self):
        report=copy.deepcopy(self.report);report['rows'].append(report['rows'][-1])
        with self.assertRaises(ValueError):evaluate(report,self.control,'development','candidate')

    def test_perceptual_gain_does_not_override_detail_loss(self):
        report=copy.deepcopy(self.report)
        for row in report['rows']:
            if row['variant']=='coarse' and row['source_id']=='source0':row['metrics']['fine_correlation']=.88
        self.assertFalse(evaluate(report,self.control,'development','candidate')['spatial_gate_pass'])


if __name__=='__main__':unittest.main()
