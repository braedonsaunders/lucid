import copy
import unittest
from gate_frozen_holdout import evaluate


class HoldoutGateTests(unittest.TestCase):
    def fixture(self):
        spec = {'sources':[{'id':s,'frames':1} for s in ['easy','hard']],
            'conditions':{'codecs':['h264'],'bitrates':[100],'spatial_stride':1},
            'gate':{'source_balanced_lpips_improvement_min':.03,'source_balanced_dists_improvement_min':.03,
                    'per_source_perceptual_regression_max':.02,'per_source_fine_correlation_drop_max':.01}}
        frozen = {'candidate_sha256':'c','shipping_sha256':'s'}
        report = {'complete':True,'split':'quality-holdout','checkpoint_sha256':{'blend80':'c','shipping':'s'},
                  'presentation_adapters':['shipping'],'rows':[]}
        manifest = {'stride':1,'split':'quality-holdout','frames':[]}
        for source in ['easy','hard']:
            base = {'source_id':source,'sequence_id':source+'-h264-100','frame':0}
            for side in ['reference','degraded']:
                manifest['frames'].append(dict(base,side=side))
            for label,value in [('lanczos',.4),('shipping',.3),('blend80',.28)]:
                report['rows'].append(dict(base,variant=label,metrics={'lpips':value,'dists':value,'fine_correlation':.6}))
        return report,spec,frozen,manifest

    def test_aggregate_win_cannot_hide_one_source_losing_detail(self):
        args=self.fixture()
        self.assertTrue(evaluate(*args)['spatial_gate_pass'])
        args[0]['rows'][-1]['metrics']['fine_correlation']=.58
        result=evaluate(*args)
        self.assertFalse(result['spatial_gate_pass'])
        self.assertIn('hard: fine correlation drop exceeds limit',result['failure_reasons'])

    def test_incomplete_mismatched_or_nonfinite_evidence_rejected(self):
        mutations=[lambda r:r.update(complete=False),lambda r:r['rows'].pop(),
            lambda r:r['rows'].append(copy.deepcopy(r['rows'][0])),
            lambda r:r['checkpoint_sha256'].update(blend80='retuned'),
            lambda r:r['rows'][0]['metrics'].update(lpips=float('nan'))]
        for mutate in mutations:
            args=self.fixture();mutate(args[0])
            with self.assertRaises(ValueError): evaluate(*args)


if __name__=='__main__': unittest.main()
