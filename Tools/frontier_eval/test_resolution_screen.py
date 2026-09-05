import copy
import unittest
from gate_resolution_screen import evaluate


class ResolutionScreenTest(unittest.TestCase):
    def setUp(self):
        self.spec={'split':'development-validation','frames_manifest_sha256':'frames',
            'candidate':'folded','candidate_sha256':'fixed','shipping_sha256':'shipping',
            'sources':['scene'],'codecs':['h264'],'bitrates':[1000],'frames':[0,4],
            'interpolation':['lanczos','bicubic','bilinear'],
            'gate':{'source_balanced_lpips_improvement_min':.03,'source_balanced_dists_improvement_min':.03,
                'per_source_perceptual_regression_max':.02,'per_source_fine_correlation_drop_max':.01}}
        self.manifest={'sequence_manifest_sha256':'sequences','frames':[
            {'source_id':'scene','sequence_id':'scene-h264-1000','frame':frame,'side':side}
            for frame in [0,4] for side in ['reference','degraded']]}
        self.report={'complete':True,'split':'development-validation','manifest_sha256':'frames',
            'sequence_manifest_sha256':'sequences','checkpoint_sha256':{'folded':'fixed','shipping':'shipping'},
            'interpolation':['lanczos','bicubic','bilinear'],'presentation_adapters':['shipping'],
            'rows':[{'source_id':'scene','sequence_id':'scene-h264-1000','frame':frame,'variant':label,
                'metrics':{'lpips':.9 if label in ('folded','shipping') else 1.,
                    'dists':.9 if label in ('folded','shipping') else 1.,'fine_correlation':.5}}
                for frame in [0,4] for label in ['folded','shipping','lanczos','bicubic','bilinear']]}

    def test_all_floors_required_without_claiming_shipping_improvement(self):
        result=evaluate(self.report,self.spec,self.manifest)
        self.assertTrue(result['coverage_spatial_screen_pass'])
        self.assertFalse(result['comparisons']['shipping']['spatial_gate_pass'])
        for row in self.report['rows']:
            if row['variant']=='bicubic':row['metrics']['lpips']=.8
        result=evaluate(self.report,self.spec,self.manifest)
        self.assertFalse(result['coverage_spatial_screen_pass'])
        self.assertTrue(result['comparisons']['lanczos']['spatial_gate_pass'])

    def test_changed_evidence_and_missing_samples_fail_closed(self):
        for key,value in [('complete',False),('manifest_sha256','changed'),
                          ('checkpoint_sha256',{'folded':'other','shipping':'shipping'}),
                          ('interpolation',['lanczos']),('presentation_adapters',[])]:
            changed=copy.deepcopy(self.report);changed[key]=value
            with self.assertRaises(ValueError):evaluate(changed,self.spec,self.manifest)
        changed=copy.deepcopy(self.report);changed['rows'].pop()
        with self.assertRaises(ValueError):evaluate(changed,self.spec,self.manifest)
        changed=copy.deepcopy(self.manifest);changed['frames'].pop()
        with self.assertRaises(ValueError):evaluate(self.report,self.spec,changed)

    def test_per_source_detail_guard_survives_perceptual_gain(self):
        for row in self.report['rows']:
            if row['variant']=='folded':row['metrics']['fine_correlation']=.48
        self.assertFalse(evaluate(self.report,self.spec,self.manifest)['coverage_spatial_screen_pass'])


if __name__=='__main__':unittest.main()
