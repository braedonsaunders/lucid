import copy
import unittest
from train_activation_control import validate_coarse_native


class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.report=dict(complete=True,direct_architecture='CoarseSpatialActivationControl',
                         checkpoint_sha256='anchor',samples=40,
                         direct_model_sources={'architectures/spatial_activation_control.py':'source'},
                         rows=[dict(input_size=size,compute_units='CPU_AND_GPU',
                                    timings={'direct2x_trained':dict(mean_ms=mean,p95_ms=p95)},
                                    correctness={'direct2x_trained':dict(max_rgb=1,mean_rgb=.3)})
                               for size,mean,p95 in [('640x360',7,8),('1280x720',24,26)]])

    def test_complete_correct_fast_matching_graph_is_admitted(self):
        validate_coarse_native(self.report,'anchor','source')

    def test_missing_or_substituted_graph_is_rejected(self):
        for key,value in [('complete',False),('direct_architecture','SpatialActivationControl'),
                          ('checkpoint_sha256','other'),('direct_model_sources',{}),('samples',1),('rows',self.report['rows'][:1])]:
            with self.subTest(key=key):
                report=copy.deepcopy(self.report);report[key]=value
                with self.assertRaises(ValueError):validate_coarse_native(report,'anchor','source')

    def test_slow_incorrect_or_nonfinite_720p_is_rejected(self):
        for section,key,value in [('timings','mean_ms',26.123),('timings','p95_ms',31),
                                  ('timings','mean_ms',float('nan')),('correctness','max_rgb',4)]:
            with self.subTest(section=section,key=key,value=value):
                report=copy.deepcopy(self.report)
                report['rows'][1][section]['direct2x_trained'][key]=value
                with self.assertRaises(ValueError):validate_coarse_native(report,'anchor','source')


if __name__=='__main__':unittest.main()
