import unittest

import numpy as np
from PIL import Image

from flow_temporal import ReferenceMotion, correspondence, score_pair


class FlowTemporalTests(unittest.TestCase):
    def test_estimated_reference_motion_recovers_translation(self):
        previous = np.random.default_rng(35).integers(20, 235, (96, 128), dtype=np.uint8)
        current = np.roll(previous, 2, axis=1)
        frames = [Image.fromarray(frame) for frame in (previous, current)]
        motion = ReferenceMotion(frames, max_width=128)
        backward = motion.contexts[0]['backward'][16:-16, 16:-16]
        self.assertAlmostEqual(float(np.median(backward[..., 0])), -2, delta=0.2)
        self.assertGreater(motion.score(frames)['flow_nonoccluded_fraction'], 0.8)
        self.assertEqual(motion.score(frames)['flow_residual_l1'], 0)

    def context(self):
        previous = np.random.default_rng(12).uniform(30, 220, (40, 48)).astype(np.float32)
        current = np.roll(previous, 2, axis=1)
        forward = np.zeros((40, 48, 2), np.float32); forward[..., 0] = 2
        backward = -forward
        return previous, current, correspondence(previous, current, forward, backward)

    def test_exact_translation_is_zero_and_disoccluded_border_is_excluded(self):
        previous, current, context = self.context()
        self.assertFalse(context['visible'][:, :2].any())
        self.assertGreater(context['visible'].mean(), 0.8)
        result = score_pair(previous, current, context)
        self.assertEqual(result['flow_residual_l1'], 0)
        self.assertEqual(result['flow_motion_residual_l1'], 0)

    def test_flicker_remains_visible_and_output_cannot_change_mask(self):
        previous, current, context = self.context()
        before = context['visible'].copy()
        result = score_pair(previous, current+7, context)
        self.assertAlmostEqual(result['flow_residual_l1'], 7, places=5)
        np.testing.assert_array_equal(before, context['visible'])

    def test_inconsistent_flow_has_no_false_perfect_score(self):
        previous, current, _ = self.context()
        forward = np.zeros((40, 48, 2), np.float32); forward[..., 0] = 8
        backward = np.zeros_like(forward)
        context = correspondence(previous, current, forward, backward)
        result = score_pair(previous, current, context)
        self.assertEqual(result['flow_nonoccluded_fraction'], 0)
        self.assertIsNone(result['flow_residual_l1'])


if __name__ == '__main__':
    unittest.main()
