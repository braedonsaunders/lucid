import unittest
from unittest.mock import patch

import torch

from evaluate_taa_motion import policy_motion
from native_stages import motion_blocks, preprocess_sequence


class TAAMotionPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.previous = ((torch.arange(56).float() + 77) / 255)[None, None, None].repeat(1, 1, 48, 1)

    def test_low_error_translation_keeps_a_strong_match(self):
        current = torch.roll(self.previous, 4, -1)
        legacy = policy_motion(current, self.previous, 'taa')
        torch.testing.assert_close(legacy, motion_blocks(current, self.previous), rtol=0, atol=0)
        self.assertTrue((legacy[:, :2, 1:-1, 1:-1] == 0).all())
        field = policy_motion(current, self.previous, 'gain_only')[:, :, 1:-1, 1:-1]
        self.assertTrue((field[:, 0] == -4).all())
        self.assertTrue((field[:, 1] == 0).all())
        self.assertLess(float(field[:, 3].abs().max()), 1e-6)

    def test_weak_gain_is_still_rejected(self):
        current = torch.roll(self.previous, 1, -1)
        legacy = policy_motion(current, self.previous, 'taa')[:, :, 1:-1, 1:-1]
        raw = policy_motion(current, self.previous, 'search')[:, :, 1:-1, 1:-1]
        field = policy_motion(current, self.previous, 'gain_only')[:, :, 1:-1, 1:-1]
        self.assertTrue((raw[:, 0] == -1).all())
        torch.testing.assert_close(field, legacy, rtol=0, atol=0)

    def test_explicit_source_search_matches_prior_replay_without_changing_legacy(self):
        torch.manual_seed(23)
        previous = .3 + .04 * torch.rand(1, 3, 32, 48)
        frames = torch.stack([torch.roll(previous, t * 4, -1) for t in range(3)], 1)
        legacy = preprocess_sequence(frames)
        torch.testing.assert_close(legacy, preprocess_sequence(frames, motion_policy='taa'), rtol=0, atol=0)
        with patch('native_stages.motion_blocks',
                   lambda a, b: motion_blocks(a, b, stationary_override=False)):
            prior_replay = preprocess_sequence(frames)
        current = preprocess_sequence(frames, motion_policy='search')
        torch.testing.assert_close(current, prior_replay, rtol=0, atol=0)
        torch.testing.assert_close(current[:, 0], legacy[:, 0], rtol=0, atol=0)
        self.assertGreater(float((current[:, 1:] - legacy[:, 1:]).abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'motion policy'):
            preprocess_sequence(frames, motion_policy='invalid')


if __name__ == '__main__':
    unittest.main()
