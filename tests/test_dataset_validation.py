"""Canonical metadata and array acceptance; synthetic CPU inputs only."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'torch'))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'course/liftcube/src'))
    from starter.data.schema import EpisodeStep
    from starter.data.validator import DatasetValidationError, validate_arrays, validate_step
    from starter.interfaces.action import ActionSpec
    from starter.interfaces.observation import ObservationSpec


@unittest.skipUnless(AVAILABLE, 'Requires NumPy/Torch')
class DatasetValidationTests(unittest.TestCase):
    def payload(self):
        obs = ObservationSpec()
        return dict(image=np.zeros((2, *obs.image_shape), np.uint8),
                    proprio=np.zeros((2, obs.proprio_dim), np.float32),
                    action=np.zeros((2, 6), np.float32), timestamp=np.array([0., 1/30]),
                    episode_id=np.array([0, 0]), step_id=np.array([0, 1]),
                    terminated=np.array([False, True]), truncated=np.array([False, False]),
                    success=np.array([-1, -1]))

    def check(self, payload):
        return validate_arrays(payload, ObservationSpec(), ActionSpec())

    def test_valid_unknown_success_and_input_preservation(self):
        payload = self.payload()
        original = {k: v.copy() for k, v in payload.items()}
        self.assertEqual(self.check(payload), {'num_steps': 2, 'num_episodes': 1})
        for key in payload:
            np.testing.assert_array_equal(payload[key], original[key])

    def test_misaligned_metadata_rejected(self):
        for key in ('timestamp', 'episode_id', 'step_id', 'terminated', 'truncated', 'success'):
            with self.subTest(key=key):
                payload = self.payload()
                payload[key] = payload[key][:1]
                with self.assertRaises(DatasetValidationError):
                    self.check(payload)

    def test_invalid_values_and_empty_inputs_rejected(self):
        bad = {'timestamp': [np.nan, -1], 'episode_id': [0.5, -1],
               'step_id': [True, -1], 'terminated': [2, np.nan], 'truncated': [2], 'success': [2]}
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    payload = self.payload()
                    payload[key] = np.full(2, value)
                    with self.assertRaises(DatasetValidationError):
                        self.check(payload)
        payload = self.payload()
        payload['proprio'][0, 0] = np.inf
        with self.assertRaises(DatasetValidationError):
            self.check(payload)
        payload = self.payload()
        payload['image'] = payload['image'].astype(np.float32)
        with self.assertRaises(DatasetValidationError):
            self.check(payload)
        with self.assertRaises(DatasetValidationError):
            self.check({k: v[:0] for k, v in self.payload().items()})
        payload = self.payload()
        payload['action'] = np.array(0.)
        with self.assertRaises(DatasetValidationError):
            self.check(payload)

    def test_step_metadata_checked_without_clipping_labels(self):
        p = self.payload()
        step = EpisodeStep(p['image'][0], p['proprio'][0], p['action'][0], 0., 0, 0, False, False)
        validate_step(step, ObservationSpec(), ActionSpec())
        step.timestamp = np.nan
        with self.assertRaises(DatasetValidationError):
            validate_step(step, ObservationSpec(), ActionSpec())
        step.timestamp = 0.
        step.action[3] = 150
        with self.assertRaises(DatasetValidationError):
            validate_step(step, ObservationSpec(), ActionSpec())
        self.assertEqual(step.action[3], 150)
