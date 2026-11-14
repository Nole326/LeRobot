"""Evo-1 adapter contracts: protocol match, request whitelist, action shape.

Synthetic CPU inputs only; no model weights, CUDA, camera or simulator.
"""
import json
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE = importlib.util.find_spec('numpy') is not None
ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / 'course/pickorange/protocols'
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(ROOT / 'course/pickorange'))
    from common import load_protocol, timing, validate_protocol, validate_request
    from service_evo1 import check_action_chunk


@unittest.skipUnless(AVAILABLE, 'Requires NumPy')
class Evo1ProtocolTests(unittest.TestCase):
    def test_evo1_protocol_is_frozen_and_whole_chunk(self):
        cfg = load_protocol(PROTOCOLS / 'evo1-native-assisted-30hz.json')
        self.assertEqual(cfg['name'], 'pickorange-evo1-native-assisted-30hz-v1')
        self.assertEqual((cfg['action_hz'], cfg['chunk_length']), (30, 50))
        self.assertEqual((cfg['execute_steps'], cfg['denoising_steps']), (50, 32))
        self.assertEqual(cfg['instruction'],
                         'Pick three oranges and place them on the plate, then return the arm to rest.')
        self.assertEqual(timing(cfg), (2, 100, 3600))

    def test_other_protocols_do_not_masquerade_as_evo1(self):
        for name in ('smolvla-native-assisted-30hz.json', 'pi05-native-assisted-30hz.json',
                     'groot-native-assisted.json'):
            self.assertNotEqual(load_protocol(PROTOCOLS / name)['name'],
                                'pickorange-evo1-native-assisted-30hz-v1')

    def test_protocol_value_drift_is_rejected(self):
        cfg = json.loads((PROTOCOLS / 'evo1-native-assisted-30hz.json').read_text(encoding='utf-8'))
        for key, wrong in (('execute_steps', 16), ('denoising_steps', 10), ('action_hz', 60),
                           ('chunk_length', 16), ('episode_seconds', 120)):
            broken = dict(cfg, **{key: wrong})
            with self.assertRaises(ValueError):
                validate_protocol(broken)


@unittest.skipUnless(AVAILABLE, 'Requires NumPy')
class Evo1RequestTests(unittest.TestCase):
    def payload(self):
        return {'front': np.zeros((1, 480, 640, 3), np.uint8),
                'wrist': np.zeros((1, 480, 640, 3), np.uint8),
                'joint_motor': np.zeros((1, 6), np.float32)}

    def test_real_shape_passes(self):
        q = validate_request(self.payload())
        self.assertEqual(q.shape, (1, 6))
        self.assertEqual(q.dtype, np.float64)

    def test_extra_fields_and_unknown_state_are_rejected(self):
        data = self.payload()
        data['object_pos_w'] = np.zeros((1, 4, 3), np.float32)
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_non_finite_joints_are_rejected(self):
        data = self.payload()
        data['joint_motor'][0, 3] = np.nan
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_stacked_or_float_views_are_rejected(self):
        stacked = self.payload()
        stacked['front'] = np.zeros((2, 480, 640, 3), np.uint8)
        with self.assertRaises(ValueError):
            validate_request(stacked)
        floating = self.payload()
        floating['wrist'] = np.zeros((1, 480, 640, 3), np.float32)
        with self.assertRaises(ValueError):
            validate_request(floating)


@unittest.skipUnless(AVAILABLE, 'Requires NumPy')
class Evo1ActionShapeTests(unittest.TestCase):
    def test_whole_chunk_passes(self):
        chunk = check_action_chunk(np.zeros((50, 6), np.float32))
        self.assertEqual(chunk.shape, (50, 6))
        self.assertTrue(np.isfinite(chunk).all())

    def test_transposed_or_ragged_chunks_are_rejected(self):
        for shape in ((6, 50), (100, 6), (50, 5), (50,)):
            with self.assertRaises(ValueError):
                check_action_chunk(np.zeros(shape, np.float32))

    def test_non_finite_chunk_is_rejected(self):
        chunk = np.zeros((50, 6), np.float32)
        chunk[7, 2] = np.inf
        with self.assertRaises(ValueError):
            check_action_chunk(chunk)


if __name__ == '__main__':
    unittest.main()
