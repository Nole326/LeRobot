"""CPU interface checks with a raw-env double; never start Isaac or move hardware."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'torch'))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'course/liftcube/src'))
    from starter.envs.lift_cube import LiftCubeEnvAdapter
    from starter.interfaces.action import motor_to_radians, radians_to_motor


@unittest.skipUnless(AVAILABLE, 'Requires the course NumPy/Torch environment')
class SO101AdapterTests(unittest.TestCase):
    def test_owned_environment_disables_implicit_demo_recorder(self):
        raw = self.raw()
        cfg = SimpleNamespace(recorders=object(), scene=SimpleNamespace(num_envs=4),
            sim=SimpleNamespace(device='cpu', dt=1/60), decimation=1,
            actions=SimpleNamespace(arm_action=SimpleNamespace(), gripper_action=SimpleNamespace()),
            use_teleop_device=lambda device: None)
        class Manager:
            def __new__(cls, **kwargs):
                self.assertIs(kwargs['cfg'], cfg)
                return raw
        with patch('starter.envs.container.isaac_runtime_ready', return_value=True), \
             patch('starter.envs.lift_cube._launch_sim_app'), \
             patch('starter.envs.lift_cube._import_lift_cube', return_value=(lambda: cfg, Manager)):
            env = LiftCubeEnvAdapter(device='cuda:0')
        self.assertIsNone(cfg.recorders)
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertEqual(cfg.episode_length_s, 640/60)
        self.assertEqual(env.action_spec.frequency_hz, 60)

    def raw(self):
        class Raw:
            device = 'cpu'
            step_dt = 1 / 60

            def __init__(self):
                self.policy = {'joint_pos': np.zeros((1, 6), np.float32),
                    'front': np.full((1, 64, 64, 3), 80, np.uint8),
                    'joint_vel': np.full((1, 6), 999), 'ee_frame_state': np.full((1, 3), 999)}
                self.calls = 0

            def reset(self, **kwargs):
                return {'policy': self.policy}, {}

            def step(self, action):
                self.calls += 1
                self.sent = action.detach().cpu().numpy().copy()
                return {'policy': self.policy}, 0, False, False, {}
        return Raw()

    def test_command_request_applied_and_measured_remain_distinct(self):
        raw = self.raw()
        env = LiftCubeEnvAdapter(raw_env=raw, device='cpu')
        env.reset()
        self.assertEqual(env.action_spec.frequency_hz, 60)
        requested = np.array([120, -120, 20, 30, 40, 150], np.float32)
        applied = np.array([100, -100, 20, 30, 40, 100], np.float32)
        obs, _, _, _, info = env.step(requested)
        np.testing.assert_array_equal(info['command']['requested_motor'], requested)
        np.testing.assert_array_equal(info['command']['applied_motor'], applied)
        np.testing.assert_allclose(raw.sent[0], motor_to_radians(applied))
        np.testing.assert_allclose(obs.proprio[:6], radians_to_motor(np.zeros(6)))
        np.testing.assert_array_equal(obs.proprio[12:18], applied)
        self.assertEqual(info['command']['clipped'], [True, True, False, False, False, True])
        self.assertFalse(obs.proprio[6:12].any() or obs.proprio[18:].any())
        self.assertFalse(info['success'])
        requested[:] = 0
        self.assertEqual(info['command']['requested_motor'][0], 120)

    def test_feedback_missing_malformed_or_nonfinite_is_not_zero_padding(self):
        for value in [None, [0] * 5, [0] * 7, [[0] * 3] * 2, [0, 0, float('nan'), 0, 0, 0]]:
            with self.subTest(value=value):
                raw = self.raw()
                if value is None:
                    del raw.policy['joint_pos']
                else:
                    raw.policy['joint_pos'] = value
                with self.assertRaises(ValueError):
                    LiftCubeEnvAdapter(raw_env=raw, device='cpu').reset()

    def test_motor_alias_requires_explicit_units_without_double_conversion(self):
        raw = self.raw()
        del raw.policy['joint_pos']
        raw.policy['observation.state'] = np.array([[10, 20, 30, 40, 50, 60]], np.float32)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            LiftCubeEnvAdapter(raw_env=raw, device='cpu').reset()
        obs, _ = LiftCubeEnvAdapter(raw_env=raw, device='cpu', raw_joint_units='motor').reset()
        np.testing.assert_array_equal(obs.proprio[:6], raw.policy['observation.state'][0])

    def test_invalid_command_rejected_before_raw_step(self):
        raw = self.raw()
        env = LiftCubeEnvAdapter(raw_env=raw, device='cpu')
        for command in [np.zeros(5), np.zeros((2, 6)), np.full(6, np.nan)]:
            with self.assertRaises(ValueError):
                env.step(command)
        self.assertEqual(raw.calls, 0)

    def test_invalid_camera_is_not_silently_converted(self):
        for image in [np.full((64, 64, 3), np.nan), np.zeros((2, 64, 64, 3)), np.zeros((64, 64))]:
            raw = self.raw()
            raw.policy['front'] = image
            with self.assertRaises(ValueError):
                LiftCubeEnvAdapter(raw_env=raw, device='cpu').reset()


if __name__ == '__main__':
    unittest.main()
