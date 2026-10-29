"""Synthetic CPU checks of the opt-in EE development contract, never Isaac/hardware."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy'))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'course/liftcube/src'))
    from starter.envs.planar_ee import PlanarEEController, PlanarEEEnv
    from starter.interfaces.action import ActionSpec
    from starter.envs.factory import make_env


@unittest.skipUnless(AVAILABLE, 'Requires NumPy/SciPy; no simulator')
class PlanarEETests(unittest.TestCase):
    def kin(self, movable=True):
        class SyntheticChain:
            def forward_kinematics(self, degrees):
                pose = np.eye(4)
                pose[:3, 3] = [degrees[0]*.001 if movable else 0,
                               degrees[1]*.001 if movable else 0, .1]
                angle = np.deg2rad(degrees[4])
                pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                return pose
        return SyntheticChain()

    def controller(self, movable=True, **changes):
        kwargs = dict(table_to_base=np.eye(4), xy_bounds=[[-.09, -.09], [.09, .09]],
            fixed_height_m=.1, tool_axis_local=[0,0,1], tool_axis_table=[0,0,1])
        kwargs.update(changes)
        return PlanarEEController(self.kin(movable), **kwargs)

    def initial(self):
        return np.array([0.,0.,0.,0.,0.,50.])

    def env(self, terminated=False):
        test = self
        class RawDouble:
            action_spec = ActionSpec(frequency_hz=60.)
            observation_spec = SimpleNamespace(control_hz=60.)
            name = 'synthetic_not_isaac'
            def __init__(self):
                self.sent = []
            def reset(self, **kwargs):
                self.q = test.initial()
                return SimpleNamespace(proprio=self.q.copy()), {}
            def step(self, motor):
                self.sent.append(motor.copy()); self.q = motor.copy()
                return SimpleNamespace(proprio=self.q.copy()), 0., terminated, False, {'command': 'native preserved'}
            def close(self):
                self.closed = True
        return RawDouble()

    def test_action_units_fixed_plane_axis_and_bounded_commands(self):
        controller = self.controller(); measured = self.initial(); controller.reset(measured)
        self.assertEqual(controller.action_spec.dim, 3)
        self.assertEqual(controller.action_spec.units, ('m_table', 'm_table', 'motor_0_100'))
        previous = measured.copy()
        for _ in range(40):
            sent = controller.command([.01, .006, 60], measured)
            self.assertLessEqual(np.max(np.abs(sent-previous)), 1.00001)
            self.assertLessEqual(np.linalg.norm(controller.pose(sent)[:3,3]-controller.pose(previous)[:3,3]), .00400001)
            self.assertAlmostEqual(controller.pose(sent)[2,3], .1)
            np.testing.assert_allclose(controller.pose(sent)[:3,:3] @ controller.local_axis, [0,0,1])
            measured, previous = sent, sent.copy()
        np.testing.assert_allclose(controller.pose(sent)[:2,3], [.01,.006], atol=.003)

    def test_requests_are_not_clipped_or_guessed(self):
        controller = self.controller(); controller.reset(self.initial())
        for request in ([1,0,50], [0,0,101], [0,0,float('nan')], [0,0], [[0,0,50]]):
            with self.assertRaises(ValueError):
                controller.command(request, self.initial())
        self.assertIsNone(controller.failure_reason)  # Bad API arguments are rejected before dispatch.

    def test_unreachable_target_latches_hold_until_reset(self):
        controller = self.controller(movable=False); controller.reset(self.initial())
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            controller.command([.02,0,50], self.initial())
        np.testing.assert_array_equal(controller.command([0,0,50], self.initial()), self.initial())
        self.assertIsNotNone(controller.failure_reason)
        controller.reset(self.initial()); self.assertIsNone(controller.failure_reason)

    def test_lag_guard_preserves_last_safe_command(self):
        controller = self.controller(); controller.reset(self.initial())
        for _ in range(20):
            previous = controller.last.copy()
            try:
                controller.command([.03,0,50], self.initial())
            except ValueError as error:
                self.assertIn('tracking lag', str(error))
                np.testing.assert_array_equal(controller.last, previous)
                break
        else:
            self.fail('Expected the original 8-motor lead guard')

    def test_calibration_and_reset_prerequisites_reject(self):
        for changes in (dict(fixed_height_m=float('nan')), dict(tool_axis_table=[0,0,0]),
                        dict(xy_bounds=[[0,0],[-1,1]]), dict(table_to_base=np.zeros((4,4)))):
            with self.assertRaises(ValueError): self.controller(**changes)
        controller = self.controller(fixed_height_m=.2)
        with self.assertRaisesRegex(ValueError, 'Initial TCP'): controller.reset(self.initial())
        controller = self.controller(tool_axis_table=[0,1,0])
        with self.assertRaisesRegex(ValueError, 'Initial TCP'): controller.reset(self.initial())

    def test_calibrated_table_transform_and_axial_rotation(self):
        frame = np.eye(4); frame[:3,3] = [.01, -.02, .03]
        controller = self.controller(table_to_base=frame, fixed_height_m=.07)
        measured = self.initial(); measured[4] = 20
        controller.reset(measured)
        np.testing.assert_allclose(controller.pose(measured)[:3,3], [-.01,.02,.07])
        controller.command([-.01,.02,50], measured)  # Axial rotation is intentionally not locked.

    def test_wrapper_solves_once_and_holds_two_native_ticks(self):
        raw = self.env(); controller = self.controller(); wrapped = PlanarEEEnv(raw,controller)
        obs, _ = wrapped.reset(seed=1)
        self.assertEqual(len(obs.proprio), 6)
        with patch.object(controller, 'command', wraps=controller.command) as solve:
            for _ in range(4):
                result = wrapped.step([.01,0,50])
            self.assertEqual(solve.call_count, 2)
        np.testing.assert_array_equal(raw.sent[0], raw.sent[1])
        np.testing.assert_array_equal(raw.sent[2], raw.sent[3])
        self.assertEqual(result[-1]['command'], 'native preserved')
        self.assertEqual(result[-1]['ee_command']['requested_ee'], [.01,0,50])

    def test_wrapper_cannot_change_request_between_decisions(self):
        raw = self.env(); wrapped = PlanarEEEnv(raw,self.controller()); wrapped.reset()
        wrapped.step([0,0,50])
        with self.assertRaisesRegex(ValueError, 'decision boundaries'): wrapped.step([.001,0,50])
        self.assertEqual(len(raw.sent), 1)

    def test_terminal_is_not_stepped_and_reset_clears_state(self):
        raw = self.env(terminated=True); wrapped = PlanarEEEnv(raw,self.controller()); wrapped.reset()
        self.assertTrue(wrapped.step([0,0,50])[2])
        with self.assertRaises(RuntimeError): wrapped.step([0,0,50])
        wrapped.reset(); self.assertFalse(wrapped.ended)

    def test_wrapper_failure_is_explicit_and_native_hold_is_preserved(self):
        raw = self.env(); controller = self.controller(movable=False); wrapped = PlanarEEEnv(raw, controller)
        wrapped.reset(); result = wrapped.step([.02,0,50])
        self.assertTrue(result[-1]['ee_control_failed'])
        np.testing.assert_array_equal(raw.sent[0], self.initial())

    def test_factory_rejects_unknown_or_incomplete_interface_before_isaac(self):
        with patch('starter.envs.factory.LiftCubeEnvAdapter') as native:
            with self.assertRaisesRegex(ValueError, 'control_mode'): make_env({'env':{'control_mode':'course_ee'}})
            with self.assertRaisesRegex(ValueError, 'planar_ee_config'):
                make_env({'env':{'control_mode':'planar_ee_dev'}})
            native.assert_not_called()

    def test_factory_default_and_explicit_ee_modes_stay_separate(self):
        raw = self.env()
        with patch('starter.envs.factory.isaac_runtime_ready', return_value=True), \
             patch('starter.envs.factory.LiftCubeEnvAdapter', return_value=raw), \
             patch.object(PlanarEEController, 'from_file', return_value=self.controller()) as loader:
            self.assertIs(make_env({'env':{}}), raw)
            loader.assert_not_called()
            wrapped = make_env({'env':{'control_mode':'planar_ee_dev', 'planar_ee_config':'calibrated.json'}})
            self.assertIsInstance(wrapped, PlanarEEEnv)
            loader.assert_called_once_with('calibrated.json')

    def test_failed_reset_cannot_reuse_old_commands(self):
        controller = self.controller(); controller.reset(self.initial())
        bad = self.initial(); bad[0] = float('nan')
        with self.assertRaises(ValueError): controller.reset(bad)
        with self.assertRaises(RuntimeError): controller.command([0,0,50], self.initial())
        raw = self.env(); wrapped = PlanarEEEnv(raw, self.controller()); wrapped.reset()
        wrapped.controller.height = .2
        with self.assertRaises(ValueError): wrapped.reset()
        with self.assertRaises(RuntimeError): wrapped.step([0,0,50])

    def test_opt_in_compensation_keeps_reference_separate_and_is_bounded(self):
        controller = self.controller(tracking_compensation=True)
        measured = self.initial(); controller.reset(measured)
        previous = measured.copy()
        for _ in range(160):
            sent = controller.command([0,0,50], measured)
            self.assertLessEqual(np.max(np.abs(sent-previous)), 1.000001)
            self.assertLessEqual(np.max(np.abs(controller.bias)), 3.)
            np.testing.assert_allclose(controller.pose(controller.last)[:3,3], [0,0,.1], atol=1e-7)
            measured = sent.copy(); measured[0] -= 2.  # Synthetic static load, not physics.
            previous = sent
        self.assertLess(abs(measured[0]), 1e-4)
        self.assertGreater(abs(sent[0]), 1.)
        self.assertAlmostEqual(controller.telemetry['reference_motor'][0], 0., places=6)
        controller.reset(self.initial())
        np.testing.assert_array_equal(controller.bias, np.zeros(5))

    def test_compensation_failure_holds_actual_sent_not_nominal_reference(self):
        controller = self.controller(tracking_compensation=True); controller.reset(self.initial())
        measured = self.initial()
        for _ in range(8):
            controller.command([0,0,50], measured)
            measured = controller.sent.copy(); measured[0] -= 2.
        last_sent = controller.sent.copy()
        self.assertNotEqual(last_sent[0], controller.last[0])
        bad_feedback = self.initial(); bad_feedback[0] = 50.
        with self.assertRaisesRegex(ValueError, 'tracking lag'):
            controller.command([0,0,50], bad_feedback)
        np.testing.assert_array_equal(controller.command([0,0,50], bad_feedback), last_sent)


if __name__=='__main__':
    unittest.main()
