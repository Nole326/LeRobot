"""CPU tests for calibrated RGB rules; synthetic evidence is not task success."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT/'course/pickorange/visual'
sys.path.insert(0, str(VISUAL))
try:
    import numpy as np
except ImportError:
    np = None


class EntrypointTests(unittest.TestCase):
    def test_check_does_not_import_simulator_or_sdk(self):
        program = """
import runpy,sys
sys.argv=[sys.argv[1],'--check']
runpy.run_path(sys.argv[0],run_name='__main__')
assert not any(k.startswith(('isaaclab','omni','torch','swanlab')) for k in sys.modules)
"""
        result = subprocess.run([sys.executable, '-S', '-c', program, str(VISUAL/'evaluate.py')],
                                capture_output=True, text=True, check=True)
        protocol = json.loads(result.stdout)
        self.assertFalse(protocol['dynamic_gripper_truth_helper'])
        self.assertEqual(protocol['horizon_seconds'], 55)
        self.assertEqual(protocol['native_horizon_seconds'], 25)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
class ResetCaptureTests(unittest.TestCase):
    def load_helper(self, math=None):
        spec = importlib.util.spec_from_file_location('visual_reset_capture', VISUAL/'reset_capture.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': importlib.import_module('torch'), 'isaaclab.utils.math': math or Mock()}):
            spec.loader.exec_module(module)
        return module

    def test_repeated_front_resets_use_nominal_mount(self):
        import torch

        quaternion = torch.tensor([[1., 0., 0., 0.]])
        delta = torch.tensor([[.02, 0., -.01, 0., 0., 0.]])
        math = SimpleNamespace(combine_frame_transforms=lambda p, q, offset, rotation: (p + offset, q),
            sample_uniform=Mock(return_value=delta), quat_from_euler_xyz=Mock(return_value=quaternion),
            quat_mul=lambda q, update: q)
        helper = self.load_helper(math)
        front = SimpleNamespace(cfg=SimpleNamespace(offset=SimpleNamespace(pos=(.1, .2, .3), rot=(1., 0., 0., 0.),
            convention='ros')), set_world_poses=Mock())
        root = torch.tensor([[1., 2., 3., 1., 0., 0., 0.]])
        robot = SimpleNamespace(data=SimpleNamespace(default_root_state=root))

        class Scene(dict):
            env_origins = torch.tensor([[10., 0., 0.]])

        env = SimpleNamespace(scene=Scene(front=front, robot=robot), device='cpu')
        event = SimpleNamespace(params={'asset_cfg': SimpleNamespace(name='front'), 'pose_range': {'x': (-.025, .025)}})
        cfg = SimpleNamespace(events=SimpleNamespace(front=event), scene=SimpleNamespace(front=front.cfg, wrist=Mock()))
        helper.configure_front_reset(cfg)
        for _ in range(2):
            event.func(env, [0], **event.params)
            torch.testing.assert_close(front.set_world_poses.call_args.args[0], torch.tensor([[11.12, 2.2, 3.29]]))
        self.assertEqual(math.sample_uniform.call_count, 2)
        self.assertTrue(cfg.scene.front.update_latest_camera_pose)
        self.assertTrue(cfg.scene.wrist.update_latest_camera_pose)

    def test_refresh_returns_new_observations_without_physics(self):
        helper = self.load_helper()
        calls = []

        class Camera:
            def reset(self, indices):
                calls.append(('reset', indices.tolist()))

            @property
            def data(self):
                calls.append('camera-data')

        env = SimpleNamespace(device='cpu', _sim_step_counter=91, scene={'front': Camera(), 'wrist': Camera()},
            sim=SimpleNamespace(render=lambda: calls.append('render')),
            observation_manager=SimpleNamespace(compute=Mock(return_value='fresh-observations')))
        self.assertEqual(helper.refresh_reset_observations(env, [0]), 'fresh-observations')
        self.assertEqual(calls, ['render']*4 + [('reset', [0]), 'camera-data']*2)
        env.observation_manager.compute.assert_called_once_with(update_history=False)
        self.assertEqual(env._sim_step_counter, 91)


@unittest.skipIf(np is None, 'NumPy required')
class GeometryTests(unittest.TestCase):
    def calibration(self):
        from perception import Calibration
        k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
        t = np.diag([1., -1, -1, 1])
        t[:3, 3] = [0, -.25, .6]
        return Calibration(k, t)

    def test_projection_round_trip_and_failure_cases(self):
        from perception import Calibration
        c = self.calibration()
        point = np.array([.10, -.18, .056])
        optical = c.camera_to_base[:3, :3].T@(point-c.camera_to_base[:3, 3])
        projected = c.intrinsic@optical
        np.testing.assert_allclose(c.intersect(projected[:2]/projected[2], point[2]), point, atol=1e-10)
        with self.assertRaisesRegex(ValueError, 'behind'):
            c.intersect([320, 240], .8)
        bad = c.camera_to_base.copy(); bad[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'rotation'):
            Calibration(c.intrinsic, bad)
        with self.assertRaisesRegex(ValueError, 'finite'):
            Calibration(c.intrinsic, np.full((4, 4), np.nan))

    def test_controller_input_boundary_and_bounded_joint_requests(self):
        import policy

        class Kinematics:
            limits = np.deg2rad([[-110, 110], [-100, 100], [-100, 90], [-95, 95], [-160, 160]])
            def set_opening(self, opening):
                self.opening = opening
            def fk(self, q):
                return np.eye(4), np.array([.1, -.25, .056]), None
            def solve(self, target, q, local=False):
                return dict(q=np.asarray(q), position_error_m=0)

        class Geometry:
            def pad_pair(self, depth):
                return np.zeros(3), np.ones(3)*.01

        with patch.object(policy, 'DiameterKinematics', return_value=Kinematics()), \
             patch.object(policy, 'FingerGeometry', return_value=Geometry()):
            actor = policy.VisualPolicy('unused-robot-asset', self.calibration())
        self.assertFalse(hasattr(actor, 'env'))
        self.assertFalse(hasattr(actor, 'robot'))
        q = np.array([0., 0, 0, 0, 0, .5])
        actor.phase = 'close'
        actor.target = np.array([.1, -.25, .056])
        requested = actor.update(None, q, np.zeros(5))
        self.assertGreaterEqual(requested[5], q[5]-.020)
        self.assertLessEqual(np.max(np.abs(requested[:5]-q[:5])), .010)
        # A jaw pushed open can leave the previous sent command below the new bound.
        for phase in ('close', 'lift', 'carry', 'lower'):
            with self.subTest(phase=phase):
                actor.phase, actor.age, actor.stable = phase, 0, 0
                actor.previous = np.r_[q[:5], .43]
                self.assertGreaterEqual(actor.update(None, q, np.zeros(5))[5], q[5]-.020)
        np.testing.assert_array_equal(q, [0, 0, 0, 0, 0, .5])
        with self.assertRaisesRegex(ValueError, 'finite'):
            actor.update(None, np.full(6, np.nan), np.zeros(5))
        with self.assertRaisesRegex(ValueError, 'q6'):
            actor.update(None, np.zeros(5), np.zeros(5))

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required')
    def test_oranges_plate_yellow_robot_and_background(self):
        import cv2
        from perception import Detector
        rgb = np.full((480, 640, 3), [120, 85, 55], dtype=np.uint8)
        cv2.circle(rgb, (120, 250), 75, (210, 215, 215), -1)
        for center in [(300, 170), (370, 240), (450, 310)]:
            cv2.ellipse(rgb, center, (21, 19), 12, 0, 360, (230, 100, 8), -1)
        cv2.rectangle(rgb, (300, 5), (355, 90), (235, 225, 45), -1)
        detector = Detector(self.calibration())
        found = detector.detect(rgb)
        self.assertEqual(len(found['oranges']), 3)
        self.assertIsNotNone(found['plate'])
        np.testing.assert_allclose(found['plate']['pixel'], [120, 250], atol=1)
        self.assertEqual(detector.annotate(rgb, found).shape, rgb.shape)
        with self.assertRaisesRegex(ValueError, 'uint8'):
            detector.detect(rgb.astype(float))


@unittest.skipIf(np is None, 'NumPy required')
class StrictGeometryTests(unittest.TestCase):
    def test_runner_strict_dispatch_cannot_query_robot_force(self):
        import ast
        from types import SimpleNamespace
        source = ast.parse((VISUAL/'evaluate.py').read_text())
        branches = [node for node in ast.walk(source) if isinstance(node, ast.If)
            and any(isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr == 'update'
                for statement in node.body)]
        self.assertEqual(len(branches), 1)

        class ForbiddenRobot:
            def __getattr__(self, name):
                raise AssertionError(f'Strict actor queried forbidden robot interface: {name}')

        class RGB:
            def __getitem__(self, item):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return np.zeros((8, 8, 3), np.uint8)

        received = []
        actor = SimpleNamespace(update=lambda *args: received.append(args) or np.zeros(6))
        scope = dict(args=SimpleNamespace(input_contract='rgb-q-calibration'), actor=actor,
            obs=dict(policy=dict(wrist=RGB())), rgb=np.zeros((8, 8, 3), np.uint8),
            q=np.zeros(6), robot=ForbiddenRobot(), ids=list(range(6)))
        exec(compile(ast.Module(body=branches, type_ignores=[]), 'strict_dispatch', 'exec'), scope)
        self.assertEqual(len(received), 1)
        self.assertEqual(len(received[0]), 3)  # Front, wrist, q; no hidden force argument.
        np.testing.assert_array_equal(scope['wanted'], np.zeros(6))

    def test_strict_protocol_excludes_force_and_scene_priors(self):
        result = subprocess.run([sys.executable, '-S', str(VISUAL/'evaluate.py'), '--check',
            '--input-contract', 'rgb-q-calibration'], capture_output=True, text=True, check=True)
        cfg = json.loads(result.stdout)
        self.assertEqual(set(cfg['actor_inputs']), {'front_rgb', 'wrist_rgb', 'measured_q6_rad',
            'known_camera_intrinsics_and_mount_extrinsics'})
        for excluded in ('gravity_compensation_force', 'table_height_prior', 'object_size_prior', 'depth'):
            self.assertIn(excluded, cfg['excluded_inputs'])
        self.assertNotIn('support_plane_z_base_m', cfg)
        self.assertNotIn('orange_center_height_prior_m', cfg)

    def test_plane_estimation_recovers_unknown_tilt_and_height_with_outliers(self):
        from stereo import fit_plane, CameraModel
        rng = np.random.default_rng(37)
        xy = rng.uniform(-.25, .25, (120, 2))
        # Arbitrary height and tilt differ from the task; they are not inputs
        # to the estimator. Independent synthetic 3-D measurements provide it.
        z = .143+.11*xy[:, 0]-.07*xy[:, 1]+rng.normal(0, .00015, len(xy))
        points = np.r_[np.c_[xy, z], rng.uniform(-.3, .3, (30, 3))]
        plane, report = fit_plane(points)
        self.assertGreaterEqual(report['inliers'], 118)
        self.assertLess(report['residual_rms_m'], .0004)
        expected = np.array([-.11, .07, 1.]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(plane[:3], expected, atol=.002)
        self.assertAlmostEqual(-plane[3]/plane[2], .143, places=3)
        c = CameraModel(np.eye(3), np.eye(4))
        with self.assertRaisesRegex(ValueError, 'parallel'):
            c.intersect([0, 0], [1, 0, 0, -1])
        with self.assertRaisesRegex(ValueError, '18'):
            fit_plane(points[:8])

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required')
    def test_metric_triangulation_rejects_inconsistent_correspondences(self):
        from stereo import CameraModel, triangulate
        k = np.array([[450., 0, 320], [0, 450, 240], [0, 0, 1.]])
        t1, t2 = np.eye(4), np.eye(4)
        t2[0, 3] = .17
        a, b = CameraModel(k, t1), CameraModel(k, t2)
        points = np.array([[.10, -.05, .45], [.08, .03, .65], [-.1, .1, .50]])
        pixels = []
        for camera in (a, b):
            projected = np.c_[points, np.ones(len(points))]@camera.projection.T
            pixels.append(projected[:, :2]/projected[:, 2:])
        pixels[1][-1, 1] += 35  # Off-epipolar match must not create a trusted point.
        estimated, valid = triangulate(a, b, *pixels)
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(estimated[:2], points[:2], atol=1e-7)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required')
    def test_detection_has_no_metric_size_or_table_height_default(self):
        from stereo import CameraModel, VisualGeometryDetector
        import cv2
        k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1.]])
        t = np.diag([1., -1, -1, 1]); t[2, 3] = .8
        detector = VisualGeometryDetector(CameraModel(k, t))
        image = np.full((480, 640, 3), [110, 85, 65], dtype=np.uint8)
        cv2.circle(image, (320, 240), 22, (230, 100, 8), -1)
        self.assertEqual(detector.detect(image), dict(oranges=[], plate=None))
        detector.plane = np.array([0., 0, 1, -.20])
        first = detector.detect(image)['oranges'][0]
        detector.plane = np.array([0., 0, 1, -.40])
        second = detector.detect(image)['oranges'][0]
        # Metric radius responds to visually measured plane depth rather than
        # a frozen fruit diameter. The same pixels at nearer depth mean smaller fruit.
        self.assertAlmostEqual(first['radius_m']/second['radius_m'], 1.5, places=5)
        self.assertAlmostEqual(first['point'][2], .20+first['radius_m'])


if __name__ == '__main__':
    unittest.main()
