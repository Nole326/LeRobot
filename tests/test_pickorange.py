"""CPU contracts for the portable PickOrange native-assisted source port."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'course/pickorange'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy required')
class PickOrangeTests(unittest.TestCase):
    def test_protocol_and_check_have_no_simulator_dependency(self):
        from common import load_protocol
        cfg = load_protocol()
        self.assertEqual(cfg['scene_seeds'], list(range(17, 117)))
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'course/pickorange/evaluate.py'), '--check'],
                                capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['simulation_run'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            cfg['dynamic_gripper_effort'] = False
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'dynamic_gripper_effort'):
                load_protocol(path)

    def test_model_whitelist_no_clipping_and_shape_guards(self):
        from common import predict_groot, validate_request
        data = dict(joint_motor=np.zeros((1, 6)),
                    front=np.zeros((1, 480, 640, 3), np.uint8),
                    wrist=np.ones((1, 480, 640, 3), np.uint8))
        class Policy:
            def get_action(self, payload):
                self.payload = payload
                return {'action.single_arm': np.full((16, 5), 101.), 'action.gripper': np.zeros((16, 1))}
        policy = Policy()
        actions = predict_groot(policy, data, 'task')
        self.assertEqual(actions[0, 0], 101)  # Explicitly native, not silently clamped.
        self.assertEqual(set(policy.payload), {'video.front', 'video.wrist', 'state.single_arm', 'state.gripper', 'annotation.human.task_description'})
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            validate_request(dict(data, objects=np.zeros((3, 3))))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            validate_request(dict(data, wrist=data['front'].astype(float)))

    def test_atomic_ipc_and_bounded_wait(self):
        from common import save_arrays, wait_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'request.npz'
            save_arrays(path, q=np.arange(6))
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['q'], np.arange(6))
            with self.assertRaises(FileExistsError):
                save_arrays(path, q=np.arange(6))
            with self.assertRaises(TimeoutError):
                wait_file(Path(tmp) / 'absent', 0)

    def test_pinned_backends_checked_before_optional_imports(self):
        from common import verify_backend
        import service_n16
        import service_pi05
        import service_smolvla
        with patch('common.subprocess.check_output', side_effect=['expected\n', '']):
            verify_backend(Path('backend'), 'expected')
        for outputs in (['wrong\n', ''], ['expected\n', ' M tracked.py\n']):
            with patch('common.subprocess.check_output', side_effect=outputs):
                with self.assertRaisesRegex(ValueError, 'pinned model backend'):
                    verify_backend(Path('backend'), 'expected')
        for module in (service_n16, service_pi05, service_smolvla):
            with patch.object(module, 'verify_backend', side_effect=ValueError('rejected')) as check:
                with self.assertRaisesRegex(ValueError, 'rejected'):
                    module.load_policy(Path('model'), Path('backend'), 'cpu')
                check.assert_called_once()

    def test_smol_export_compatibility_is_narrow(self):
        from service_smolvla import compatible_config
        raw = dict(type='smolvla', chunk_size=50, use_peft=False,
                   compile_model=False, compile_mode='max-autotune')
        self.assertEqual(compatible_config(raw, {'chunk_size'}), {'chunk_size': 50})
        self.assertIn('type', raw)
        for change in ({'use_peft': True}, {'unexpected': 1}, {'compile_model': 0}):
            with self.assertRaisesRegex(ValueError, 'Unsupported export'):
                compatible_config(dict(raw, **change), {'chunk_size'})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch required for SmolVLA fixture')
    def test_smol_dual_views_processors_reset_and_clock(self):
        import torch
        from common import load_protocol, timing
        from service_smolvla import predict
        cfg = load_protocol(ROOT / 'course/pickorange/protocols/smolvla-native-assisted-30hz.json')
        self.assertEqual(timing(cfg), (2, 32, 3600))
        data = dict(joint_motor=np.arange(6)[None], front=np.zeros((1,480,640,3), np.uint8),
                    wrist=np.full((1,480,640,3), 255, np.uint8))
        calls = []
        def pre(batch):
            calls.append('pre')
            self.assertEqual(batch['task'], 'test')
            self.assertEqual(tuple(batch['observation.images.front'].shape), (3,480,640))
            self.assertTrue(torch.all(batch['observation.images.front'] == 0))
            self.assertTrue(torch.all(batch['observation.images.wrist'] == 1))
            np.testing.assert_array_equal(batch['observation.state'].numpy(), np.arange(6))
            return batch
        class Policy:
            def reset(self):
                calls.append('reset')
            def predict_action_chunk(self, batch):
                calls.append('predict')
                return torch.full((1,50,6), 101., dtype=torch.bfloat16)
        def post(actions):
            calls.append('post')
            self.assertEqual(actions.dtype, torch.float32)
            return actions * 2
        for _ in range(2):
            np.testing.assert_array_equal(predict((Policy(), pre, post), data, 'test'), np.full((50,6), 202.))
        self.assertEqual(calls, ['reset', 'pre', 'predict', 'post'] * 2)
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            predict((Policy(), pre, post), dict(data, object_positions=np.zeros(3)), 'test')

    def test_tracking_closes_before_application_even_on_errors(self):
        from common import close_resources
        for failing in (None, 'writer', 'env'):
            calls = []

            class Resource:
                def __init__(self, name):
                    self.name = name

                def close(self):
                    calls.append(self.name)
                    if self.name == failing:
                        raise RuntimeError(self.name)

            resources = [Resource(name) for name in ('writer', 'env', 'app')]
            if failing is None:
                close_resources(*resources)
            else:
                with self.assertRaisesRegex(RuntimeError, failing):
                    close_resources(*resources)
            self.assertEqual(calls, ['writer', 'env', 'app'])
        close_resources(None, None, None)

    def test_candidate_protocol_clocks_and_n16_absolute_output(self):
        from common import load_protocol, timing
        from service_n16 import predict
        protocols = ROOT / 'course/pickorange/protocols'
        n15 = load_protocol()
        n16 = load_protocol(protocols / 'groot-n16-native-assisted.json')
        pi05 = load_protocol(protocols / 'pi05-native-assisted-30hz.json')
        self.assertEqual({k for k in n15 if n15[k] != n16[k]},
                         {'name', 'model_repo', 'model_revision', 'backend_revision'})
        self.assertEqual(timing(n16), (1, 16, 3600))
        self.assertEqual(timing(pi05), (2, 32, 3600))
        q = np.array([[20., -40., 60., -80., 0., 10.]])
        data = dict(joint_motor=q, front=np.zeros((1, 480, 640, 3), np.uint8),
                    wrist=np.ones((1, 480, 640, 3), np.uint8))

        class DecodedPolicy:
            def get_action(self, observation):
                self.observation = observation
                return dict(single_arm=np.full((1, 16, 5), 101.),
                            gripper=np.full((1, 16, 1), 25.)), {}

        policy = DecodedPolicy()
        actions = predict(policy, data, 'task')
        np.testing.assert_array_equal(actions[0], [101., 101., 101., 101., 101., 25.])
        self.assertEqual(policy.observation['video']['front'].shape, (1, 1, 480, 640, 3))
        self.assertTrue((policy.observation['video']['wrist'] == 1).all())
        np.testing.assert_array_equal(policy.observation['state']['single_arm'], q[:, None, :5])

    def test_native_success_strict_ranges_and_rest(self):
        from audit import native_success
        joints = np.deg2rad([0, -100, 90, 50, 0, -10])
        objects = np.zeros((4, 3))
        self.assertTrue(native_success(joints, objects))
        objects[0, 0] = .10
        self.assertFalse(native_success(joints, objects))
        objects[:] = 0
        self.assertFalse(native_success(np.zeros(6), objects))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch required for PI05 fixture')
    def test_pi05_saved_processor_boundary_and_view_order(self):
        import torch
        from service_pi05 import predict, VIEWS
        q = np.array([[10., 20., 30., 40., 50., 60.]])
        data = dict(joint_motor=q, front=np.zeros((1, 480, 640, 3), np.uint8),
                    wrist=np.full((1, 480, 640, 3), 255, np.uint8))
        calls = []

        def preprocess(batch):
            calls.append('pre')
            self.assertEqual(batch['task'], 'task')
            np.testing.assert_array_equal(batch['observation.state'].numpy(), q[0])
            self.assertEqual(VIEWS, ('wrist', 'front'))
            for view, value in (('wrist', 1.), ('front', 0.)):
                image = batch['observation.images.'+view]
                self.assertEqual(tuple(image.shape), (3, 480, 640))
                self.assertTrue(torch.all(image == value))
            return 'processed'

        class Policy:
            def predict_action_chunk(self, batch):
                calls.append(batch)
                return torch.full((1, 50, 6), 101., dtype=torch.bfloat16)

        def postprocess(actions):
            calls.append('post')
            self.assertEqual(actions.dtype, torch.float32)
            return actions * 2 + 5

        actions = predict((Policy(), preprocess, postprocess), data, 'task')
        np.testing.assert_array_equal(actions, np.full((50, 6), 207.))
        self.assertEqual(calls, ['pre', 'processed', 'post'])

    def test_oft_dual_view_and_fail_closed(self):
        from openvla_oft import PickOrangeOFT, read_package
        views = dict(front=np.zeros((1, 480, 640, 3), np.uint8),
                     wrist=np.ones((1, 480, 640, 3), np.uint8))
        q = np.array([[10., 20., 30., 40., 50., 60.]])

        def native(cfg, model, processor, payload, task, **kwargs):
            self.assertEqual(cfg.num_images_in_input, 2)
            self.assertEqual(cfg.unnorm_key, 'test_key')
            self.assertEqual(task, 'task')
            np.testing.assert_array_equal(payload['state'], q[0])
            np.testing.assert_array_equal(payload['full_image'], views['front'][0])
            np.testing.assert_array_equal(payload['additional_images'][0], views['wrist'][0])
            self.assertIsNotNone(kwargs['action_head'])
            self.assertIsNotNone(kwargs['proprio_projector'])
            return np.full((30, 6), 101.)

        p = PickOrangeOFT(None, None, object(), object(), native, unnorm_key='test_key', center_crop=False)
        np.testing.assert_array_equal(p.predict(dict(views, joint_motor=q), 'task'), np.full((30, 6), 101.))
        with self.assertRaisesRegex(ValueError, 'requires trained'):
            PickOrangeOFT(None, None, None, object(), native, unnorm_key='test_key', center_crop=False)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps(dict(trained_camera_keys=['front'])))
            with self.assertRaisesRegex(ValueError, 'dual-view'):
                read_package(manifest)
            manifest.write_text(json.dumps(dict(task_id='LeIsaac-SO101-PickOrange-v0',
                trained_camera_keys=['front', 'wrist'], state_units='so101_motor', action_units='so101_motor',
                action_representation='absolute_joint_targets', normalization='bounds', chunk_size=30,
                center_crop=False, model_path='.', source='test', revision='test', unnorm_key='test_key',
                action_head=dict(path='missing.safetensors', sha256='invalid'))))
            with self.assertRaisesRegex(ValueError, 'action_head'):
                read_package(manifest)

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py required for trajectory audit fixture')
    def test_closed_trajectory_audit_and_corruption(self):
        import h5py
        from audit import audit_episode
        from common import publish, save_arrays, sha256
        sys.path.insert(0, str(ROOT / 'course/scripts'))
        from joint_contract import motor_to_rad, rad_to_motor
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            publish(out / 'manifest.json', dict(action_offsets_rad=[0]*6, action_scales=[1]*6))
            path = out / 'trajectory.hdf5'
            joints = np.tile(np.deg2rad([0, -100, 90, 50, 0, -10]), (2, 1)).astype(np.float32)
            motor = rad_to_motor(joints)
            targets = motor_to_rad(motor)
            (out / 'ipc').mkdir()
            (out / 'model-rng-after-episode.pt').write_bytes(b'synthetic RNG artifact, not a model checkpoint')
            rng_digest = sha256(out / 'model-rng-after-episode.pt')
            save_arrays(out / 'ipc/request-000000.npz', joint_motor=motor[:1],
                        front=np.zeros((1, 480, 640, 3), np.uint8), wrist=np.zeros((1, 480, 640, 3), np.uint8))
            save_arrays(out / 'ipc/response-000000.npz', actions_motor=np.tile(motor[:1], (16, 1)),
                        inference_sec=np.asarray(.1), query_id=np.asarray(0), model_rng_sha256=np.asarray(rng_digest))
            with h5py.File(path, 'x') as f:
                f.attrs.update(complete=True, physics_steps=2, env_seed=17, success=True, terminated=True, truncated=False)
                for key, values in dict(raw_model_motor=motor, target_rad=targets, action_manager_raw=targets,
                                        pre_joint_rad=joints, post_joint_rad=joints).items():
                    f.create_dataset(key, data=values)
                f.create_dataset('diagnostics/object_pos_w', data=np.zeros((2, 4, 3)))
                f.create_dataset('diagnostics/gripper_effort_limit', data=np.ones(2))
                f.create_dataset('sample_steps', data=[0])
                f.create_dataset('query_ids', data=[0])
                f.create_dataset('inference_seconds', data=[.1])
                for cam in ('front', 'wrist'):
                    f.create_dataset('policy_rgb/'+cam, data=np.zeros((1, 480, 640, 3), np.uint8))
                    f.create_dataset('terminal_rgb/'+cam, data=np.zeros((480, 640, 3), np.uint8))
            row = dict(episode=0, env_seed=17, physics_steps=2, success=True, model_queries=1,
                       terminated=True, truncated=False, trajectory=path.name, trajectory_sha256=sha256(path))
            row['model_rng_sha256'] = rng_digest
            result = audit_episode(out, row)
            self.assertTrue(result['trajectory_audit_passed'])
            self.assertFalse(result['automatic_bc_admission'])
            row['success'] = False
            with self.assertRaisesRegex(ValueError, 'success'):
                audit_episode(out, row)


if __name__ == '__main__':
    unittest.main()
