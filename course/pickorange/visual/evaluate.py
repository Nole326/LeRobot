"""Evaluate calibrated RGB rules using native joint servos and native scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback


def publish(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check protocol without importing simulator')
    parser.add_argument('--run-simulation', action='store_true')
    parser.add_argument('--sim-camera-calibration', action='store_true', help='Explicit actual camera/base transform auxiliary')
    parser.add_argument('--save-debug-images', action='store_true', help='Save periodic RGB and detection images')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--horizon', type=int, choices=(25, 55), default=55)
    parser.add_argument('--table-z', type=float, default=.030, help='Calibrated support-plane Z in robot base, metres')
    parser.add_argument('--input-contract', choices=('model-assisted', 'rgb-q-calibration'), default='model-assisted')
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        print('Simulation options: --headless --enable_cameras --device cuda:0 --kit_args=--allow-root')
        return None, None
    if '--check' in sys.argv:
        args = parser.parse_args()
        print(json.dumps(protocol(args), indent=2))
        return None, None
    if '--run-simulation' not in sys.argv:
        parser.error('Use --check or explicitly --run-simulation')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.sim_camera_calibration:
        parser.error('This runner requires explicit --sim-camera-calibration')
    if args.out is None or args.out.exists():
        parser.error('--out must be a fresh directory')
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error('Seeds must be nonempty and unique')
    args.out.mkdir(parents=True)
    return args, AppLauncher


def protocol(args):
    if args.input_contract == 'rgb-q-calibration':
        return dict(name=f'visual-rgb-q-calibration-fixed-{args.horizon}s-v2',
            actor_inputs=['front_rgb', 'wrist_rgb', 'measured_q6_rad', 'known_camera_intrinsics_and_mount_extrinsics'],
            excluded_inputs=['object_pose', 'object_mass', 'segmentation', 'depth', 'success', 'raycast',
                'gravity_compensation_force', 'table_height_prior', 'object_size_prior'],
            camera_calibration='reset_front_to_base_and_wrist_to_tool_opt_in_then_FK_from_measured_q',
            robot_geometry='fixed_kinematic_chain_and_finger_mesh_only_no_dynamics_model',
            action_hz=30, physics_hz=60, horizon_seconds=args.horizon, native_horizon_seconds=25,
            native_success='all_three_in_native_XYZ_box_and_robot_home',
            dynamic_gripper_truth_helper=False, native_actuator_gains_gravity_collision=True,
            geometry='RGB_feature_triangulation_support_plane_and_silhouette_size_estimates',
            compression_command_bound_rad=.020, seeds=args.seeds, logging='shared_SwanLab_offline_scalars')
    return dict(name=f'visual-calibrated-native-fixed-{args.horizon}s-v2',
        actor_inputs=['front_rgb', 'measured_q6_rad', 'explicit_camera_intrinsics_extrinsics',
                      'calibrated_support_plane', 'robot_usd_geometry', 'robot_model_gravity_bias'],
        excluded_inputs=['object_pose', 'object_mass', 'segmentation', 'success', 'raycast'],
        camera_calibration='actual_sim_camera_and_robot_base_only_opt_in',
        action_hz=30, physics_hz=60, horizon_seconds=args.horizon,
        native_horizon_seconds=25, native_success='all_three_in_native_XYZ_box_and_robot_home',
        dynamic_gripper_truth_helper=False, native_actuator_gains_gravity_collision=True,
        compression_command_bound_rad=.020, support_plane_z_base_m=args.table_z,
        orange_center_height_prior_m=.026, plate_center_height_prior_m=.012,
        seeds=args.seeds, logging='shared_SwanLab_offline_scalars')


def main():
    args, launcher = arguments()
    if args is None:
        return
    out = args.out
    app = env = writer = None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        app = launcher(args).app
        import gymnasium as gym
        import leisaac.tasks
        if args.save_debug_images:
            from PIL import Image
        from scipy.spatial.transform import Rotation
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm
        from isaaclab.utils import configclass
        from isaaclab_tasks.utils import parse_env_cfg
        from perception import Calibration
        from reset_capture import configure_front_reset, refresh_reset_observations
        from policy import VisualPolicy
        from strict_policy import StrictVisualPolicy
        from stereo import CameraModel
        from joint_contract import JOINTS
        from leisaac.utils.robot_utils import is_so101_at_rest_pose
        starter = Path(__file__).resolve().parents[2] / 'liftcube' / 'src'
        sys.path.insert(0, str(starter))
        from starter.logging.swanlab import ScalarWriter

        def state(e):
            # Scoring-only boundary. This dictionary is never passed to actor.
            term = e.cfg.terminations.success
            robot = e.scene['robot']
            plate = e.scene[term.params['plate_cfg'].name].data.root_pos_w[0]
            oranges = torch.stack([e.scene[o.name].data.root_pos_w[0] for o in term.params['oranges_cfg']])
            relative = oranges-plate
            bounds = relative.new_tensor([term.params.get('x_range', (-.1, .1)),
                term.params.get('y_range', (-.1, .1)), term.params.get('height_range', (-.07, .07))])
            inside = ((relative > bounds[:, 0]) & (relative < bounds[:, 1])).all(-1)
            rest = bool(is_so101_at_rest_pose(robot.data.joint_pos, robot.data.joint_names)[0])
            success = bool(term.func(e, **term.params)[0])
            assert success == (bool(inside.all()) and rest)
            return dict(inside=inside.tolist(), plate_count=int(inside.sum()), at_rest=rest,
                success=success, oranges_xyz=oranges.tolist(), plate_xyz=plate.tolist(),
                joint_rad=robot.data.joint_pos[0, ids].tolist())

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if getattr(self._env, 'visual_capture_enabled', False) and bool(self._env.reset_buf.any()):
                    self._env.visual_final = state(self._env)
                return None, None

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device=args.device, num_envs=1)
        cfg.use_teleop_device('so101leader')
        cfg.seed = args.seeds[0]
        cfg.episode_length_s = args.horizon
        cfg.dynamic_reset_gripper_effort_limit = False
        cfg.recorders = Recorder()
        configure_front_reset(cfg)
        env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
        robot = env.scene['robot']
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt-1/60) < 1e-7
        terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
        assert sum([list(t._joint_names) for t in terms], []) == JOINTS
        scales = torch.cat([torch.as_tensor(t._scale, device=env.device).expand(1, t.action_dim) for t in terms], 1)
        offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1, t.action_dim) for t in terms], 1)
        manifest = protocol(args)
        manifest['debug_images'] = args.save_debug_images
        manifest.update(robot_usd_sha256=hashlib.sha256(Path(cfg.scene.robot.spawn.usd_path).read_bytes()).hexdigest(),
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            robot_geometry_source_sha256={name:hashlib.sha256((Path(__file__).parents[1]/'expert'/name).read_bytes()).hexdigest()
                for name in ('geometric_teacher.py','diameter_teacher.py','finger_geometry.py','diameter_geometry.json','joint_contract.py')},
            python=sys.version, torch=torch.__version__, leisaac=leisaac.tasks.__file__,
            actuator_configuration=str(cfg.scene.robot.actuators))
        publish(out/'manifest.json', manifest)
        writer = ScalarWriter(dict(logging=dict(dir=str(out), swanlab=dict(mode='offline', name='pickorange-visual')),
            env=dict(id='LeIsaac-SO101-PickOrange-v0', control_hz=30), policy=dict(type='visual_rules')), 'eval')
        results = []
        for seed in args.seeds:
            run = out/f'seed-{seed}'
            run.mkdir()
            env.visual_capture_enabled = False
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.reset(seed=seed)
            obs = refresh_reset_observations(env, [0])
            env.visual_capture_enabled, env.visual_final = True, None
            camera = env.scene['front']
            positions, orientations = camera.data.pos_w, camera.data.quat_w_ros
            base_id = list(robot.body_names).index('base')
            base = robot.data.body_pos_w[0, base_id].cpu().numpy()
            quat = robot.data.body_quat_w[0, base_id].cpu().numpy()
            base_r = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            camera_q = orientations[0].cpu().numpy()
            camera_r = Rotation.from_quat(camera_q[[1, 2, 3, 0]]).as_matrix()
            transform = np.eye(4)
            transform[:3, :3] = base_r.T@camera_r
            transform[:3, 3] = base_r.T@(positions[0].cpu().numpy()-base)
            intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()
            if args.input_contract == 'rgb-q-calibration':
                # One-time hand-eye calibration only; no dynamic wrist-pose query.
                wrist = env.scene['wrist']
                wrist_p, wrist_q = wrist.data.pos_w, wrist.data.quat_w_ros
                wq = wrist_q[0].cpu().numpy()
                wt = np.eye(4)
                wt[:3, :3] = base_r.T@(Rotation.from_quat(wq[[1, 2, 3, 0]]).as_matrix())
                wt[:3, 3] = base_r.T@(wrist_p[0].cpu().numpy()-base)
                wk = wrist.data.intrinsic_matrices[0].cpu().numpy()
                # Robot geometry is known calibration, not object/environment truth.
                from diameter_teacher import DiameterKinematics
                kin = DiameterKinematics(cfg.scene.robot.spawn.usd_path)
                tool, _, _ = kin.fk(robot.data.joint_pos[0, ids[:5]].cpu().numpy())
                mount = np.linalg.inv(tool)@wt
                calibration = CameraModel(intrinsic, transform)
                publish(run/'calibration.json', dict(intrinsic=intrinsic.tolist(), camera_to_base=transform.tolist(),
                    wrist_intrinsic=wk.tolist(), wrist_to_tool=mount.tolist(), source='known_camera_calibration_only'))
                actor = StrictVisualPolicy(cfg.scene.robot.spawn.usd_path, calibration, wk, mount)
            else:
                calibration = Calibration(intrinsic, transform, args.table_z)
                publish(run/'calibration.json', dict(intrinsic=intrinsic.tolist(), camera_to_base=transform.tolist(),
                    table_z=args.table_z, source='explicit_actual_camera_base_auxiliary'))
                actor = VisualPolicy(cfg.scene.robot.spawn.usd_path, calibration)
            action_history, measured_history, phases = [], [], []
            initial = state(env)
            publish(run/'initial-scoring-only.json', initial)
            trace = (run/'trace.jsonl').open('w')
            try:
                for step in range(args.horizon*60):
                    if step % 2 == 0:
                        q = robot.data.joint_pos[0, ids].cpu().numpy()
                        rgb = obs['policy']['front'][0].cpu().numpy()
                        if args.input_contract == 'rgb-q-calibration':
                            wanted = actor.update(rgb, obs['policy']['wrist'][0].cpu().numpy(), q)
                        else:
                            # Legacy robot-model auxiliary is excluded by strict branch.
                            bias = robot.root_physx_view.get_gravity_compensation_forces()[0, ids[:5]].cpu().numpy()/17.8
                            wanted = actor.update(rgb, q, bias)
                        command = (torch.as_tensor(wanted, device=env.device, dtype=torch.float32).unsqueeze(0)-offsets)/scales
                        action_history.append(wanted.copy()); measured_history.append(q.copy()); phases.append(actor.phase)
                    obs, _, term, trunc, _ = env.step(command)
                    done = bool(term[0] | trunc[0])
                    if done and env.visual_final is None:
                        raise RuntimeError('Missing pre-reset terminal capture')
                    current = env.visual_final if done else state(env)
                    if step % 30 == 29 or done:
                        trace.write(json.dumps(dict(step=step+1, phase=actor.phase, age=actor.age,
                            tcp_base_m=None if done else actor.tcp(robot.data.joint_pos[0, ids].cpu().numpy()).tolist(),
                            target_base_m=None if actor.target is None else actor.target.tolist(), scoring_only=current))+'\n')
                        trace.flush()
                        publish(run/'events.json', actor.events)
                        publish(out/'progress.json', dict(seed=seed, step=step+1, phase=actor.phase,
                            completed_episodes=len(results), scoring_only=current))
                        writer.log(dict(plate_count=current['plate_count'], at_rest=int(current['at_rest'])),
                            len(results)*args.horizon*60+step+1, prefix='control')
                    if args.save_debug_images and not done and (step % 300 == 299 or step == 149):
                        Image.fromarray(obs['policy']['front'][0].cpu().numpy()).save(run/f'front-{step+1:04}.png')
                        Image.fromarray(obs['policy']['wrist'][0].cpu().numpy()).save(run/f'wrist-{step+1:04}.png')
                        detections = actor.detector.detect(obs['policy']['front'][0].cpu().numpy())
                        Image.fromarray(actor.detector.annotate(obs['policy']['front'][0].cpu().numpy(), detections)).save(run/f'detections-{step+1:04}.png')
                        publish(run/'detections.json', detections)
                    if done:
                        break
            finally:
                trace.close()
                np.savez(run/'commands.npz', requested_rad=np.asarray(action_history), measured_rad=np.asarray(measured_history),
                         phase=np.asarray(phases), timestamp=np.arange(len(phases))/30)
                publish(run/'events.json', actor.events)
            result = dict(seed=seed, success=current['success'], terminated=bool(term[0]), truncated=bool(trunc[0]),
                physics_steps=step+1, seconds=(step+1)/60, final=current, policy_failure=actor.failure,
                attempted_picks=actor.attempts, completed_place_sequences=actor.completed)
            publish(run/'result.json', result)
            results.append(result)
            writer.log(dict(success=int(result['success']), success_rate=sum(x['success'] for x in results)/len(results),
                completed_episodes=len(results)), len(results), prefix='eval')
            publish(out/'result.json', dict(status='completed' if len(results)==len(args.seeds) else 'partial',
                protocol=manifest['name'], results=results, success_rate=sum(x['success'] for x in results)/len(results)))
    except BaseException:
        publish(out/'error.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close()


if __name__ == '__main__':
    main()
