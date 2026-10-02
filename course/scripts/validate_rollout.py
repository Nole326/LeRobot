"""Bounded public-task validation: horizon, final observation, replay and IK.

No training, no hardware, no claiming this is a successful expert episode.
Uses an independent diagnostic recorder; upstream code/config remains frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run-tag", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path('/workspace/course/logs') / args.run_tag
data = Path('/data/course/validation') / args.run_tag
out.mkdir(exist_ok=False)
data.mkdir(exist_ok=False)
start = time.monotonic()


def record(name, result):
    (out / (name + '.json')).write_text(json.dumps(result, indent=2))
    print('RESULT ' + name + ': ' + json.dumps(result), flush=True)


launcher = AppLauncher(args)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    import leisaac
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import RecorderTermCfg, DatasetExportMode
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManagerBaseCfg
    from isaaclab.utils import configclass
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv
    from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
    from joint_contract import JOINTS, USD_LIMITS_DEG, MOTOR_LIMITS, rad_to_motor, motor_to_rad

    class FinalCapture(RecorderTerm):
        def record_pre_reset(self, env_ids):
            if hasattr(self._env, 'reset_buf') and bool(self._env.reset_buf.any()):
                # obs_buf was computed pre-reset because this recorder is active.
                self._env.validation_final = {
                    key: self._env.obs_buf['policy'][key].clone()
                    for key in ('joint_pos', 'front', 'wrist')}
            return None, None

    @configclass
    class CaptureCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset = False
        final_frame = RecorderTermCfg(class_type=FinalCapture)

    cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device='cuda:0', num_envs=1)
    cfg.use_teleop_device('so101leader')
    cfg.recorders = CaptureCfg()
    env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
    robot = env.scene['robot']
    ids, names = robot.find_joints(JOINTS, preserve_order=True)
    terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
    actual_order = sum([list(t._joint_names) for t in terms], [])
    assert actual_order == JOINTS, actual_order
    assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
    offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1, t.action_dim) for t in terms], dim=1)
    scales = torch.cat([torch.as_tensor(t._scale, device=env.device).expand(1, t.action_dim) for t in terms], dim=1)
    assert bool((scales == 1).all())
    limits = torch.tensor(np.deg2rad(USD_LIMITS_DEG), device=env.device)

    def command(q_absolute):
        assert torch.isfinite(q_absolute).all()
        assert bool((q_absolute >= limits[:, 0]).all() & (q_absolute <= limits[:, 1]).all())
        return (q_absolute - offsets) / scales

    def snapshot():
        return robot.data.joint_pos[:, ids].detach().cpu().numpy()[0].copy()

    # Verify formulas against frozen upstream, including limits and midpoint.
    samples = np.deg2rad(np.stack([USD_LIMITS_DEG[:, 0], USD_LIMITS_DEG.mean(axis=1), USD_LIMITS_DEG[:, 1]]))
    upstream = convert_leisaac_action_to_lerobot(samples)
    mapped = rad_to_motor(samples)
    assert np.allclose(mapped, upstream, atol=2e-5)
    roundtrip = motor_to_rad(mapped)
    assert np.allclose(roundtrip, samples, atol=1e-6)
    assert np.allclose(convert_lerobot_action_to_leisaac(mapped), roundtrip, atol=1e-6)
    record('action-contract', {'order': names, 'units': 'absolute radians <-> normalized calibration coordinates',
                              'offset_rad': offsets[0].tolist(), 'scale': scales[0].tolist(),
                              'roundtrip_max_error_rad': float(np.max(np.abs(roundtrip - samples))),
                              'physical_motor_degrees': False})

    obs, _ = env.reset(seed=17)
    env.validation_final = None
    q_base = robot.data.joint_pos[:, ids].clone()
    total_reward = 0.0
    final_info = None
    for t in range(env.max_episode_length + 1):
        q_goal = q_base.clone()
        q_goal[0, 0] += 0.035 * np.sin(2 * np.pi * t / 240)
        q_goal[0, 5] = 0.25
        obs, reward, term, trunc, info = env.step(command(q_goal))
        assert torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(reward).all()
        total_reward += float(reward[0])
        if (t + 1) % 100 == 0:
            print(f'STAGE full episode {t+1}/{env.max_episode_length}', flush=True)
        if bool(term[0] | trunc[0]):
            assert env.validation_final is not None
            final_info = {'steps': t + 1, 'terminated': bool(term[0]), 'truncated': bool(trunc[0]),
                          'returned_episode_length_buf': int(env.episode_length_buf[0]),
                          'info_keys': sorted(info), 'has_final_observation_in_info': 'final_observation' in info,
                          'terminal_vs_returned_joint_max_rad': float((env.validation_final['joint_pos'] - obs['policy']['joint_pos']).abs().max())}
            break
    assert final_info is not None and final_info['truncated'] and final_info['steps'] == env.max_episode_length
    final_info.update({'reward_terms': list(env.reward_manager.active_terms), 'return': total_reward,
                       'physics_dt_s': env.physics_dt, 'step_dt_s': env.step_dt,
                       'success_demo': False, 'note': 'Controlled non-task trajectory; horizon and auto-reset test only'})
    record('full-episode', final_info)

    # Short real sim trajectory: actions held for six physics steps, 10 Hz BC.
    obs, _ = env.reset(seed=17)
    q_base = robot.data.joint_pos[:, ids].clone()
    hold = command(q_base)
    for _ in range(30):
        obs, *_ = env.step(hold)
    initial_state = env.scene.get_state(is_relative=True)
    initial_state = {kind: {name: {key: value.clone() for key, value in state.items()}
                            for name, state in assets.items()} for kind, assets in initial_state.items()}
    torch.save(initial_state, data / 'diagnostic-initial-state.pt')
    initial_q = snapshot()
    pre, post, raw, abs_targets = [], [], [], []
    after_each_step = []
    for frame in range(20):
        q_goal = q_base.clone()
        q_goal[0, 0] += 0.045 * np.sin(2 * np.pi * frame / 20)
        q_goal[0, 5] = 0.20 + 0.04 * np.sin(2 * np.pi * frame / 10)
        action = command(q_goal)
        pre.append(snapshot())
        raw.append(action[0].cpu().numpy().copy())
        abs_targets.append(q_goal[0].cpu().numpy().copy())
        for camera in ('front', 'wrist'):
            image = obs['policy'][camera][0].cpu().numpy()
            Image.fromarray(image).resize((128, 128)).save(data / f'{camera}-{frame:03d}.png')
        for _ in range(6):
            obs, _, term, trunc, _ = env.step(action)
            assert not bool(term.any() | trunc.any())
            after_each_step.append(snapshot())
        post.append(snapshot())
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in
              {'pre_joint_rad': pre, 'post_joint_rad': post, 'raw_action': raw, 'absolute_target_rad': abs_targets,
               'after_each_step_rad': after_each_step, 'initial_q_rad': initial_q}.items()}
    np.savez_compressed(data / 'trajectory.npz', **arrays)

    # Restore saved scene state, replay the commands, do not inject post states.
    env.reset_to(initial_state, env_ids=torch.tensor([0], device=env.device), seed=17, is_relative=True)
    replay_start_diff = float(np.max(np.abs(snapshot() - initial_q)))
    replay = []
    for action in arrays['raw_action']:
        tensor = torch.tensor(action[None], device=env.device)
        for _ in range(6):
            obs, *_ = env.step(tensor)
            replay.append(snapshot())
    differences = np.asarray(replay) - arrays['after_each_step_rad']
    replay_report = {'steps': len(replay), 'start_max_error_rad': replay_start_diff,
                     'rmse_rad': float(np.sqrt(np.mean(differences ** 2))),
                     'max_error_rad': float(np.max(np.abs(differences))),
                     'tolerance_rad': 0.01, 'within_tolerance': bool(np.max(np.abs(differences)) <= 0.01),
                     'success_demo': False, 'image_bitwise_replay_claimed': False,
                     'scene_state_is_not_full_physics_rng_checkpoint': True}
    record('replay', replay_report)

    # Test IK math on actual articulation Jacobian without changing actuators.
    body_ids, _ = robot.find_bodies('gripper')
    body_id = body_ids[0]
    p, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                       robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id])
    jac = robot.root_physx_view.get_jacobians()[:, body_id - 1, :, ids[:5]].clone()
    rot = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jac[:, :3] = torch.bmm(rot, jac[:, :3])
    jac[:, 3:] = torch.bmm(rot, jac[:, 3:])
    target = torch.cat([p.clone(), quat.clone()], dim=1)
    target[:, 0] += 0.001
    controller = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls'), num_envs=1, device=env.device)
    controller.set_command(target, p, quat)
    q_next = controller.compute(p, quat, jac, robot.data.joint_pos[:, ids[:5]])
    assert torch.isfinite(q_next).all()
    record('ik-api', {'jacobian_shape': list(jac.shape), 'singular_values': torch.linalg.svdvals(jac)[0].tolist(),
                      'requested_base_frame_dx_m': 0.001, 'max_joint_change_rad': float((q_next - robot.data.joint_pos[:, ids[:5]]).abs().max()),
                      'finite': True, 'applied_to_sim': False, 'course_table_frame_controller_validated': False,
                      'note': '5 arm joints do not provide arbitrary 6-DoF pose reachability'})

    manifest = {'kind': 'controlled-real-sim-trajectory-not-expert-demo', 'seed': 17, 'frames': 20,
                'fps': 10, 'physics_hz': 60, 'hold_steps': 6, 'joint_names': JOINTS,
                'timing': 'image/q before action; post q after six steps; no reset transition',
                'state_input': 'actual measured six-joint positions; no object or target ground truth',
                'action': 'absolute joint radians, converted to normalized calibration coordinates for BC',
                'camera_test_resize': [128, 128], 'task': 'SOFTWARE VALIDATION - NOT EXPERT DEMO',
                'trajectory_sha256': hashlib.sha256((data / 'trajectory.npz').read_bytes()).hexdigest(),
                'replay': replay_report}
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    record('summary', {'status': 'completed', 'data_path': str(data), 'elapsed_s': time.monotonic() - start,
                       'formal_training': False, 'simulated_steps': final_info['steps'] + 30 + 120 + 120})
except BaseException:
    diagnostic = traceback.format_exc()
    (out / 'error.txt').write_text(diagnostic)
    print(diagnostic, flush=True)
    raise
finally:
    if env is not None:
        env.close()
    app.close()
