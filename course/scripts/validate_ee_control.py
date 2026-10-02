"""Development-only constrained base-frame XY controller in simulation.

This is not the unpublished course table-frame controller. Keeps initial
height/orientation targets, derives bounded joint commands from measured
proprioception, does not use object ground truth or teleport during rollout.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--run-tag', required=True)
parser.add_argument('--joint-update', choices=['measured', 'command_integral'], default='measured')
parser.add_argument('--settle-steps', type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path('/workspace/course/logs') / args.run_tag
out.mkdir(exist_ok=False)
start = time.monotonic()
launcher = AppLauncher(args)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import leisaac
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv, compute_pose_error
    from joint_contract import JOINTS, USD_LIMITS_DEG

    cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device='cuda:0', num_envs=1)
    cfg.use_teleop_device('so101leader')
    cfg.recorders = {}
    env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is False
    robot = env.scene['robot']
    ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    body_ids, _ = robot.find_bodies('gripper')
    body_id = body_ids[0]
    terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
    assert sum([list(t._joint_names) for t in terms], []) == JOINTS
    offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1, t.action_dim) for t in terms], dim=1)
    limits = torch.tensor(np.deg2rad(USD_LIMITS_DEG), device=env.device)
    env.reset(seed=19)
    hold = robot.data.joint_pos[:, ids].clone() - offsets
    for _ in range(30):
        obs, *_ = env.step(hold)

    def pose():
        return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id])

    initial_p, initial_quat = pose()
    initial_p, initial_quat = initial_p.clone(), initial_quat.clone()
    target_p, target_quat = initial_p.clone(), initial_quat.clone()
    controller = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls'), num_envs=1, device=env.device)
    previous_command = robot.data.joint_pos[:, ids].clone()
    records = []
    rate_hits = 0
    max_command_delta = 0.0
    assert 0 <= args.settle_steps <= 240
    for t in range(240 + args.settle_steps):
        # 2.4mm excursion in each base-frame direction; return to initial XY.
        xy = torch.zeros(1, 2, device=env.device)
        phase = (t-args.settle_steps) // 60 if t >= args.settle_steps else -1
        if phase >= 0:
            xy[0, phase // 2] = 0.00004 * (1 if phase % 2 == 0 else -1)
        target_p[:, :2] += xy
        p, quat = pose()
        target = torch.cat([target_p, target_quat], dim=-1)
        controller.set_command(target, p, quat)
        jac = robot.root_physx_view.get_jacobians()[:, body_id - 1, :, ids[:5]].clone()
        rot = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jac[:, :3] = torch.bmm(rot, jac[:, :3])
        jac[:, 3:] = torch.bmm(rot, jac[:, 3:])
        wanted = previous_command.clone()
        measured_q = robot.data.joint_pos[:, ids[:5]]
        solved_q = controller.compute(p, quat, jac, measured_q)
        if args.joint_update == 'measured':
            wanted[:, :5] = solved_q
        else:
            # Isolated diagnostic alternative: integrate the pose correction
            # into the previous servo setpoint; do not change actuator gains.
            wanted[:, :5] = previous_command[:, :5] + 0.25 * (solved_q - measured_q)
        wanted[:, 5] = 0.25
        assert torch.isfinite(wanted).all()
        delta = wanted - previous_command
        rate_hits += int(bool((delta.abs() > 0.005).any()))
        # Explicit development safety limits: command change <=0.005rad/step.
        q_command = previous_command + delta.clamp(-0.005, 0.005)
        q_command = torch.maximum(torch.minimum(q_command, limits[:, 1]), limits[:, 0])
        max_command_delta = max(max_command_delta, float((q_command-previous_command).abs().max()))
        assert max_command_delta <= 0.005001
        obs, reward, term, trunc, info = env.step(q_command-offsets)
        assert not bool(term.any() | trunc.any())
        assert torch.isfinite(robot.data.joint_pos).all()
        previous_command = q_command.clone()
        p_after, quat_after = pose()
        pos_error, rot_error = compute_pose_error(p_after, quat_after, target_p, target_quat)
        records.append({'step': t+1, 'phase': phase, 'target_xyz_base_m': target_p[0].tolist(),
                        'actual_xyz_base_m': p_after[0].tolist(),
                        'position_error_m': float(pos_error.norm()), 'orientation_error_rad': float(rot_error.norm()),
                        'height_error_m': float((p_after[:, 2]-initial_p[:, 2]).abs().max()),
                        'joint_target_rad': q_command[0].tolist(),
                        'joint_actual_rad': robot.data.joint_pos[0, ids].tolist(),
                        'arm_servo_error_max_rad': float((q_command[:, :5]-robot.data.joint_pos[:, ids[:5]]).abs().max())})
        if (t+1) % 60 == 0:
            print('STAGE ' + json.dumps(records[-1]), flush=True)
    max_position = max(r['position_error_m'] for r in records)
    max_height = max(r['height_error_m'] for r in records)
    max_orientation = max(r['orientation_error_rad'] for r in records)
    movement = [r for r in records if r['phase'] >= 0]
    movement_max_position = max(r['position_error_m'] for r in movement)
    movement_max_height = max(r['height_error_m'] for r in movement)
    movement_max_orientation = max(r['orientation_error_rad'] for r in movement)
    settled = [r for r in records if r['phase'] < 0][-30:]
    readiness = bool(settled) and all(r['position_error_m'] <= 0.001 and r['orientation_error_rad'] <= 0.01 for r in settled)
    result = {'status': 'completed', 'kind': 'development-constrained-base-XY-control-not-course-eval',
              'steps': len(records), 'warmup_steps': 30, 'controller_settle_steps': args.settle_steps, 'action_dim': 3,
              'action_semantics': 'base-frame dx_m, dy_m, absolute gripper_rad; fixed reset height/orientation targets',
              'gravity_enabled': True, 'object_ground_truth_used': False, 'joint_teleport_in_rollout': False,
              'joint_update': args.joint_update,
              'integration_gain': 0.25 if args.joint_update == 'command_integral' else None,
              'max_arm_servo_error_rad': max(r['arm_servo_error_max_rad'] for r in records),
              'requested_xy_excursion_m': 0.0024, 'max_joint_command_delta_rad': max_command_delta,
              'command_rate_cap_rad_s': 0.3, 'rate_limit_hit_steps': rate_hits,
              'max_position_error_m': max_position, 'max_height_error_m': max_height,
              'max_orientation_error_rad': max_orientation,
              'final_position_error_m': records[-1]['position_error_m'],
              'final_orientation_error_rad': records[-1]['orientation_error_rad'],
              'tracking_within_test_limits': bool(max_position <= 0.01 and max_height <= 0.005 and max_orientation <= 0.05),
              'readiness_passed_last_30_settle_steps': readiness,
              'post_settle_max_position_error_m': movement_max_position,
              'post_settle_max_height_error_m': movement_max_height,
              'post_settle_max_orientation_error_rad': movement_max_orientation,
              'post_settle_tracking_within_test_limits': bool(readiness and movement_max_position <= 0.01 and movement_max_height <= 0.005 and movement_max_orientation <= 0.05),
              'test_limits': {'position_m': 0.01, 'height_m': 0.005, 'orientation_rad': 0.05},
              'course_table_transform_available': False, 'course_controller_validated': False,
              'formal_training': False, 'elapsed_s': time.monotonic()-start}
    (out/'trajectory.json').write_text(json.dumps(records, indent=2))
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
except BaseException:
    error = traceback.format_exc()
    (out/'error.txt').write_text(error)
    print(error, flush=True)
    raise
finally:
    if env is not None:
        env.close()
    app.close()
