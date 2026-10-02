"""Public PickOrange reset/step/RGB smoke; not a benchmark or course controller."""
import argparse
import json
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
start = time.monotonic()
launcher = AppLauncher(args)
app = launcher.app
try:
    print('STAGE: SimulationApp initialized', flush=True)
    import gymnasium as gym
    import torch
    import leisaac
    print('STAGE: LeIsaac imported from ' + str(leisaac.__file__), flush=True)
    assert leisaac.__file__ is not None, 'Repository namespace shadowed installed package'
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device='cuda:0', num_envs=1)
    print('STAGE: task config resolved', flush=True)
    cfg.use_teleop_device('so101leader')
    cfg.recorders = {}
    env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
    print('STAGE: task created', flush=True)
    try:
        obs, info = env.reset(seed=1)
        robot = env.scene['robot']
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        ids, names = robot.find_joints(joint_names, preserve_order=True)
        target = (robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]).clone()
        assert target.shape == (1, 6) and env.action_manager.total_action_dim == 6
        frames = {k: obs['policy'][k].clone() for k in ('front', 'wrist')}
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step(target)
            assert torch.isfinite(robot.data.joint_pos).all()
        cameras = {}
        for key in ('front', 'wrist'):
            rgb = obs['policy'][key]
            assert rgb.shape == (1, 480, 640, 3)
            assert torch.isfinite(rgb).all() and float(rgb.float().std()) > 1
            cameras[key] = {'shape': list(rgb.shape), 'dtype': str(rgb.dtype),
                            'min': float(rgb.min()), 'max': float(rgb.max()),
                            'std': float(rgb.float().std()),
                            'changed_fraction': float((rgb != frames[key]).float().mean())}
            from PIL import Image
            Image.fromarray(rgb[0].cpu().numpy()).save(f'/workspace/course/logs/sim-{key}.png')
        obs, _ = env.reset(seed=2)
        for _ in range(5):
            obs, *_ = env.step(torch.zeros_like(target))
        report = {'kind': 'public-joint-controller-smoke-not-course-eval',
                  'task': 'LeIsaac-SO101-PickOrange-v0', 'num_envs': 1,
                  'reset_seeds': [1, 2], 'steps': 65, 'action_dim': 6,
                  'action_interface': 'so101leader joint target; offsets preserved',
                  'disable_gravity': cfg.scene.robot.spawn.rigid_props.disable_gravity,
                  'physics_dt': env.physics_dt, 'step_dt': env.step_dt,
                  'episode_length_s': cfg.episode_length_s, 'max_episode_length': env.max_episode_length,
                  'policy_keys': sorted(obs['policy']), 'observation_groups': sorted(obs),
                  'reward_terms': list(env.reward_manager.active_terms),
                  'cameras': cameras, 'joint_names': names,
                  'elapsed_s': time.monotonic()-start,
                  'cuda_allocated_mib': torch.cuda.memory_allocated()/2**20,
                  'cuda_reserved_mib': torch.cuda.memory_reserved()/2**20}
    finally:
        env.close()
    Path('/workspace/course/logs/sim-smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
except BaseException:
    diagnostic = traceback.format_exc()
    Path('/workspace/course/logs/sim-smoke-error.txt').write_text(diagnostic)
    print(diagnostic, flush=True)
    raise
finally:
    app.close()
