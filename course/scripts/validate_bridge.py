"""Actual sim sample -> LeRobot v3 -> ACT CPU forward -> sim action units.

Only runs after validate_rollout produced complete manifest. No optimization,
policy deployment, upload, robot hardware or formal expert dataset creation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from joint_contract import rad_to_motor, motor_to_rad, JOINTS

parser = argparse.ArgumentParser()
parser.add_argument('--source-tag', required=True)
parser.add_argument('--output-tag', required=True)
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(11)
start = time.monotonic()
source = Path('/data/course/validation') / args.source_tag
dest = Path('/data/course/validation') / args.output_tag
manifest = json.loads((source / 'manifest.json').read_text())
assert manifest['kind'] == 'controlled-real-sim-trajectory-not-expert-demo'
assert manifest['frames'] == 20 and manifest['fps'] == 10
trajectory = source / 'trajectory.npz'
assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == manifest['trajectory_sha256']
assert not dest.exists(), 'Do not replace prior validation evidence'
with np.load(trajectory) as arrays:
    pre = arrays['pre_joint_rad'].copy()
    post = arrays['post_joint_rad'].copy()
    commands = arrays['absolute_target_rad'].copy()
    raw = arrays['raw_action'].copy()
assert np.allclose(pre[1:], post[:-1], atol=1e-7), 'Transition continuity changed'
assert pre.shape == post.shape == commands.shape == raw.shape == (20, 6)
assert np.isfinite(pre).all() and np.isfinite(commands).all()
state = rad_to_motor(pre)
actions = rad_to_motor(commands)
assert np.allclose(motor_to_rad(actions), commands, atol=1e-6)
names = [j + '.pos' for j in JOINTS]
features = {key: {'dtype': 'float32', 'shape': (6,), 'names': names} for key in ('observation.state', 'action')}
features.update({f'observation.images.{cam}': {'dtype': 'image', 'shape': (128, 128, 3),
                                            'names': ['height', 'width', 'channels']} for cam in ('front', 'wrist')})
dataset = LeRobotDataset.create(repo_id='local/' + args.output_tag, root=dest, fps=10,
                                robot_type='so101_follower', features=features, use_videos=False)
for i in range(20):
    frame = {'observation.state': state[i], 'action': actions[i],
             'task': 'CONTROLLED SIM SOFTWARE VALIDATION - NOT EXPERT DEMO'}
    for cam in ('front', 'wrist'):
        frame[f'observation.images.{cam}'] = np.asarray(Image.open(source / f'{cam}-{i:03d}.png').convert('RGB'))
    dataset.add_frame(frame)
dataset.save_episode()
dataset.finalize()
reader = LeRobotDataset(repo_id='local/' + args.output_tag, root=dest,
                        delta_timestamps={'action': [0., 0.1, 0.2, 0.3]})
assert len(reader) == 20
for i in range(20):
    item = reader[i]
    assert torch.allclose(item['observation.state'], torch.tensor(state[i]), atol=1e-5)
    assert torch.allclose(item['action'][0], torch.tensor(actions[i]), atol=1e-5)
    assert abs(float(item['timestamp']) - i / 10) < 1e-6
    assert item['action_is_pad'].tolist() == [i + j >= 20 for j in range(4)]
    for cam in ('front', 'wrist'):
        image = item[f'observation.images.{cam}']
        assert image.shape == (3, 128, 128) and torch.isfinite(image).all()
        assert float(image.min()) >= 0 and float(image.max()) <= 1

# Normalize only this test sample set; these are not deployable dataset stats.
mean_s, std_s = torch.tensor(state.mean(0)), torch.tensor(state.std(0)).clamp_min(1e-3)
mean_a, std_a = torch.tensor(actions.mean(0)), torch.tensor(actions.std(0)).clamp_min(1e-3)
items = [reader[0], reader[18]]
batch = {key: torch.stack([item[key] for item in items]) for key in features}
batch['action_is_pad'] = torch.stack([item['action_is_pad'] for item in items])
batch['observation.state'] = (batch['observation.state'] - mean_s) / std_s
batch['action'] = (batch['action'] - mean_a) / std_a
for cam in ('front', 'wrist'):
    key = f'observation.images.{cam}'
    batch[key] = (batch[key] - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
cfg = ACTConfig(input_features={
    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    'observation.images.front': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    'observation.images.wrist': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))},
    output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    pretrained_backbone_weights=None, chunk_size=4, n_action_steps=1, dim_model=128, n_heads=4,
    dim_feedforward=256, n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1, device='cpu')
policy = ACTPolicy(cfg)
with torch.no_grad():
    policy.train()
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    policy.eval()
    policy.reset()
    output = policy.select_action(batch)
assert output.shape == (2, 6) and torch.isfinite(output).all()
output_motor = (output * std_a + mean_a).numpy()
output_rad = motor_to_rad(output_motor)
assert np.isfinite(output_rad).all()
report = {'status': 'passed', 'source_sha256': manifest['trajectory_sha256'], 'frames': len(reader),
          'format': reader.meta.info['codebase_version'], 'fps': 10, 'BC_input': 'pre-action RGB + actual joints only',
          'BC_target': 'subsequent absolute command, not subsequent measured state',
          'post_state_continuity_checked': True, 'all_chunk_padding_checked': True,
          'ACT_shape': list(output.shape), 'finite_CPU_forward_loss': float(loss),
          'optimizer_steps': 0, 'untrained_policy_applied_to_sim': False,
          'observation_ground_truth_used': False, 'full_sim_policy_loop_validated': False,
          'elapsed_s': time.monotonic() - start, 'root': str(dest)}
Path('/workspace/course/logs/' + args.output_tag + '.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
