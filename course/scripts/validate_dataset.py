"""Local synthetic LeRobot v3 write/read/padding test, never a demonstration."""
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path('/data/course/validation/synthetic-so101-v3')
if root.exists():
    raise RuntimeError('Retain existing evidence; choose a new validation directory before rerunning')
names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
features = {key: {'dtype': 'float32', 'shape': (6,), 'names': names} for key in ('observation.state', 'action')}
features.update({f'observation.images.{key}': {'dtype': 'image', 'shape': (64, 64, 3), 'names': ['height', 'width', 'channels']} for key in ('front', 'wrist')})
dataset = LeRobotDataset.create(repo_id='local/synthetic-so101-v3', root=root, fps=10, robot_type='so101_follower', features=features, use_videos=False)
for t in range(5):
    dataset.add_frame({'observation.state': np.full(6, t, dtype=np.float32),
                       'action': np.full(6, t+1, dtype=np.float32),
                       'observation.images.front': np.full((64, 64, 3), t*40, dtype=np.uint8),
                       'observation.images.wrist': np.full((64, 64, 3), 200-t*40, dtype=np.uint8),
                       'task': 'SYNTHETIC SOFTWARE TEST - NOT AN EXPERT DEMO'})
dataset.save_episode()
dataset.finalize()
reader = LeRobotDataset(repo_id='local/synthetic-so101-v3', root=root, delta_timestamps={'action': [0.0, 0.1, 0.2, 0.3]})
first, last = reader[0], reader[4]
assert len(reader) == 5
assert first['observation.state'].shape == (6,)
assert first['observation.images.front'].shape == (3, 64, 64)
assert first['action'].shape == (4, 6)
assert first['action_is_pad'].tolist() == [False, False, False, False]
assert last['action_is_pad'].tolist() == [False, True, True, True]
assert torch.allclose(first['action'][:, 0], torch.tensor([1., 2., 3., 4.]))
report = {'kind': 'synthetic-dataset-test-not-collected-demo', 'episodes': 1, 'frames': 5,
          'fps': 10, 'image_shape': list(first['observation.images.front'].shape),
          'action_chunk_shape': list(first['action'].shape), 'last_action_padding': last['action_is_pad'].tolist(),
          'format': reader.meta.info['codebase_version'], 'root': str(root)}
Path('/workspace/course/logs/dataset-smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
