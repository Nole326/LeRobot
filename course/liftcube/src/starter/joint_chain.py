"""Fixed SO101 joint-chain FK/IK; no simulator imports or scene-state access."""
import json
from pathlib import Path
import numpy as np
from .interfaces.action import USD_LOW_DEG, USD_HIGH_DEG
from .scripted_liftcube import _array, _rotation


def transform(value, name):
    result = _array(value, (4, 4), name)
    _rotation(result[:3, :3], name)
    if not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous transform: {name}")
    return result


class JointChainKinematics:
    def __init__(self, path, tcp_offset_m, joint_names):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema') != 'so101-joint-chain-v1' or data.get('meters_per_unit') != 1:
            raise ValueError('Expected a metre-scale SO101 joint chain')
        if len(data.get('asset_sha256', '')) != 64:
            raise ValueError('Joint chain requires source asset SHA256')
        self.asset_sha256 = data['asset_sha256']
        joints = data['joints']
        if len(joints) != 5 or [j['name'] for j in joints] != list(joint_names):
            raise ValueError('Five ordered arm joints must match calibration')
        self.parents, self.children = [], []
        for index, joint in enumerate(joints):
            if joint['axis'] != 'Z':
                raise ValueError('This SO101 chain requires local Z revolute axes')
            if not np.allclose([joint['lower'], joint['upper']],
                               [USD_LOW_DEG[index], USD_HIGH_DEG[index]], atol=1e-5):
                raise ValueError('Joint limits differ from the public SO101 mapping')
            self.parents.append(transform(joint['parent_frame'], 'parent_frame'))
            self.children.append(np.linalg.inv(transform(joint['child_frame'], 'child_frame')))
        self.tcp = _array(tcp_offset_m, (3,), 'tcp_offset_m')

    def forward_kinematics(self, degrees):
        return self._forward_radians(np.deg2rad(_array(degrees, (5,), 'joint degrees')))

    def _forward_radians(self, q):
        from scipy.spatial.transform import Rotation
        pose = np.eye(4)
        for angle, parent, child in zip(q, self.parents, self.children):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
            pose = pose @ parent @ rotation @ child
        pose[:3, 3] += pose[:3, :3] @ self.tcp
        return pose

    def inverse_kinematics(self, degrees, target, orientation_weight=1.0):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        target = transform(target, 'IK target')
        initial = np.deg2rad(_array(degrees, (5,), 'IK initial degrees'))

        def residual(radians):
            pose = self._forward_radians(radians)
            return np.r_[pose[:3, 3]-target[:3, 3],
                .1*orientation_weight*Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()]

        fit = least_squares(residual, initial,
            bounds=(np.deg2rad(USD_LOW_DEG[:5].astype(float)),
                    np.deg2rad(USD_HIGH_DEG[:5].astype(float))), max_nfev=100)
        return np.rad2deg(fit.x)
