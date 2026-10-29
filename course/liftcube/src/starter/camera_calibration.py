"""Opt-in simulator-pose calibration; this auxiliary input influences the policy."""
import numpy as np


def read_camera_to_base(env):
    """Read camera/base transforms only; no objects, segmentation or scoring.

    An opt-in adapter for the matching runtime, not a promised hardware API.
    camera.data can retain nominal poses after the native randomizer runs.
    """
    from scipy.spatial.transform import Rotation
    scene = env._unwrapped.scene
    camera, robot = scene['front'], scene['robot']
    positions, orientations = camera._view.get_world_poses()  # OpenGL wxyz.
    position = positions[0].detach().cpu().numpy()
    quat = orientations[0].detach().cpu().numpy()
    index = list(robot.body_names).index('base')
    base_position = robot.data.body_pos_w[0, index].detach().cpu().numpy()
    base_quat = robot.data.body_quat_w[0, index].detach().cpu().numpy()
    cam_R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix() @ np.diag([1., -1., -1.])
    base_R = Rotation.from_quat(base_quat[[1, 2, 3, 0]]).as_matrix()
    result = np.eye(4)
    result[:3, :3] = base_R.T @ cam_R
    result[:3, 3] = base_R.T @ (position-base_position)
    if not np.isfinite(result).all() or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError('Invalid actual camera/base calibration')
    return result
