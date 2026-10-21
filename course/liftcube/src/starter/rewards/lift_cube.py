"""LiftCube reward used by PPO.

Replace the body of :func:`shape_reward`. The value it returns is what PPO
learns. ``LiftCubeEnvAdapter.step`` keeps returning 0 or 1, and evaluation
still reads ``info["success"]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from starter.envs.lift_cube_privileged import LiftCubePrivilegedState


def shape_reward(
    *,
    privileged: LiftCubePrivilegedState,
    public_reward: float,
    action: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> float:
    """Student reward for one LiftCube transition.

    ``privileged`` is the post-step scene from ``env.get_privileged_state()``:

    - ``cube_pos``, ``robot_base_pos``, ``ee_jaw_pos``: env-frame meters
    - ``gripper_joint_rad``: simulator gripper joint, radians
    - success height is ``cube_pos[2] - robot_base_pos[2] > 0.20``

    ``public_reward`` is 1.0 only on that success, otherwise 0.0.
    ``action`` is the 6-D motor-unit command sent to ``step``.
    """
    del privileged, action, terminated, truncated, info
    if not np.isfinite(public_reward):
        raise ValueError("public_reward is non-finite")
    return float(public_reward)
