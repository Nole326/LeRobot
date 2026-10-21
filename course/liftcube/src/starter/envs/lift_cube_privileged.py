"""LiftCube-only privileged scene state.

Produced by :meth:`LiftCubeEnvAdapter.get_privileged_state` for reward code.
These fields are not part of the policy observation. The public step reward
is 1 only when the cube is at least 0.20 m above the robot base, otherwise 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CUBE_NAME = "cube"
ROBOT_NAME = "robot"
ROBOT_BASE_NAME = "base"
HEIGHT_THRESHOLD_M = 0.20


@dataclass(frozen=True)
class LiftCubePrivilegedStateSpec:
    """Layout of :meth:`LiftCubeEnvAdapter.get_privileged_state`."""

    cube_name: str = CUBE_NAME
    robot_name: str = ROBOT_NAME
    robot_base_name: str = ROBOT_BASE_NAME
    position_dim: int = 3
    position_frame: str = "env"
    position_units: str = "m"
    gripper_unit: str = "rad"
    ee_frame: str = "jaw"
    height_threshold_m: float = HEIGHT_THRESHOLD_M
    fields: tuple[str, ...] = (
        "cube_pos",
        "robot_base_pos",
        "ee_jaw_pos",
        "gripper_joint_rad",
    )
    notes: str = (
        "LiftCube only. Env-frame meters: world position minus env_origins. "
        "Task success is cube_pos[2] - robot_base_pos[2] > 0.20. "
        "ee_jaw_pos is ee_frame target index 1. "
        "gripper_joint_rad is the simulator joint, not the policy's 0-100 motor unit. "
        "The public step reward does not use these fields. Do not copy them into Observation."
    )

    def to_dict(self) -> dict:
        return {
            "cube_name": self.cube_name,
            "robot_name": self.robot_name,
            "robot_base_name": self.robot_base_name,
            "position_dim": self.position_dim,
            "position_frame": self.position_frame,
            "position_units": self.position_units,
            "gripper_unit": self.gripper_unit,
            "ee_frame": self.ee_frame,
            "height_threshold_m": self.height_threshold_m,
            "fields": list(self.fields),
            "notes": self.notes,
        }


@dataclass
class LiftCubePrivilegedState:
    """Cube, base, and jaw state for a LiftCube reward. Not a policy observation."""

    cube_pos: np.ndarray
    robot_base_pos: np.ndarray
    ee_jaw_pos: np.ndarray
    gripper_joint_rad: float

    def validate(self, spec: LiftCubePrivilegedStateSpec | None = None) -> None:
        spec = LiftCubePrivilegedStateSpec() if spec is None else spec
        cube = np.asarray(self.cube_pos, dtype=np.float32)
        base = np.asarray(self.robot_base_pos, dtype=np.float32)
        jaw = np.asarray(self.ee_jaw_pos, dtype=np.float32)
        dim = (spec.position_dim,)
        if cube.shape != dim:
            raise ValueError(f"cube_pos shape {cube.shape} != {dim}")
        if base.shape != dim:
            raise ValueError(f"robot_base_pos shape {base.shape} != {dim}")
        if jaw.shape != dim:
            raise ValueError(f"ee_jaw_pos shape {jaw.shape} != {dim}")
        if not np.isfinite(cube).all() or not np.isfinite(base).all() or not np.isfinite(jaw).all():
            raise ValueError("privileged positions contain non-finite values")
        if not np.isfinite(self.gripper_joint_rad):
            raise ValueError("gripper_joint_rad is non-finite")
        self.cube_pos = cube
        self.robot_base_pos = base
        self.ee_jaw_pos = jaw
        self.gripper_joint_rad = float(self.gripper_joint_rad)
