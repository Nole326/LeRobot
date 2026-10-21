"""Canonical action schema for the SO-101 follower (joint position targets).

Policy and dataset actions use LeRobot SO-101 motor normalization
(``use_degrees=False`` body joints, gripper ``RANGE_0_100``), matching
``SO101_FOLLOWER_MOTOR_LIMITS`` in LeIsaac. LiftCube uses the same 6-D arm.
Isaac ``set_joint_position_target`` wants radians; convert with
``motor_to_radians`` at the env boundary using the per-joint USD degree
limits. Do not treat these motor values as degrees or as a uniform ±π/2 box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# LeRobot motor units. Arm: full calibrated travel mapped to [-100, 100].
# Gripper: 0 at the low USD stop, 100 at the high USD stop.
DEFAULT_LOW = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0], dtype=np.float32)
DEFAULT_HIGH = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)

# SO101_FOLLOWER_USD_JOINT_LIMITS from LeIsaac, degrees. Order matches JOINT_NAMES.
USD_LOW_DEG = np.array([-110.0, -100.0, -100.0, -95.0, -160.0, -10.0], dtype=np.float32)
USD_HIGH_DEG = np.array([110.0, 100.0, 90.0, 95.0, 160.0, 100.0], dtype=np.float32)

DEFAULT_UNITS: tuple[str, ...] = (
    "motor_-100_100",
    "motor_-100_100",
    "motor_-100_100",
    "motor_-100_100",
    "motor_-100_100",
    "motor_0_100",
)


@dataclass(frozen=True)
class ActionSpec:
    """Explicit action contract. Do not pass unexplained ndarrays around."""

    kind: str = "joint_position"
    names: tuple[str, ...] = JOINT_NAMES
    dim: int = 6
    units: tuple[str, ...] = DEFAULT_UNITS
    low: np.ndarray = field(default_factory=lambda: DEFAULT_LOW.copy())
    high: np.ndarray = field(default_factory=lambda: DEFAULT_HIGH.copy())
    frequency_hz: float = 30.0
    normalized: bool = False
    normalize_mean: np.ndarray | None = None
    normalize_std: np.ndarray | None = None
    notes: str = (
        "6-D joint position targets in LeRobot motor units: arm [-100, 100], "
        "gripper [0, 100]. Isaac receives per-joint radians via motor_to_radians "
        "(USD degree limits, not a uniform ±π/2 clip)."
    )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "names": list(self.names),
            "dim": self.dim,
            "units": list(self.units),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "frequency_hz": self.frequency_hz,
            "normalized": self.normalized,
            "normalize_mean": None if self.normalize_mean is None else self.normalize_mean.tolist(),
            "normalize_std": None if self.normalize_std is None else self.normalize_std.tolist(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSpec":
        return cls(
            kind=data.get("kind", "joint_position"),
            names=tuple(data.get("names", JOINT_NAMES)),
            dim=int(data.get("dim", 6)),
            units=tuple(data.get("units", DEFAULT_UNITS)),
            low=np.asarray(data.get("low", DEFAULT_LOW), dtype=np.float32),
            high=np.asarray(data.get("high", DEFAULT_HIGH), dtype=np.float32),
            frequency_hz=float(data.get("frequency_hz", 30.0)),
            normalized=bool(data.get("normalized", False)),
            normalize_mean=(
                None if data.get("normalize_mean") is None else np.asarray(data["normalize_mean"], dtype=np.float32)
            ),
            normalize_std=(
                None if data.get("normalize_std") is None else np.asarray(data["normalize_std"], dtype=np.float32)
            ),
            notes=str(data.get("notes", "")),
        )


def validate_action(action: np.ndarray, spec: ActionSpec, *, clip: bool = False) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != spec.dim:
        raise ValueError(f"Action dim {arr.shape[-1]} != spec.dim {spec.dim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Action contains non-finite values")
    if clip:
        arr = np.clip(arr, spec.low, spec.high)
    else:
        if np.any(arr < spec.low - 1e-4) or np.any(arr > spec.high + 1e-4):
            raise ValueError(
                f"Action out of range: got {arr}, low={spec.low}, high={spec.high}"
            )
    return arr


def clip_action(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    return validate_action(action, spec, clip=True)


def denormalize_action(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if spec.normalize_mean is not None and spec.normalize_std is not None:
        return arr * np.maximum(spec.normalize_std, 1e-6) + spec.normalize_mean
    return spec.low + (arr + 1.0) * 0.5 * (spec.high - spec.low)


def normalize_action(action: np.ndarray, spec: ActionSpec) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if spec.normalize_mean is not None and spec.normalize_std is not None:
        return (arr - spec.normalize_mean) / np.maximum(spec.normalize_std, 1e-6)
    return 2.0 * (arr - spec.low) / np.maximum(spec.high - spec.low, 1e-6) - 1.0


def _as_joint_batch(action: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(action, dtype=np.float32)
    squeezed = arr.ndim == 1
    if squeezed:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != DEFAULT_LOW.shape[0]:
        raise ValueError(f"Expected {DEFAULT_LOW.shape[0]} joints, got shape {arr.shape}")
    return arr, squeezed


def motor_to_degrees(action: np.ndarray) -> np.ndarray:
    """Map LeRobot motor units onto SO-101 USD joint limits, in degrees."""
    arr, squeezed = _as_joint_batch(action)
    motor_span = np.maximum(DEFAULT_HIGH - DEFAULT_LOW, 1e-6)
    joint_span = USD_HIGH_DEG - USD_LOW_DEG
    degrees = (arr - DEFAULT_LOW) / motor_span * joint_span + USD_LOW_DEG
    return degrees[0] if squeezed else degrees


def degrees_to_motor(degrees: np.ndarray) -> np.ndarray:
    """Inverse of ``motor_to_degrees``."""
    arr, squeezed = _as_joint_batch(degrees)
    motor_span = DEFAULT_HIGH - DEFAULT_LOW
    joint_span = np.maximum(USD_HIGH_DEG - USD_LOW_DEG, 1e-6)
    motor = (arr - USD_LOW_DEG) / joint_span * motor_span + DEFAULT_LOW
    return motor[0] if squeezed else motor


def motor_to_radians(action: np.ndarray) -> np.ndarray:
    """LeRobot motor units → Isaac joint-position target (radians)."""
    radians = np.deg2rad(motor_to_degrees(action)).astype(np.float32)
    return radians


def radians_to_motor(radians: np.ndarray) -> np.ndarray:
    """Isaac joint position (radians) → LeRobot motor units."""
    return degrees_to_motor(np.rad2deg(radians).astype(np.float32))


def radians_velocity_to_motor(velocity: np.ndarray) -> np.ndarray:
    """Scale rad/s joint velocity into motor-units/s with the same linear map."""
    arr, squeezed = _as_joint_batch(velocity)
    motor_span = DEFAULT_HIGH - DEFAULT_LOW
    joint_span = np.maximum(USD_HIGH_DEG - USD_LOW_DEG, 1e-6)
    scale = motor_span / joint_span * (180.0 / np.pi)
    out = (arr * scale).astype(np.float32)
    return out[0] if squeezed else out


def as_1d(action: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return arr
