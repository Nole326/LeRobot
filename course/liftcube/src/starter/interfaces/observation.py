"""Policy-visible observation schema. Privileged GT must never appear here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROPRIO_LAYOUT: tuple[str, ...] = (
    "joint_pos[6]",
    "joint_vel[6]",
    "last_action[6]",
    "ee_pos[3]  # optional robot EE, never object GT",
)


@dataclass(frozen=True)
class ObservationSpec:
    image_shape: tuple[int, int, int] = (64, 64, 3)  # H, W, C
    image_dtype: str = "uint8"
    image_layout: str = "HWC"
    camera: str = "front"
    proprio_dim: int = 21
    proprio_layout: tuple[str, ...] = PROPRIO_LAYOUT
    include_ee_in_proprio: bool = True
    control_hz: float = 30.0

    def to_dict(self) -> dict:
        return {
            "image_shape": list(self.image_shape),
            "image_dtype": self.image_dtype,
            "image_layout": self.image_layout,
            "camera": self.camera,
            "proprio_dim": self.proprio_dim,
            "proprio_layout": list(self.proprio_layout),
            "include_ee_in_proprio": self.include_ee_in_proprio,
            "control_hz": self.control_hz,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationSpec":
        shape = data.get("image_shape", (64, 64, 3))
        return cls(
            image_shape=tuple(int(x) for x in shape),
            image_dtype=str(data.get("image_dtype", "uint8")),
            image_layout=str(data.get("image_layout", "HWC")),
            camera=str(data.get("camera", "front")),
            proprio_dim=int(data.get("proprio_dim", 21)),
            proprio_layout=tuple(data.get("proprio_layout", PROPRIO_LAYOUT)),
            include_ee_in_proprio=bool(data.get("include_ee_in_proprio", True)),
            control_hz=float(data.get("control_hz", 30.0)),
        )


@dataclass
class Observation:
    """Student-policy observation. Metadata must not contain object/target GT."""

    image: np.ndarray
    proprio: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        self.proprio = np.asarray(self.proprio, dtype=np.float32).reshape(-1)

    def validate(self, spec: ObservationSpec) -> None:
        if self.image.shape != spec.image_shape:
            raise ValueError(f"image shape {self.image.shape} != {spec.image_shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"image dtype {self.image.dtype} != uint8")
        if self.proprio.shape != (spec.proprio_dim,):
            raise ValueError(f"proprio dim {self.proprio.shape} != {(spec.proprio_dim,)}")
        if not np.all(np.isfinite(self.proprio)):
            raise ValueError("proprio contains non-finite values")


def pack_proprio(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    ee_pos: np.ndarray | None = None,
    *,
    include_ee: bool = True,
) -> np.ndarray:
    parts = [
        np.asarray(joint_pos, dtype=np.float32).reshape(6),
        np.asarray(joint_vel, dtype=np.float32).reshape(6),
        np.asarray(last_action, dtype=np.float32).reshape(6),
    ]
    if include_ee:
        if ee_pos is None:
            ee_pos = np.zeros(3, dtype=np.float32)
        parts.append(np.asarray(ee_pos, dtype=np.float32).reshape(3))
    return np.concatenate(parts, axis=0)


def proprio_dim(include_ee: bool) -> int:
    return 21 if include_ee else 18
