"""Canonical episode schema for BC datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from starter.interfaces.action import ActionSpec
from starter.interfaces.observation import ObservationSpec


@dataclass
class EnvManifest:
    env_id: str
    backend: str
    robot: str = "so101_follower"
    control_hz: float = 30.0
    action_kind: str = "joint_position"
    camera: str = "front"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "backend": self.backend,
            "robot": self.robot,
            "control_hz": self.control_hz,
            "action_kind": self.action_kind,
            "camera": self.camera,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvManifest":
        return cls(
            env_id=str(data.get("env_id", "unknown")),
            backend=str(data.get("backend", "unknown")),
            robot=str(data.get("robot", "so101_follower")),
            control_hz=float(data.get("control_hz", 30.0)),
            action_kind=str(data.get("action_kind", "joint_position")),
            camera=str(data.get("camera", "front")),
            extra=dict(data.get("extra", {})),
        )


@dataclass
class EpisodeStep:
    observation_image: np.ndarray
    observation_proprio: np.ndarray
    action: np.ndarray
    timestamp: float
    episode_id: int
    step_id: int
    terminated: bool
    truncated: bool
    success: bool | None = None
    env_manifest: EnvManifest | None = None

    def to_arrays(self) -> dict[str, Any]:
        return {
            "image": self.observation_image,
            "proprio": self.observation_proprio,
            "action": self.action,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": -1 if self.success is None else int(self.success),
        }


@dataclass
class CanonicalEpisode:
    episode_id: int
    steps: list[EpisodeStep]
    env_manifest: EnvManifest
    success: bool | None = None


def empty_like_specs(
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
) -> dict[str, np.ndarray]:
    h, w, c = observation_spec.image_shape
    return {
        "image": np.zeros((0, h, w, c), dtype=np.uint8),
        "proprio": np.zeros((0, observation_spec.proprio_dim), dtype=np.float32),
        "action": np.zeros((0, action_spec.dim), dtype=np.float32),
    }

