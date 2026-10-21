"""Field, shape, unit, and boundary checks for canonical datasets."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from starter.interfaces.action import ActionSpec
from starter.interfaces.observation import ObservationSpec

from .schema import EpisodeStep


class DatasetValidationError(ValueError):
    pass


def validate_step(
    step: EpisodeStep,
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
    *,
    check_action_range: bool = True,
) -> None:
    img = np.asarray(step.observation_image)
    if img.shape != observation_spec.image_shape:
        raise DatasetValidationError(f"image shape {img.shape} != {observation_spec.image_shape}")
    if img.dtype != np.uint8:
        raise DatasetValidationError(f"image dtype {img.dtype} != uint8")
    proprio = np.asarray(step.observation_proprio, dtype=np.float32).reshape(-1)
    if proprio.shape != (observation_spec.proprio_dim,):
        raise DatasetValidationError(f"proprio dim {proprio.shape} != {(observation_spec.proprio_dim,)}")
    if not np.all(np.isfinite(proprio)):
        raise DatasetValidationError("proprio not finite")
    action = np.asarray(step.action, dtype=np.float32).reshape(-1)
    if action.shape != (action_spec.dim,):
        raise DatasetValidationError(f"action dim {action.shape} != {(action_spec.dim,)}")
    if not np.all(np.isfinite(action)):
        raise DatasetValidationError("action not finite")
    if check_action_range:
        if np.any(action < action_spec.low - 1e-3) or np.any(action > action_spec.high + 1e-3):
            raise DatasetValidationError(f"action {action} outside [{action_spec.low}, {action_spec.high}]")
    if step.success is not None and step.success not in (True, False, 0, 1):
        raise DatasetValidationError(f"invalid success flag {step.success}")


def validate_dataset(
    steps: Iterable[EpisodeStep],
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
) -> dict[str, int]:
    count = 0
    episode_ids: set[int] = set()
    for step in steps:
        validate_step(step, observation_spec, action_spec)
        episode_ids.add(int(step.episode_id))
        count += 1
    if count == 0:
        raise DatasetValidationError("dataset is empty")
    return {"num_steps": count, "num_episodes": len(episode_ids)}


def validate_arrays(
    payload: dict[str, np.ndarray],
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
) -> dict[str, int]:
    required = ("image", "proprio", "action", "timestamp", "episode_id", "step_id", "terminated", "truncated")
    missing = [k for k in required if k not in payload]
    if missing:
        raise DatasetValidationError(f"missing fields: {missing}")
    n = int(payload["action"].shape[0])
    if payload["image"].shape[0] != n:
        raise DatasetValidationError("image/action length mismatch")
    if tuple(payload["image"].shape[1:]) != observation_spec.image_shape:
        raise DatasetValidationError(
            f"image shape {payload['image'].shape[1:]} != {observation_spec.image_shape}"
        )
    if payload["proprio"].shape != (n, observation_spec.proprio_dim):
        raise DatasetValidationError(
            f"proprio shape {payload['proprio'].shape} != {(n, observation_spec.proprio_dim)}"
        )
    if payload["action"].shape != (n, action_spec.dim):
        raise DatasetValidationError(f"action shape {payload['action'].shape} != {(n, action_spec.dim)}")
    if not np.all(np.isfinite(payload["action"])):
        raise DatasetValidationError("action has non-finite values")
    if np.any(payload["action"] < action_spec.low - 1e-3) or np.any(payload["action"] > action_spec.high + 1e-3):
        raise DatasetValidationError("action out of spec range")
    return {"num_steps": n, "num_episodes": int(len(np.unique(payload["episode_id"])))}
