"""Gym-like environment protocol used by training and evaluation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from starter.interfaces.action import ActionSpec
from starter.interfaces.observation import Observation, ObservationSpec


@runtime_checkable
class EnvAdapter(Protocol):
    action_spec: ActionSpec
    observation_spec: ObservationSpec
    name: str

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        ...

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        ...
