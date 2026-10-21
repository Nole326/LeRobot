"""Replaceable policy/model protocol used by BC, RL, and evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .observation import Observation


@runtime_checkable
class PolicyModel(Protocol):
    def act(self, observation: Observation, deterministic: bool = False) -> np.ndarray:
        ...

    def forward(self, batch: Mapping[str, Any]) -> Any:
        ...

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "PolicyModel":
        ...
