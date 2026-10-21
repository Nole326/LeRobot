"""Untrained random policy for checking the shared evaluation entry point.

This is a protocol example, not a learning baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from starter.interfaces.action import ActionSpec
from starter.interfaces.observation import Observation


class RandomPolicy:
    """Sample valid 6-D motor-unit joint targets using NumPy's global RNG."""

    def __init__(self, action_spec: ActionSpec | None = None) -> None:
        self.action_spec = action_spec or ActionSpec()

    def act(self, observation: Observation, deterministic: bool = False) -> np.ndarray:
        del observation, deterministic
        return np.random.uniform(self.action_spec.low, self.action_spec.high).astype(np.float32)

    def forward(self, batch: Mapping[str, Any]) -> Any:
        """Return random actions for a batch; not a trainable network."""
        image = np.asarray(batch["image"])
        count = 1 if image.ndim == 3 else len(image)
        return np.random.uniform(
            self.action_spec.low, self.action_spec.high, size=(count, self.action_spec.dim)
        ).astype(np.float32)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        del batch
        return {"loss": 0.0}

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.action_spec.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> RandomPolicy:
        del kwargs
        return cls(ActionSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))))
