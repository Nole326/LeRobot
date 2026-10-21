"""Training reward hook. This is not the public 0/1 signal from ``env.step``."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RewardShaper(Protocol):
    """Map one transition to the reward stored in the PPO rollout.

    ``env.step`` still returns the sparse success reward. Call this after
    ``step`` and before the next ``reset``, while ``privileged`` still describes
    the post-step scene. For LiftCube that object is
    ``LiftCubePrivilegedState``. Do not copy it into the policy observation.
    """

    def __call__(
        self,
        *,
        privileged: Any,
        public_reward: float,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> float:
        ...
