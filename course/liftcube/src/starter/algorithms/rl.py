"""RL trainer protocol and rollout buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from starter.interfaces.observation import Observation
from starter.interfaces.policy import PolicyModel


@dataclass
class RolloutBuffer:
    images: list[np.ndarray] = field(default_factory=list)
    proprio: list[np.ndarray] = field(default_factory=list)
    actions_env: list[np.ndarray] = field(default_factory=list)
    actions_norm: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)  # terminated or truncated
    truncated: list[float] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)

    def add(
        self,
        obs: Observation,
        action_env: np.ndarray,
        action_norm: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        truncated: bool,
        next_value: float,
    ) -> None:
        self.images.append(np.asarray(obs.image))
        self.proprio.append(np.asarray(obs.proprio, dtype=np.float32))
        self.actions_env.append(np.asarray(action_env, dtype=np.float32))
        self.actions_norm.append(np.asarray(action_norm, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.truncated.append(float(truncated))
        self.next_values.append(float(next_value))

    def __len__(self) -> int:
        return len(self.rewards)

    def as_tensors(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        images = np.stack(self.images, axis=0)
        # HWC uint8 -> caller converts via policy processor
        return {
            "image": images,
            "proprio": torch.from_numpy(np.stack(self.proprio, axis=0)).to(device),
            "action": torch.from_numpy(np.stack(self.actions_env, axis=0)).to(device),
            "norm_action": torch.from_numpy(np.stack(self.actions_norm, axis=0)).to(device),
            "log_prob": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "value": torch.tensor(self.values, dtype=torch.float32, device=device),
            "reward": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "done": torch.tensor(self.dones, dtype=torch.float32, device=device),
            "truncated": torch.tensor(self.truncated, dtype=torch.float32, device=device),
            "next_value": torch.tensor(self.next_values, dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self.__init__()


@runtime_checkable
class RLAlgorithm(Protocol):
    policy: PolicyModel

    def collect(self, env: Any, num_steps: int) -> RolloutBuffer:
        ...

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
