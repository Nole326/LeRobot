"""Shared helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from starter.interfaces.action import ActionSpec
from starter.interfaces.model import SmallCNNPolicy
from starter.interfaces.observation import ObservationSpec


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(
    cfg: dict[str, Any],
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
    device: str,
) -> SmallCNNPolicy:
    model_cfg = cfg.get("model", {})
    lr = float(cfg.get("train", {}).get("learning_rate", cfg.get("ppo", {}).get("learning_rate", 1e-3)))
    return SmallCNNPolicy(
        observation_spec=observation_spec,
        action_spec=action_spec,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        device=device,
        learning_rate=lr,
    )


def clone_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]
