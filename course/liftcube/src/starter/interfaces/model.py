"""Small CNN+MLP example model. Trainers depend on PolicyModel, not this class."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .action import ActionSpec, clip_action, denormalize_action
from .observation import Observation, ObservationSpec
from .vision import IdentityProcessor

_NORM_EPS = 1e-6


class _GridAveragePool(nn.AdaptiveAvgPool2d):
    """Use deterministic ordinary pooling when adaptive bins form a uniform grid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        rows, columns = self.output_size
        if height % rows == 0 and width % columns == 0:
            return F.avg_pool2d(x, kernel_size=(height // rows, width // columns))
        return super().forward(x)


class ConvEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            _GridAveragePool((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNNPolicy(nn.Module):
    """Minimal visual policy with actor (Gaussian) and critic heads."""

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        hidden_dim: int = 64,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        feature_mode: str = "joint_command",
    ) -> None:
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.hidden_dim = hidden_dim
        if feature_mode not in ("joint_command", "image_only"):
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        if feature_mode == "joint_command" and (action_spec.dim != 6 or observation_spec.proprio_dim not in (18, 21)):
            raise ValueError("joint_command expects the SO101 18/21-slot observation and six actions")
        self.feature_mode = feature_mode
        self.device_name = device
        self.processor = IdentityProcessor(input_shape=observation_spec.image_shape)
        self.encoder = ConvEncoder(out_dim=hidden_dim)
        fused = hidden_dim + (12 if feature_mode == "joint_command" else observation_spec.proprio_dim)
        self.body = nn.Sequential(nn.Linear(fused, hidden_dim), nn.ReLU())
        self.actor_mean = nn.Linear(hidden_dim, action_spec.dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_spec.dim,), -0.5))
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def _batch_from_obs_list(self, observations: list[Observation]) -> dict[str, torch.Tensor]:
        images = np.stack([obs.image for obs in observations], axis=0)
        proprio = np.stack([obs.proprio for obs in observations], axis=0)
        image_t = self.processor(images).to(self.device_name)
        proprio_t = torch.from_numpy(proprio).to(self.device_name)
        return {"image": image_t, "proprio": proprio_t}

    def features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        image = batch["image"]
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        vis = self.encoder(image)
        if self.feature_mode == "image_only":
            return vis  # Preserve inference semantics of pre-fusion checkpoints.
        proprio = batch["proprio"].float()
        # SO101 exposes q; previous applied commands are maintained locally.
        # Ignore compatibility padding even if a caller supplies nonzero values.
        robot = torch.cat([proprio[..., :6], proprio[..., 12:18]], dim=-1)
        low = torch.as_tensor(self.action_spec.low, device=robot.device).repeat(2)
        high = torch.as_tensor(self.action_spec.high, device=robot.device).repeat(2)
        robot = 2 * (robot - low) / (high - low) - 1
        return self.body(torch.cat([vis, robot], dim=-1))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        feats = self.features(self._as_tensors(batch))
        mean = self.actor_mean(feats)
        return torch.tanh(mean)

    def _as_tensors(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        image = batch["image"]
        proprio = batch["proprio"]
        if isinstance(image, np.ndarray):
            image = self.processor(image)
        if isinstance(proprio, np.ndarray):
            proprio = torch.from_numpy(proprio)
        image = image.to(self.device_name)
        proprio = proprio.float().to(self.device_name)
        if image.dim() == 3:
            image = image.unsqueeze(0)
            proprio = proprio.unsqueeze(0)
        return {"image": image, "proprio": proprio}

    def dist(self, batch: Mapping[str, Any]) -> torch.distributions.Normal:
        """Diagonal Gaussian in pre-tanh coordinates.

        The normalized action is ``tanh`` of a draw from this Gaussian, so it
        stays inside (-1, 1). ``log_prob`` includes the tanh Jacobian and is
        the density of that executed action.
        """
        feats = self.features(self._as_tensors(batch))
        mean = self.actor_mean(feats)
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def _bound_norm_action(raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw).clamp(-1.0 + _NORM_EPS, 1.0 - _NORM_EPS)

    def _log_prob_and_entropy(
        self, dist: torch.distributions.Normal, norm_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Density of the squashed action, plus base-Gaussian entropy.

        Entropy is the unsquashed Gaussian entropy used as the PPO bonus.
        The returned log-probability is the change-of-variables density of
        ``norm_action`` itself.
        """
        squashed = norm_action.clamp(-1.0 + _NORM_EPS, 1.0 - _NORM_EPS)
        raw = torch.atanh(squashed)
        # log|d tanh/du| = 2 (log 2 - u - softplus(-2u))
        log_det = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_prob = dist.log_prob(raw).sum(dim=-1) - log_det.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def _env_action(self, norm_action: torch.Tensor) -> np.ndarray:
        norm_np = norm_action.detach().cpu().numpy()[0]
        return clip_action(denormalize_action(norm_np, self.action_spec), self.action_spec)

    def act(self, observation: Observation, deterministic: bool = False) -> np.ndarray:
        action, _ = self.act_with_info(observation, deterministic=deterministic)
        return action

    def act_with_info(self, observation: Observation, deterministic: bool = False) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            batch = self._batch_from_obs_list([observation])
            dist = self.dist(batch)
            raw = dist.mean if deterministic else dist.sample()
            norm_action = self._bound_norm_action(raw)
            log_prob, _ = self._log_prob_and_entropy(dist, norm_action)
            value = self.critic(self.features(batch)).squeeze(-1)
            action = self._env_action(norm_action)
        return action, {"log_prob": log_prob, "value": value, "norm_action": norm_action}

    def evaluate_actions(
        self, batch: Mapping[str, Any], norm_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.dist(batch)
        log_prob, entropy = self._log_prob_and_entropy(dist, norm_actions)
        value = self.critic(self.features(self._as_tensors(batch))).squeeze(-1)
        return log_prob, entropy, value

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Default BC-style MSE update on normalized actions."""
        self.train()
        tensors = self._as_tensors(batch)
        pred = self.forward(tensors)
        target = batch["action"]
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        target = target.float().to(self.device_name)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        low = torch.as_tensor(self.action_spec.low, device=self.device_name)
        high = torch.as_tensor(self.action_spec.high, device=self.device_name)
        target_norm = 2.0 * (target - low) / torch.clamp(high - low, min=1e-6) - 1.0
        loss = torch.mean((pred - target_norm) ** 2)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.detach().cpu())}

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path_obj)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_class": self.__class__.__name__,
            "state_dict": self.state_dict(),
            "observation_spec": self.observation_spec.to_dict(),
            "action_spec": self.action_spec.to_dict(),
            "hidden_dim": self.hidden_dim,
            "device": self.device_name,
            "feature_mode": self.feature_mode,
        }

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "SmallCNNPolicy":
        map_location = kwargs.get("device", "cpu")
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        return cls.from_payload(payload, **kwargs)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], **kwargs: Any) -> "SmallCNNPolicy":
        device = kwargs.get("device", "cpu")
        model = cls(
            observation_spec=ObservationSpec.from_dict(payload["observation_spec"]),
            action_spec=ActionSpec.from_dict(payload["action_spec"]),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            device=device,
            learning_rate=float(kwargs.get("learning_rate", 1e-3)),
            feature_mode=payload.get("feature_mode", "image_only"),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.device_name = device
        return model
