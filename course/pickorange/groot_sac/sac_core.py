"""Visual-feature SAC primitives. No simulator, pretrained loader, or disk pixels.

The caller supplies frozen spatial visual features and full executed base chunks.
This module is not a training launcher; model/encoder integration has separate gates.
"""

from copy import deepcopy
import math
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs):
    layers = []
    for _ in range(3):
        layers += [nn.Linear(inputs, 256), nn.LayerNorm(256), nn.ReLU()]
        inputs = 256
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def state(visual, joints, velocity):
    """Deployable state: no object truth or external rollout-limit clock."""
    assert visual.ndim == joints.ndim == velocity.ndim == 2
    assert joints.shape[-1] == velocity.shape[-1] == 6
    assert visual.shape[-1] > 6
    x = torch.cat((visual, joints, velocity), -1).float()
    assert torch.isfinite(x).all()
    return x


class VisualProjection(nn.Module):
    def __init__(self, features, visual_shape=None):
        super().__init__()
        self.visual_shape = visual_shape
        if visual_shape is None:
            self.output_dim = features
        else:
            tokens, width = visual_shape
            assert features == tokens * width + 12
            self.projection = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 64), nn.LayerNorm(64), nn.ReLU()
            )
            self.output_dim = tokens * 64 + 12

    def forward(self, features):
        if self.visual_shape is None:
            return features
        tokens, width = self.visual_shape
        visual = features[:, : tokens * width].reshape(-1, tokens, width)
        return torch.cat(
            (self.projection(visual).flatten(1), features[:, tokens * width :]), -1
        )


class QHead(nn.Module):
    def __init__(self, features, action_dim, visual_shape, privileged_dim=0):
        super().__init__()
        self.features = VisualProjection(features, visual_shape)
        self.privileged_dim = privileged_dim
        self.network = mlp(self.features.output_dim + action_dim + privileged_dim, 1)

    def forward(self, features, actual_action, privileged=None):
        inputs = [self.features(features), actual_action.flatten(1)]
        if self.privileged_dim:
            assert privileged is not None and privileged.shape == (
                features.shape[0],
                self.privileged_dim,
            )
            assert torch.isfinite(privileged).all()
            inputs.append(privileged.detach())
        return self.network(torch.cat(inputs, -1))


class TwinQ(nn.Module):
    def __init__(self, features, action_dim, visual_shape=None, privileged_dim=0):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                QHead(features, action_dim, visual_shape, privileged_dim)
                for _ in range(2)
            ]
        )

    def forward(self, features, actual_action, privileged=None):
        return torch.cat(
            [q(features, actual_action, privileged) for q in self.heads], -1
        )


def bounded_sample(mean, log_std, *, center, scale, deterministic=False, noise=None):
    """Density in actual normalized-action coordinates, including affine Jacobian."""
    assert (scale > 0).all() and torch.isfinite(mean).all()
    log_std = log_std.clamp(-5.0, 1.0)
    eps = (
        torch.zeros_like(mean)
        if deterministic
        else (torch.randn_like(mean) if noise is None else noise)
    )
    z = mean + log_std.exp() * eps
    action = center + scale * z.tanh()
    log_jacobian = 2 * (math.log(2) - z - F.softplus(-2 * z))
    log_prob = (
        -0.5 * eps.square()
        - 0.5 * math.log(2 * math.pi)
        - log_std
        - log_jacobian
        - scale.log()
    ).sum(-1)
    # Match target entropy to the affine support, rather than forcing a tiny
    # residual distribution to attain the entropy of the entire motor range.
    target_entropy = -mean.shape[-1] + scale.log().sum(-1)
    return action, log_prob, target_entropy


class ResidualActor(nn.Module):
    def __init__(self, features, action_dim, visual_shape=None):
        super().__init__()
        self.action_dim = action_dim
        self.features = VisualProjection(features, visual_shape)
        self.mean = mlp(self.features.output_dim + action_dim, action_dim)
        self.log_std = mlp(self.features.output_dim + action_dim, action_dim)
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        nn.init.zeros_(self.log_std[-1].weight)
        nn.init.constant_(self.log_std[-1].bias, -3.0)

    def forward(self, features, base, bound, *, deterministic=False, noise=None):
        base = base.flatten(1).clamp(-1 + 1e-5, 1 - 1e-5)
        bound = torch.broadcast_to(bound, base.shape)
        assert (bound > 0).all()
        low, high = (base - bound).clamp(min=-1), (base + bound).clamp(max=1)
        center, scale = (low + high) / 2, (high - low) / 2
        offset = ((base - center) / scale).clamp(-1 + 1e-6, 1 - 1e-6).atanh()
        x = torch.cat((self.features(features), base), -1)
        return bounded_sample(
            offset + self.mean(x),
            self.log_std(x),
            center=center,
            scale=scale,
            deterministic=deterministic,
            noise=noise,
        )


class SAC:
    def __init__(
        self, features, action_dim, device="cpu", visual_shape=None, privileged_dim=0
    ):
        self.actor = ResidualActor(features, action_dim, visual_shape).to(device)
        self.q = TwinQ(features, action_dim, visual_shape, privileged_dim).to(device)
        self.target = deepcopy(self.q).requires_grad_(False)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(0.01 / action_dim), device=device)
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.actor_updates = self.critic_updates = 0

    def update(
        self, batch, bound, *, critic_only=False, privileged=None, next_privileged=None
    ):
        s, a, r, ns, discount, base, next_base = batch
        assert r.ndim == discount.ndim == 1 and s.shape[0] == 32
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp, _ = self.actor(ns, next_base, bound)
            y = r + discount * (
                self.target(ns, na, next_privileged).min(-1).values - alpha * nlp
            )
        self.q_opt.zero_grad(set_to_none=True)
        current = self.q(s, a, privileged)
        q_loss = F.mse_loss(current, y[:, None].expand_as(current))
        q_loss.backward()
        q_norm = nn.utils.clip_grad_norm_(
            self.q.parameters(), 1.0, error_if_nonfinite=True
        )
        self.q_opt.step()
        metrics = dict(
            q_loss=float(q_loss.detach()),
            q_mean=float(current.detach().mean()),
            td_target_mean=float(y.mean()),
            td_abs=float((current.detach() - y[:, None]).abs().mean()),
            q_grad_norm=float(q_norm),
            alpha=float(alpha),
        )
        self.critic_updates += 1
        if not critic_only:
            self.q.requires_grad_(False)
            try:
                self.actor_opt.zero_grad(set_to_none=True)
                pa, lp, target_entropy = self.actor(s, base, bound)
                actor_loss = (
                    alpha * lp - self.q(s, pa, privileged).min(-1).values
                ).mean()
                actor_loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 1.0, error_if_nonfinite=True
                )
                self.actor_opt.step()
            finally:
                self.q.requires_grad_(True)
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = -(self.log_alpha * (lp + target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_opt.step()
            assert torch.isfinite(self.log_alpha)
            metrics.update(
                actor_loss=float(actor_loss.detach()),
                actor_grad_norm=float(actor_norm),
                entropy=float(-lp.detach().mean()),
                target_entropy=float(target_entropy.mean()),
                alpha_loss=float(alpha_loss.detach()),
            )
            self.actor_updates += 1
        with torch.no_grad():
            for t, q in zip(self.target.parameters(), self.q.parameters()):
                t.lerp_(q, 0.005)
        return metrics

    def save(self, path, *, identity, replay, sampler, schedule):
        """Store numeric features only; caller supplies an audited episode boundary."""
        assert schedule["episode_complete"] is True
        for record in replay:
            assert set(record) in (
                {
                    "state",
                    "action",
                    "reward",
                    "next_state",
                    "discount",
                    "base",
                    "next_base",
                },
                {
                    "state",
                    "action",
                    "reward",
                    "next_state",
                    "discount",
                    "base",
                    "next_base",
                    "privileged",
                    "next_privileged",
                },
            )
            assert all(
                isinstance(t, torch.Tensor)
                and t.device.type == "cpu"
                and t.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and t.ndim <= 2
                and torch.isfinite(t).all()
                for t in record.values()
            )
        import numpy as np

        path = Path(path)
        assert not path.exists()
        bundle = dict(
            identity=identity,
            replay=replay,
            sampler=sampler.getstate(),
            schedule=schedule,
            actor_updates=self.actor_updates,
            critic_updates=self.critic_updates,
            log_alpha=self.log_alpha.detach().cpu(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else [],
        )
        for key in ("actor", "q", "target", "actor_opt", "q_opt", "alpha_opt"):
            bundle[key] = getattr(self, key).state_dict()
        temporary = path.with_suffix(".partial")
        with temporary.open("xb") as f:
            torch.save(bundle, f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)

    def restore(self, path, *, identity, sampler):
        import numpy as np

        bundle = torch.load(path, map_location="cpu", weights_only=False)
        assert (
            bundle["identity"] == identity
            and bundle["schedule"]["episode_complete"] is True
        )
        for key in ("actor", "q", "target", "actor_opt", "q_opt", "alpha_opt"):
            getattr(self, key).load_state_dict(bundle[key])
        with torch.no_grad():
            self.log_alpha.copy_(bundle["log_alpha"])
        self.actor_updates, self.critic_updates = (
            bundle["actor_updates"],
            bundle["critic_updates"],
        )
        sampler.setstate(bundle["sampler"])
        random.setstate(bundle["python_rng"])
        np.random.set_state(bundle["numpy_rng"])
        torch.set_rng_state(bundle["torch_rng"])
        if bundle["cuda_rng"]:
            assert len(bundle["cuda_rng"]) == torch.cuda.device_count()
            torch.cuda.set_rng_state_all(bundle["cuda_rng"])
        return bundle["replay"], bundle["schedule"]


def discount(physics_steps, terminated, truncated):
    assert 1 <= physics_steps <= 32
    return 0.0 if terminated else 0.99 ** (physics_steps / 16)
