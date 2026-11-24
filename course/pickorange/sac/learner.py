"""Pixel SAC with twin critics, bounded joint targets and independent view augmentation."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..act.residual import ResidualBlock, mlp


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        def tower():
            layers = []
            channels = 3
            for _ in range(4):
                layers += [nn.Conv2d(channels, 32, 3, 2, 1), nn.ReLU()]
                channels = 32
            return nn.Sequential(*layers)

        self.views = nn.ModuleList([tower(), tower()])
        # Preserve the initialization order used by the trained policies.
        self.readout = nn.Sequential(
            nn.Flatten(), nn.Linear(5120, 128), nn.LayerNorm(128), nn.Tanh()
        )

        self.readout[1] = nn.Linear(32 * 6 * 8 * 2, 128)
        for tower in self.views:
            tower.append(ResidualBlock())

    def forward(self, x, augment=False):
        x = x.float() / 255
        if augment:
            # Translate each camera consistently across its RGB channels.
            shape = x.shape
            x = x.flatten(0, 1)
            x = F.pad(x, (4, 4, 4, 4), mode="replicate")
            shift = torch.randint(0, 9, (len(x), 2), device=x.device)
            _, channels, height, width = x.shape
            y = (
                torch.arange(shape[-2], device=x.device)[None, :] + shift[:, :1]
            ) * width
            columns = torch.arange(shape[-1], device=x.device)[None, :] + shift[:, 1:]
            index = (y[:, :, None] + columns[:, None, :]).flatten(1)
            x = (
                x.flatten(2)
                .gather(2, index[:, None, :].expand(-1, channels, -1))
                .reshape(shape)
            )
        inputs = [x[:, i] - 0.5 for i in range(2)]
        return self.readout(
            torch.cat([m(view).flatten(1) for m, view in zip(self.views, inputs)], 1)
        )


class QHead(nn.Module):
    def __init__(
        self,
        features,
        action_dim,
    ):
        super().__init__()
        self.network = mlp(features + action_dim, 1)

    def forward(self, features, actual_action):
        inputs = [features, actual_action.flatten(1)]
        return self.network(torch.cat(inputs, -1))


class TwinQ(nn.Module):
    def __init__(self, features, action_dim):
        super().__init__()
        self.heads = nn.ModuleList([QHead(features, action_dim) for _ in range(2)])

    def forward(self, features, actual_action):
        return torch.cat([q(features, actual_action) for q in self.heads], -1)


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
    def __init__(self, features, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.mean = mlp(features + action_dim, action_dim)
        self.log_std = mlp(features + action_dim, action_dim)
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
        x = torch.cat((features, base), -1)
        return bounded_sample(
            offset + self.mean(x),
            self.log_std(x),
            center=center,
            scale=scale,
            deterministic=deterministic,
            noise=noise,
        )


class Learner:
    def __init__(self):
        self.encoder = SpatialEncoder().cuda()
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.actor = ResidualActor(146, 6).cuda()
        self.q = TwinQ(152, 6).cuda()
        self.target = deepcopy(self.q).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.q_opt = torch.optim.Adam(
            [*self.q.parameters(), *self.encoder.parameters()], lr=3e-4
        )
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.001), device="cuda"))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.bound = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.006, 0.04], device="cuda")[
            None
        ]
        self.updates = 0
        self.q_updates = 0

    def features(self, obs, target=False, augment=False):
        image, numeric, base = obs
        return torch.cat(
            (
                (self.target_encoder if target else self.encoder)(image, augment),
                numeric.float(),
            ),
            1,
        ), base.float()

    def sample(self, f, b, deterministic=False):
        a, lp, _ = self.actor(f, b, self.bound, deterministic=deterministic)
        scale = ((b + self.bound).clamp(max=1) - (b - self.bound).clamp(min=-1)) / 2
        # Entropy is defined in unit bounded-residual coordinates.
        return a, lp + scale.log().sum(-1)

    def update(self, batch, critic_only=False, fixed_base_target=False):
        obs, a, r, next_obs, d = batch
        f, b = self.features(obs, augment=True)
        with torch.no_grad():
            nf, nb = self.features(next_obs, target=True, augment=True)
            if fixed_base_target:
                target_q = self.target(torch.cat((nf, nb), 1), nb).min(1).values
            else:
                na, nlp = self.sample(nf, nb)
                target_q = (
                    self.target(torch.cat((nf, nb), 1), na).min(1).values
                    - self.log_alpha.exp() * nlp
                )
            y = r + d * target_q
        self.q_opt.zero_grad(set_to_none=True)
        q = self.q(torch.cat((f, b), 1), a)
        loss = F.mse_loss(q, y[:, None].expand_as(q))
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            [*self.encoder.parameters(), *self.q.parameters()],
            1,
            error_if_nonfinite=True,
        )
        self.q_opt.step()
        self.q_updates += 1
        metrics = dict(
            q_loss=float(loss.detach()),
            q_mean=float(q.detach().mean()),
            target_mean=float(y.mean()),
            q_grad=float(norm),
            alpha=float(self.log_alpha.exp().detach()),
        )
        if not critic_only:
            with torch.no_grad():
                f, b = self.features(obs)
            self.q.requires_grad_(False)
            try:
                self.actor_opt.zero_grad(set_to_none=True)
                pa, lp = self.sample(f, b)
                loss = (
                    self.log_alpha.exp().detach() * lp
                    - self.q(torch.cat((f, b), 1), pa).min(1).values
                ).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 1, error_if_nonfinite=True
                )
                self.actor_opt.step()
            finally:
                self.q.requires_grad_(True)
            self.alpha_opt.zero_grad(set_to_none=True)
            aloss = -(self.log_alpha * (lp.detach() - 6)).mean()
            aloss.backward()
            self.alpha_opt.step()
            self.updates += 1
            metrics.update(
                actor_loss=float(loss.detach()), entropy=float(-lp.detach().mean())
            )
        with torch.no_grad():
            for target, source in (
                (self.target, self.q),
                (self.target_encoder, self.encoder),
            ):
                for t, s in zip(target.parameters(), source.parameters()):
                    t.lerp_(s, 0.01)
        return metrics

    def checkpoint(self):
        return {
            k: getattr(self, k).state_dict()
            for k in (
                "encoder",
                "target_encoder",
                "actor",
                "q",
                "target",
                "actor_opt",
                "q_opt",
                "alpha_opt",
            )
        } | dict(
            log_alpha=self.log_alpha.detach().cpu(),
            actor_updates=self.updates,
            critic_updates=self.q_updates,
        )

    def restore(self, state):
        for key in (
            "encoder",
            "target_encoder",
            "actor",
            "q",
            "target",
            "actor_opt",
            "q_opt",
            "alpha_opt",
        ):
            getattr(self, key).load_state_dict(state[key])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.updates = state["actor_updates"]
        self.q_updates = state["critic_updates"]
