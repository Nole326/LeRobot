"""Teaching PPO with GAE. Updates actor/critic parameters on real losses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from starter.interfaces.model import SmallCNNPolicy
from starter.interfaces.reward import RewardShaper
from starter.rewards.lift_cube import shape_reward

from ..rl import RolloutBuffer


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE-Lambda. `dones` is 1 at terminated or truncated boundaries."""
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device)
    next_value = last_value
    for t in reversed(range(t_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


class PPOAlgorithm:
    def __init__(
        self,
        policy: SmallCNNPolicy,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 2,
        minibatch_size: int = 32,
        reward_shaper: RewardShaper | None = None,
    ) -> None:
        self.policy = policy
        self.reward_shaper = shape_reward if reward_shaper is None else reward_shaper
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.policy.opt = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def collect(self, env: Any, num_steps: int, seed: int | None = None) -> tuple[RolloutBuffer, dict[str, float]]:
        buf = RolloutBuffer()
        obs, info = env.reset(seed=seed)
        ep_return = 0.0
        ep_returns: list[float] = []
        successes = 0
        episodes = 0
        for _ in range(num_steps):
            action, extra = self.policy.act_with_info(obs, deterministic=False)
            next_obs, public_reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            reward = self._shaped_reward(
                env,
                action=action,
                public_reward=float(public_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=step_info if isinstance(step_info, dict) else {},
            )
            buf.add(
                obs=obs,
                action_env=action,
                action_norm=extra["norm_action"].detach().cpu().numpy()[0],
                log_prob=float(extra["log_prob"].detach().cpu()),
                value=float(extra["value"].detach().cpu()),
                reward=reward,
                done=done,
                truncated=bool(truncated),
            )
            ep_return += reward
            obs = next_obs
            if done:
                episodes += 1
                ep_returns.append(ep_return)
                successes += int(bool(step_info.get("success", terminated and not truncated)))
                ep_return = 0.0
                obs, info = env.reset()
        last_value = 0.0
        if len(buf) > 0:
            with torch.no_grad():
                _, extra = self.policy.act_with_info(obs, deterministic=True)
                last_value = float(extra["value"].detach().cpu())
        metrics = {
            "rollout_steps": float(len(buf)),
            "episode_count": float(episodes),
            "mean_return": float(np.mean(ep_returns) if ep_returns else ep_return),
            "success_count": float(successes),
            "last_value": last_value,
        }
        return buf, metrics

    def _shaped_reward(
        self,
        env: Any,
        *,
        action: np.ndarray,
        public_reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> float:
        getter = getattr(env, "get_privileged_state", None)
        if getter is None:
            raise RuntimeError("reward shaping requires env.get_privileged_state()")
        privileged = getter()
        reward = self.reward_shaper(
            privileged=privileged,
            public_reward=public_reward,
            action=np.asarray(action, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError(f"reward shaper returned a non-finite value: {reward}")
        return reward

    def update(self, batch: Mapping[str, Any] | RolloutBuffer, last_value: float = 0.0) -> dict[str, float]:
        if isinstance(batch, RolloutBuffer):
            device = self.policy.device_name
            tensors = batch.as_tensors(device=device)
            images = self.policy.processor(tensors["image"]).to(device)
            proprio = tensors["proprio"]
            norm_actions = tensors["norm_action"]
            old_logp = tensors["log_prob"]
            values = tensors["value"]
            rewards = tensors["reward"]
            dones = tensors["done"]
        else:
            device = self.policy.device_name
            images = batch["image"]
            if isinstance(images, np.ndarray):
                images = self.policy.processor(images)
            images = images.to(device)
            proprio = batch["proprio"].to(device)
            norm_actions = batch["norm_action"].to(device)
            old_logp = batch["log_prob"].to(device)
            values = batch["value"].to(device)
            rewards = batch["reward"].to(device)
            dones = batch["done"].to(device)

        last_v = torch.tensor(last_value, dtype=torch.float32, device=device)
        advantages, returns = compute_gae(rewards, values, dones, last_v, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = rewards.shape[0]
        idx = np.arange(n)
        last_stats: dict[str, float] = {}
        self.policy.train()
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, device=device, dtype=torch.long)
                mb_batch = {"image": images[mb_t], "proprio": proprio[mb_t]}
                logp, entropy, value = self.policy.evaluate_actions(mb_batch, norm_actions[mb_t])
                ratio = torch.exp(logp - old_logp[mb_t])
                adv = advantages[mb_t]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.mean((value - returns[mb_t]) ** 2)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                self.policy.opt.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.opt.step()
                last_stats = {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.mean().detach().cpu()),
                }
        return last_stats

    def save(self, path: str) -> None:
        self.policy.save(path)

    def load(self, path: str) -> None:
        loaded = SmallCNNPolicy.load(path, device=self.policy.device_name)
        self.policy.load_state_dict(loaded.state_dict())
