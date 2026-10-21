"""Environment factory. LiftCube runs in the course image or a local Isaac install."""

from __future__ import annotations

from typing import Any

from starter.envs.container import isaac_runtime_ready

from .lift_cube import IsaacUnavailableError, LiftCubeEnvAdapter, describe_install


def isaac_available() -> bool:
    return isaac_runtime_ready()


def make_env(cfg: dict[str, Any]):
    env_cfg = cfg.get("env", cfg)
    if not isaac_runtime_ready():
        raise IsaacUnavailableError(describe_install())
    image_height = int(env_cfg.get("image_height", 64))
    image_width = int(env_cfg.get("image_width", 64))
    max_steps = int(env_cfg.get("max_episode_steps", 640))
    include_ee = bool(env_cfg.get("include_ee_in_proprio", True))
    camera = str(env_cfg.get("camera", "front"))
    hz = float(env_cfg.get("control_hz", 30.0))
    sim_device = str(env_cfg.get("sim_device", "cuda:0"))
    return LiftCubeEnvAdapter(
        camera=camera,
        image_height=image_height,
        image_width=image_width,
        include_ee_in_proprio=include_ee,
        control_hz=hz,
        n_envs=int(env_cfg.get("n_envs", 1)),
        device=sim_device,
        max_episode_steps=max_steps,
    )
