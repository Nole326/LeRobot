"""Environment smoke test: python -m starter.scripts.smoke_test --config configs/lift_cube.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from starter.config import PROJECT_ROOT, add_common_args, apply_common_overrides, resolve_config
from starter.envs.container import ensure_in_leisaac_container
from starter.envs.factory import isaac_available, make_env
from starter.envs.lift_cube import IsaacUnavailableError, describe_install, shutdown_sim_app
from starter.utils import seed_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset/step smoke test for the container LiftCube env.")
    add_common_args(parser, default_config=str(PROJECT_ROOT / "configs" / "lift_cube.yaml"))
    parser.add_argument("--steps", type=int, default=8, help="Number of random steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    if argv is None:
        ensure_in_leisaac_container()
    args = parse_args(argv)
    cfg = apply_common_overrides(resolve_config(args.config), args)
    seed = int(cfg.get("seed", 0))
    seed_all(seed)
    report: dict = {
        "backend": "lift_cube",
        "isaac_available": isaac_available(),
        "seed": seed,
    }
    try:
        env = make_env(cfg)
    except IsaacUnavailableError as exc:
        report["status"] = "skipped_isaac_unavailable"
        report["reason"] = str(exc)
        report["install"] = describe_install()
        print(json.dumps(report, indent=2))
        return report

    obs, info = env.reset(seed=seed)
    obs.validate(env.observation_spec)
    privileged = env.get_privileged_state()
    privileged.validate(env.privileged_spec)
    report["reset"] = {
        "image_shape": list(obs.image.shape),
        "image_dtype": str(obs.image.dtype),
        "proprio_shape": list(obs.proprio.shape),
        "proprio_dim": env.observation_spec.proprio_dim,
        "action_dim": env.action_spec.dim,
        "action_kind": env.action_spec.kind,
        "action_units": list(env.action_spec.units),
        "frequency_hz": env.action_spec.frequency_hz,
        "info_keys": sorted(info.keys()),
    }
    report["privileged"] = {
        "fields": list(env.privileged_spec.fields),
        "cube_pos_shape": list(privileged.cube_pos.shape),
        "robot_base_pos_shape": list(privileged.robot_base_pos.shape),
        "ee_jaw_pos_shape": list(privileged.ee_jaw_pos.shape),
        "position_frame": env.privileged_spec.position_frame,
        "gripper_unit": env.privileged_spec.gripper_unit,
    }
    rewards = []
    for i in range(int(args.steps)):
        action = env.action_spec.low + (env.action_spec.high - env.action_spec.low) * np.random.rand(env.action_spec.dim)
        action = action.astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        obs.validate(env.observation_spec)
        if reward not in (0.0, 1.0):
            raise RuntimeError(f"public reward must be sparse 0/1, got {reward}")
        rewards.append(float(reward))
        if terminated or truncated:
            obs, info = env.reset(seed=seed + i + 1)
    env.close()
    report["step"] = {
        "num_steps": int(args.steps),
        "reward_mean": float(np.mean(rewards)),
        "reward_abs_max": float(np.max(np.abs(rewards))),
        "last_image_shape": list(obs.image.shape),
    }
    report["status"] = "ok"
    log_dir = Path(cfg.get("logging", {}).get("dir", "outputs/smoke"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    shutdown_sim_app()
    return report


if __name__ == "__main__":
    main()
