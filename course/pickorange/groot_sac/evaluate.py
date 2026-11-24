"""Evaluate the frozen GR00T base or a deterministic residual on paired scenes."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ..common import load_protocol, predict_groot, publish, save_arrays, wait_file
from ..service import load_policy, verify_model
from .memory_rgb import receive_observation
from .sac_core import ResidualActor, state
from .train import VisualHook


@torch.inference_mode()
def run(a):
    cfg = load_protocol()
    verify_model(a.model)
    base = load_policy(a.model, a.backend, "cuda")
    base.model.eval().requires_grad_(False)
    actor = None
    if a.weights:
        actor = ResidualActor(32 * 1152 + 12, 96, (32, 1152)).cuda().eval()
        weights = (
            load_file(str(a.weights))
            if a.weights.suffix == ".safetensors"
            else torch.load(a.weights, map_location="cpu", weights_only=False)["actor"]
        )
        actor.load_state_dict(weights, strict=True)
        hook = VisualHook(base)
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    ipc = out / "ipc"
    ipc.mkdir()
    seeds = list(range(a.first_seed, a.first_seed + a.episodes))
    publish(
        out / "ready.json",
        dict(
            mode="eval", gamma=0.99, protocol=cfg, reward_variant="hybrid", seeds=seeds
        ),
    )
    scale = torch.tensor([100] * 5 + [50], device="cuda")
    center = torch.tensor([0] * 5 + [50], device="cuda")
    bounds = (torch.tensor([2] * 5 + [1], device="cuda") / scale).repeat(16)[None]
    rows = []
    query = 0
    try:
        with (out / "episodes.jsonl").open("x", buffering=1) as journal:
            for index, seed in enumerate(seeds):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                publish(
                    out / f"episode-{index:06d}.json",
                    dict(index=index, seed=seed, phase="evaluation", stop=False),
                )
                steps = 0
                while True:
                    obs = receive_observation(out, query, timeout=900)
                    chunk = predict_groot(base, obs, cfg["instruction"])
                    context_path = ipc / f"context-{query:06d}.json"
                    if actor is not None:
                        velocity = json.loads(context_path.read_text())["velocity"]
                        q = (
                            torch.as_tensor(
                                obs["joint_motor"], device="cuda", dtype=torch.float32
                            )
                            - center
                        ) / scale
                        dq = (
                            torch.as_tensor(
                                velocity, device="cuda", dtype=torch.float32
                            )
                            / 5
                        )
                        features = state(hook.value.flatten(1), q, dq).half().float()
                        target = (
                            (
                                torch.as_tensor(
                                    chunk, device="cuda", dtype=torch.float32
                                )
                                - center
                            )
                            / scale
                        ).reshape(1, 96)
                        chunk = (
                            (
                                actor(features, target, bounds, deterministic=True)[
                                    0
                                ].reshape(16, 6)
                                * scale
                                + center
                            )
                            .cpu()
                            .numpy()
                        )
                    action_path = ipc / f"action-{query:06d}.npz"
                    save_arrays(action_path, actions=chunk)
                    feedback = ipc / f"feedback-{query:06d}.json"
                    wait_file(feedback, 900, out / "sim-error.json")
                    row = json.loads(feedback.read_text())
                    if (seed, query) != (row["seed"], row["query"]):
                        raise ValueError("Out-of-order episode feedback")
                    for path in (action_path, context_path, feedback):
                        path.unlink()
                    steps += row["steps"]
                    query += 1
                    if row["terminated"] or row["truncated"]:
                        result = dict(
                            seed=seed,
                            physics_steps=steps,
                            success_60s=row["success"] and steps <= 3600,
                            success_120s=row["success"] and steps <= 7200,
                        )
                        journal.write(json.dumps(result) + "\n")
                        rows.append(result)
                        break
        publish(out / f"episode-{len(seeds):06d}.json", dict(stop=True))
        publish(
            out / "result.json",
            dict(
                episodes=len(rows),
                **{
                    k: sum(r[k] for r in rows) / len(rows)
                    for k in ("success_60s", "success_120s")
                },
            ),
        )
    finally:
        if actor is not None:
            hook.handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Published actor safetensors or a trusted training milestone",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=731000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    run(args)


if __name__ == "__main__":
    main()
