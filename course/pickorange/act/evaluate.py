"""Score pure ACT or its deterministic residual on fixed PickOrange scenes."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..common import publish
from ..sac import transport


def run(args):
    from .act_policy import QposPolicy
    from .ee_policy import EEPolicy
    from .residual import EEResidual, make_policy

    ee = args.mode == "ee"
    base = (EEPolicy if ee else QposPolicy)(args.base)
    policy = None
    if args.residual:
        if ee:
            policy = EEResidual(args.residual, base.device, args.arm_fraction)
        else:
            policy = make_policy(
                base.model,
                base.stats,
                args.residual / "model.safetensors",
                base.device,
                args.arm_fraction,
            )
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = range(args.first_seed, args.first_seed + args.episodes)
    with transport.listener(args.out) as server:
        publish(
            args.out / "ready.json",
            dict(
                evaluation=True,
                control=args.mode,
                physics_limit=7200,
                first_scene_seed=seeds.start,
                arm_fraction=1.0,
            ),
        )
        peer, _ = server.accept()
        rows = []
        with peer, (args.out / "episodes.jsonl").open("x", buffering=1) as journal:
            peer.settimeout(900)
            query = 0
            for seed in seeds:
                base.reset_scene()
                if policy:
                    (policy.reset_scene if ee else policy.reset)()
                transport.send_feedback(peer, {"seed": seed})
                steps = 0
                while True:
                    obs = transport.receive_observation(peer, query)
                    inputs = (
                        obs["front"][0],
                        obs["wrist"][0],
                        obs["joint_motor"].reshape(6),
                    )
                    if ee:
                        label, _ = base.predict(*inputs)
                        transport.send_feedback(
                            peer, dict(query=query, ee=label.tolist())
                        )
                        reply = transport.receive_feedback(peer)
                        if reply["query"] != query:
                            raise ValueError("Out-of-order IK response")
                        motor = np.asarray(reply["motor"], np.float32)
                        action = (
                            policy.correction(*inputs, motor)
                            if policy
                            else np.zeros(6, np.float32)
                        )
                    else:
                        action = (
                            policy.act(*inputs) if policy else base.predict(*inputs)
                        )
                    transport.send_action(peer, np.repeat(action[None], 2, axis=0))
                    reply = transport.receive_feedback(peer)
                    if (reply["query"], reply["seed"]) != (query, seed):
                        raise ValueError("Out-of-order simulation response")
                    query += 1
                    steps += reply["steps"]
                    if reply["done"]:
                        success = reply["native_success"]
                        row = dict(
                            seed=seed,
                            physics_steps=steps,
                            success_60s=success and steps <= 3600,
                            success_120s=success and steps <= 7200,
                        )
                        journal.write(json.dumps(row) + "\n")
                        rows.append(row)
                        break
            transport.send_feedback(peer, {"stop": True})
    publish(
        args.out / "result.json",
        dict(
            episodes=len(rows),
            **{
                k: sum(row[k] for row in rows) / len(rows)
                for k in ("success_60s", "success_120s")
            },
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qpos", "ee"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--residual", type=Path)
    parser.add_argument(
        "--arm-fraction", type=float, choices=(0, 0.25, 0.5, 1), default=1
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=731000)
    args = parser.parse_args()
    if args.episodes < 1 or (not args.residual and args.arm_fraction != 1):
        parser.error(
            "Specify a positive episode count and a residual checkpoint for scaling"
        )
    run(args)


if __name__ == "__main__":
    main()
