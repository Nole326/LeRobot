"""Score pure ACT or its deterministic residual on fixed PickOrange scenes."""

import argparse
import json
from pathlib import Path


def run(args):
    import numpy as np
    from ..common import publish
    from ..sac import transport
    from .act_policy import QposPolicy
    from .ee_policy import EEPolicy
    from .residual import EEResidual
    from .historical_policy import qpos_controller

    ee = args.mode == "ee"
    base = (EEPolicy if ee else QposPolicy)(args.base)
    policy = None
    if args.residual:
        if ee:
            policy = EEResidual(args.residual, base.device, args.arm_fraction)
        else:
            policy = qpos_controller(base, args.residual, args.arm_fraction)
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
                deployment_multiplier=args.arm_fraction,
                profile=args.profile,
                gripper_effort=args.gripper_effort,
                synchronized_reset=args.synchronized_reset,
                required_slots=args.required_slots,
                check_actions=args.check_actions,
            ),
        )
        peer, _ = server.accept()
        rows = []
        with (
            peer,
            (
                args.out
                / ("action-check.jsonl" if args.check_actions else "episodes.jsonl")
            ).open("x", buffering=1) as journal,
        ):
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
                    if args.check_actions:
                        if not np.isfinite(action).all():
                            raise ValueError("Nonfinite policy action")
                        journal.write(
                            json.dumps(
                                dict(seed=seed, steps=steps, action=action.tolist())
                            )
                            + "\n"
                        )
                        if reply.get("check_complete") or reply["done"]:
                            break
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
    if args.check_actions:
        publish(
            args.out / "action-check.json",
            dict(profile=args.profile, decisions=query, steps=steps),
        )
        return
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
    parser.add_argument("mode", nargs="?", choices=("qpos", "ee"))
    profiles = json.loads(
        Path(__file__).with_name("evaluation_profiles.json").read_text()
    )
    parser.add_argument("--profile", choices=profiles)
    parser.add_argument("--model-root", type=Path, default=Path("models/pickorange"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--arm-fraction", type=float, choices=(0, 0.25, 0.5, 1))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--check-actions",
        type=int,
        help="Check the first N decisions of one scene without scoring it",
    )
    parser.add_argument("--first-seed", type=int, default=731000)
    parser.add_argument("--gripper-effort", choices=("dynamic", "fixed-10Nm"))
    parser.add_argument(
        "--synchronized-reset", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--required-slots", type=int)
    args = parser.parse_args()
    if args.check_actions is not None:
        if args.check_actions < 1:
            parser.error("Action check count must be positive")
        args.episodes = 1
    if args.profile:
        if any(
            value is not None
            for value in (
                args.mode,
                args.base,
                args.residual,
                args.arm_fraction,
                args.gripper_effort,
                args.synchronized_reset,
                args.required_slots,
            )
        ):
            parser.error("Choose a named profile or supply individual policy settings")
        profile = profiles[args.profile]
        args.mode = profile["mode"]
        args.base = args.model_root / profile["base"]
        args.residual = (
            args.model_root / profile["residual"] if "residual" in profile else None
        )
        args.arm_fraction = profile.get("deployment_multiplier", 1)
        args.gripper_effort = profile.get("gripper_effort", "dynamic")
        args.synchronized_reset = profile.get("synchronized_reset", args.mode == "ee")
        args.required_slots = profile.get("required_slots")
    if args.mode is None or args.base is None:
        parser.error("Specify --profile or supply mode and --base")
    if args.arm_fraction is None:
        args.arm_fraction = 1
    if args.residual and not args.profile:
        config = json.loads((args.residual / "config.json").read_text(encoding="utf8"))
        protocol = config.get("evaluation", {})
        if args.gripper_effort is None:
            args.gripper_effort = (
                "fixed-10Nm"
                if protocol.get("fixed_gripper_effort_nm") == 10
                or protocol.get("gripper_effort") == "fixed_10_Nm"
                else "dynamic"
            )
        if args.synchronized_reset is None:
            args.synchronized_reset = protocol.get(
                "synchronized_reset",
                protocol.get("protocol") == "synchronized_reset" or args.mode == "ee",
            )
        if args.required_slots is None:
            args.required_slots = protocol.get(
                "num_envs",
                1 if protocol.get("protocol") == "independent_single_scene" else None,
            )
    if args.gripper_effort is None:
        args.gripper_effort = "dynamic"
    if args.synchronized_reset is None:
        args.synchronized_reset = args.mode == "ee"
    if args.required_slots is not None and args.required_slots < 1:
        parser.error("Scene count must be positive")
    if args.episodes < 1 or (not args.residual and args.arm_fraction != 1):
        parser.error(
            "Specify a positive episode count and a residual checkpoint for scaling"
        )
    run(args)


if __name__ == "__main__":
    main()
