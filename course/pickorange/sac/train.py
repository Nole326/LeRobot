"""Train bounded residual SAC on a frozen qpos or segmented EE ACT."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ..act.act_policy import QposPolicy
from ..act.ee_policy import EEPolicy
from ..act.residual import BOUNDS, CENTER, SCALE
from ..common import publish
from . import transport
from .learner import Learner
from .replay import DeviceReplay


def configuration(args):
    config = json.loads(args.config.read_text())
    ee = config["control"] == "ee"
    if (
        config["control"],
        config["base"],
        config["base_updates"],
        config["gripper_effort"],
    ) not in (
        ("qpos", "ACT-1", 100000, "fixed-10Nm"),
        ("ee", "ACT-2", 80000, "dynamic"),
    ):
        raise ValueError("Incompatible ACT base or actuator configuration")
    if args.arm_fraction is not None:
        config["arm_fraction"] = args.arm_fraction
    if config["arm_fraction"] not in ((0.25, 0.5, 1.0) if ee else (1.0,)):
        raise ValueError(
            "qpos trains at full amplitude; EE supports quarter, half and full"
        )
    if args.limit is not None:
        config["physics_limit"] = args.limit
    if config["physics_limit"] not in ((800,) if ee else (600, 800, 1000)):
        raise ValueError("Unknown training horizon for this base")
    if args.smoke:
        config.update(
            physics_limit=32,
            batch_size=8,
            critic_warmup=2,
            actor_updates=2,
            precollect=4,
            replay_capacity=64,
        )
    for key in (
        "precollect",
        "critic_warmup",
        "actor_updates",
        "batch_size",
        "replay_capacity",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Expected positive integer {key}")
    if config["precollect"] > config["replay_capacity"]:
        raise ValueError("Replay capacity must cover precollection")
    return config


def save_checkpoint(out, label, learner, config, counts):
    path = out / (label + ".pt")
    temp = path.with_suffix(".partial")
    torch.save(
        dict(
            **learner.checkpoint(),
            config=config,
            counts=dict(counts),
            replay_saved=False,
            episode_boundary=(label == "latest"),
        ),
        temp,
    )
    temp.replace(path)


def export(out, learner, bounds, ee):
    out.mkdir(exist_ok=True)
    layers = torch.nn.ModuleDict({"encoder": learner.encoder, "actor": learner.actor})
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in layers.state_dict().items()},
        str(out / "model.safetensors"),
    )
    publish(
        out / "config.json",
        dict(
            action_mode="joint_motor_correction_after_absolute_ee_ik"
            if ee
            else "absolute_joint_motor",
            residual_limit_motor=bounds.tolist(),
            deployment_arm_fractions=[0.25, 0.5, 1.0],
            actor_updates=learner.updates,
        ),
    )


def run(args):
    config = configuration(args)
    ee = config["control"] == "ee"
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    frozen = (EEPolicy if ee else QposPolicy)(args.base)
    if frozen.control.get("optimizer_updates") != config["base_updates"]:
        raise ValueError(
            "ACT checkpoint update count does not match the selected experiment"
        )
    learner = Learner()
    bounds = BOUNDS.copy()
    bounds[:5] *= config["arm_fraction"]
    learner.bound.copy_(torch.as_tensor(bounds / SCALE, device="cuda"))
    replay = DeviceReplay(config["replay_capacity"])
    counts = dict(scenes=0, transitions=0, successes=0)
    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=True)
        if restored["config"] != config or not restored["episode_boundary"]:
            raise ValueError("Resume requires a matching end-of-scene checkpoint")
        learner.restore(restored)
        counts.update(restored["counts"])
    if learner.updates >= config["actor_updates"]:
        raise ValueError("Checkpoint already reached the requested target")
    # Rebuild a private pixel replay after a restart; collection uses the fixed base.
    rebuilding = bool(args.resume)
    smoke_before = {}
    if args.smoke:
        for name, mod in (("actor", learner.actor), ("critic", learner.q)):
            smoke_before[name] = [p.detach().clone() for p in mod.parameters()]

    def append(previous, next_id):
        obs_id, action, feedback = previous
        replay.append(obs_id, action, feedback["reward"], next_id, feedback["discount"])
        counts["transitions"] += 1
        if rebuilding or len(replay.records) < config["precollect"]:
            return
        if counts["transitions"] % 2 or learner.updates >= config["actor_updates"]:
            return
        warm = learner.q_updates < config["critic_warmup"]
        metrics = learner.update(
            replay.sample(rng, config["batch_size"]),
            critic_only=warm,
            fixed_base_target=warm,
        )
        if (
            args.smoke
            and warm
            and any(
                not torch.equal(a, b)
                for a, b in zip(smoke_before["actor"], learner.actor.parameters())
            )
        ):
            raise RuntimeError("ACT residual actor changed during critic warmup")
        metrics.update(
            actor_updates=learner.updates,
            critic_updates=learner.q_updates,
            replay_size=len(replay.records),
        )
        scalars.write(json.dumps(metrics, allow_nan=False) + "\n")
        if learner.updates and learner.updates % 2000 == 0:
            save_checkpoint(
                out, f"updates-{learner.updates:06d}", learner, config, counts
            )
            export(out / f"policy-{learner.updates:06d}", learner, bounds, ee)

    publish(out / "config.json", config)
    with (
        transport.listener(out) as server,
        (out / "scalars.jsonl").open("x", buffering=1) as scalars,
        (out / "episodes.jsonl").open("x", buffering=1) as episodes,
    ):
        publish(out / "ready.json", config)
        peer, _ = server.accept()
        with peer:
            peer.settimeout(900)
            query = 0
            while learner.updates < config["actor_updates"]:
                rebuilding = (
                    bool(args.resume) and len(replay.records) < config["precollect"]
                )
                seed = config["first_scene_seed"] + counts["scenes"]
                torch.manual_seed(seed)
                frozen.reset_scene()
                transport.send_feedback(peer, {"seed": seed})
                last_q = pending = None
                previous_action = np.zeros(6, np.float32)
                scene_steps = 0
                while True:
                    obs = transport.receive_observation(peer, query)
                    front, wrist = obs["front"][0], obs["wrist"][0]
                    joint = obs["joint_motor"].reshape(6)
                    base_motor = frozen.predict(front, wrist, joint)
                    if ee:
                        label, _ = base_motor
                        transport.send_feedback(
                            peer, dict(query=query, ee=label.tolist())
                        )
                        reply = transport.receive_feedback(peer)
                        assert reply["query"] == query
                        base_motor = np.asarray(reply["motor"], np.float32)
                    base = ((base_motor - CENTER) / SCALE).clip(-1 + 1e-5, 1 - 1e-5)
                    q = (joint - CENTER) / SCALE
                    dq = (
                        np.zeros(6, np.float32)
                        if last_q is None
                        else ((q - last_q) * 6).clip(-10, 10)
                    )
                    numeric = np.r_[q, dq, previous_action].astype(np.float32)
                    last_q = q
                    views = torch.as_tensor(
                        np.stack([front, wrist]), device="cuda"
                    ).permute(0, 3, 1, 2)
                    images = (
                        F.interpolate(views.float(), (96, 128), mode="area")
                        .round()
                        .byte()
                    )
                    obs_id = replay.observe(images, numeric, base)
                    base_only = (
                        rebuilding
                        or learner.q_updates < config["critic_warmup"]
                        or learner.updates >= config["actor_updates"]
                    )
                    with torch.no_grad():
                        if not base_only:
                            feats, target = learner.features(
                                (
                                    images[None],
                                    torch.as_tensor(numeric[None], device="cuda"),
                                    torch.as_tensor(base[None], device="cuda"),
                                )
                            )
                            action = learner.sample(feats, target)[0][0].cpu().numpy()
                        else:
                            action = base
                    if ee:
                        command = (action - base) * SCALE
                        motor = base * SCALE + CENTER + command
                    else:
                        motor = base_motor if base_only else action * SCALE + CENTER
                        command = motor
                    if pending is not None:
                        append(pending, obs_id)
                    previous_action = ((motor - CENTER) / SCALE).astype(np.float32)
                    transport.send_action(peer, np.repeat(command[None], 2, axis=0))
                    feedback = transport.receive_feedback(peer)
                    assert (feedback["query"], feedback["seed"]) == (query, seed)
                    query += 1
                    scene_steps += feedback["steps"]
                    pending = (obs_id, previous_action, feedback)
                    if feedback["done"]:
                        append(pending, obs_id)
                        break
                counts["scenes"] += 1
                counts["successes"] += int(feedback["native_success"])
                episodes.write(
                    json.dumps(
                        dict(
                            seed=seed,
                            steps=scene_steps,
                            **{
                                k: feedback[k]
                                for k in ("native_success", "retained", "scene_return")
                            },
                        )
                    )
                    + "\n"
                )
                save_checkpoint(out, "latest", learner, config, counts)
            transport.send_feedback(peer, {"stop": True})
    export(out / "policy", learner, bounds, ee)
    if args.smoke:
        delta = {
            name: max(
                float((a - b).detach().abs().max())
                for a, b in zip(smoke_before[name], mod.parameters())
            )
            for name, mod in (("actor", learner.actor), ("critic", learner.q))
        }
        if not all(np.isfinite(v) and v > 0 for v in delta.values()):
            raise RuntimeError(
                "Both actor and critic must change during the smoke test"
            )
        publish(
            out / "smoke.json",
            dict(
                max_parameter_change=delta,
                actor_updates=learner.updates,
                critic_updates=learner.q_updates,
                **counts,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--arm-fraction", type=float, choices=(0.25, 0.5, 1.0))
    parser.add_argument("--limit", type=int, choices=(600, 800, 1000))
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
