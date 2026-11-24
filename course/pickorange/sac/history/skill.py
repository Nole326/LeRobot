"""Train a bounded single-orange residual while preserving the physical scene at skill boundaries."""

import argparse
import json
from pathlib import Path
import traceback
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ...common import load_protocol, publish
from .. import transport as duplex
from .learner import Learner
from ...act.act_policy import QposPolicy
from .vision import PlacementDetector
from .release import ReleasePlacementTrigger
from .home import ReturnHome, at_home, score_one_orange

PROFILE = json.loads(Path(__file__).with_name("placement.json").read_text())
SCALE = np.array([100.0] * 5 + [50.0], np.float32)
CENTER = np.array([0.0] * 5 + [50.0], np.float32)
BOUNDS = np.asarray(PROFILE["residual_motor"], np.float32)
HORIZON = PROFILE["scene_horizon_seconds"]
PICK_LIMIT = PROFILE["pick_limit"]
RETURN_LIMIT = PROFILE["return_limit"]
PRECOLLECT = PROFILE["precollect"]


def run(args):
    torch.manual_seed(PROFILE["seed"])
    rng = np.random.default_rng(PROFILE["seed"])
    base = QposPolicy(args.base)
    if base.control["optimizer_updates"] != PROFILE["base_updates"]:
        raise ValueError("Expected ACT-1 100k")
    frozen, stats = (base.model, base.stats)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    from .skill_replay import DeviceReplay
    from ..learner import SpatialEncoder
    from .skill_reward import TieredReward

    learner = Learner(encoder=SpatialEncoder())
    learner.bound.copy_(torch.as_tensor(BOUNDS / SCALE, device="cuda"))
    capacity = PROFILE["replay_capacity"]
    warmup = PROFILE["critic_warmup"]
    replay = DeviceReplay(capacity)
    out = args.out
    (out / "episodes").mkdir(parents=True)
    first = PROFILE["first_scene_seed"]
    seeds = list(range(first, first + PROFILE["scene_count"]))
    config = dict(
        base="ACT-1 100k",
        actor_target=args.updates,
        batch=args.batch,
        precollect=PRECOLLECT,
        critic_warmup=warmup,
        replay_capacity=capacity,
        scene_horizon_seconds=HORIZON,
        pick_limit=PICK_LIMIT,
        return_limit=RETURN_LIMIT,
        reward="retained_orange_tiers",
        residual_motor=BOUNDS.tolist(),
        gamma_per_16_ticks=0.99,
    )
    publish(out / "config.json", config)
    server = duplex.listener(out)
    publish(
        out / "ready.json",
        dict(
            mode="eval",
            policy="single-orange-sac",
            protocol=load_protocol(),
            benchmark_seconds=HORIZON,
            execution={"steps": 2},
            seeds=seeds,
            start_cursor=0,
            score_wrist_frame=True,
        ),
    )
    publish(out / "status.json", dict(status="starting"))
    peer = None
    progress = {}
    stats_count = dict(
        decisions=0,
        transitions=0,
        physics_steps=0,
        scenes=0,
        skill_episodes=0,
        skill_successes=0,
        visual_successes=0,
    )
    recent = []
    scoring = None
    stage_parts = {}

    @torch.no_grad()
    def pick(obs, q, last_q, last_action):
        qn = (q - CENTER) / SCALE
        dq = (
            np.zeros(6, np.float32)
            if last_q is None
            else ((qn - last_q) * 6).clip(-10, 10)
        )
        numeric = np.r_[qn, dq, last_action].astype(np.float32)
        q_gpu = torch.as_tensor(q, device="cuda", dtype=torch.float32)
        inp = {"observation.state": (q_gpu[None] - stats["qmu"]) / stats["qstd"]}
        small = []
        for cam in ("front", "wrist"):
            raw = (
                torch.as_tensor(obs[cam][0].copy(), device="cuda")
                .permute(2, 0, 1)[None]
                .float()
            )
            small.append(F.interpolate(raw, (96, 128), mode="area")[0].round().byte())
            inp["observation.images." + cam] = (raw.div_(255) - mean) / std
        base_motor = (
            (frozen.select_action(inp) * stats["astd"] + stats["amu"])
            .reshape(6)
            .cpu()
            .numpy()
        )
        base = ((base_motor - CENTER) / SCALE).clip(-1 + 1e-05, 1 - 1e-05)
        images = torch.stack(small)
        obs_id = replay.observe(images, numeric, base)
        if learner.q_updates < warmup or learner.updates >= args.updates:
            motor = base_motor
        else:
            features, base_tensor = learner.features(
                (
                    images[None],
                    torch.as_tensor(numeric[None], device="cuda"),
                    torch.as_tensor(base[None], device="cuda"),
                )
            )
            action = learner.sample(features, base_tensor)[0][0].cpu().numpy()
            motor = action * SCALE + CENTER
        return (obs_id, motor, qn)

    def append(pending, next_id, tick, terminal=False, success=False):
        if pending is None:
            return
        sid, action, then = pending
        parts, discount = scoring.transition(then, tick, terminal, success)
        for key, value in parts.items():
            stage_parts[key] = stage_parts.get(key, 0.0) + value
        replay.append(sid, action, sum(parts.values()), next_id, discount)
        stats_count["transitions"] += 1
        if (
            len(replay.records) >= PRECOLLECT
            and stats_count["transitions"] % 2 == 0
            and (learner.updates < args.updates)
        ):
            warm = learner.q_updates < warmup
            progress.update(
                learner.update(
                    replay.sample(rng, args.batch),
                    critic_only=warm,
                    fixed_base_target=warm,
                )
            )

    def log(stage):
        metrics = dict(
            stage=stage,
            **stats_count,
            replay_size=len(replay.records),
            actor_updates=learner.updates,
            critic_updates=learner.q_updates,
            target=args.updates,
            recent_successes=sum(recent[-10:]),
            recent_episodes=len(recent[-10:]),
            **progress,
        )
        publish(out / "progress.json", metrics)
        with (out / "scalars.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        step=stats_count["decisions"],
                        metrics={
                            f"train/{k}": v
                            for k, v in metrics.items()
                            if isinstance(v, (float, int))
                        },
                    ),
                    allow_nan=False,
                )
                + "\n"
            )

    def save(label):
        temp = out / (label + ".partial")
        torch.save(
            dict(
                **learner.checkpoint(), config=config, **stats_count, replay_saved=False
            ),
            temp,
        )
        temp.replace(out / (label + ".pt"))

    try:
        peer, _ = server.accept()
        peer.settimeout(900)
        query = 0
        last_milestone = 0
        publish(out / "status.json", dict(status="running"))
        for index, seed in enumerate(seeds):
            torch.manual_seed(seed)
            frozen.reset()
            visual = PlacementDetector()
            release = ReleasePlacementTrigger()
            prior_q = None
            prior_action = np.zeros(6, np.float32)
            pending = None
            returning = None
            release_target = None
            scene_steps = stage_start = return_start = scene_successes = 0
            completed_skills = []
            phase = "single-orange-sac"
            publish(
                out / f"episode-{index:06d}.json",
                dict(
                    index=index,
                    seed=seed,
                    prefilled=0,
                    phase=phase,
                    train=True,
                    stop=False,
                ),
            )

            def finish(visual_success, reason):
                nonlocal \
                    pending, \
                    scene_successes, \
                    returning, \
                    release_target, \
                    stage_start, \
                    prior_q
                success, start_count, end_count = score_one_orange(
                    out / f"score-{index:02d}.jsonl", stage_start, scene_steps
                )
                if pending is not None:
                    append(
                        pending, pending[0], scene_steps, terminal=True, success=success
                    )
                if not visual_success and release_target is not None:
                    release.confirmed = release_target - 1
                stage = dict(
                    scene=index,
                    seed=seed,
                    skill=stats_count["skill_episodes"],
                    success=success,
                    reason=reason,
                    started=stage_start,
                    ended=scene_steps,
                    expected_plate_count=release_target,
                    discount=0.0,
                    elapsed=scene_steps - stage_start,
                    return_=sum(stage_parts.values()),
                    reward_parts=dict(stage_parts),
                    visual_success=visual_success,
                    scored_count_before=start_count,
                    scored_count_after=end_count,
                )
                with (out / "skills.jsonl").open("a") as stream:
                    stream.write(json.dumps(stage) + "\n")
                completed_skills.append(stage)
                stats_count["skill_episodes"] += 1
                stats_count["skill_successes"] += int(success)
                stats_count["visual_successes"] += int(visual_success)
                scene_successes += int(visual_success)
                recent.append(success)
                frozen.reset()
                pending = returning = release_target = prior_q = None
                stage_start = scene_steps
                scoring.next_skill()
                stage_parts.clear()

            while True:
                obs = duplex.receive_observation(peer, query)
                if scoring is None:
                    scoring = TieredReward(out / f"score-{index:02d}.jsonl")
                q = obs["joint_motor"].reshape(6)
                seen = visual.update(obs["front"][0], obs["wrist"][0], q)["count"]
                if returning is None:
                    notice = release.update(float(q[-1]), seen)
                    if notice is not None and pending is not None:
                        returning = ReturnHome(q)
                        release_target = notice
                        return_start = scene_steps
                if returning is not None:
                    if at_home(q) and seen == release_target:
                        finish(True, "release_and_home")
                    elif scene_steps - return_start >= RETURN_LIMIT:
                        finish(False, "return_timeout")
                elif scene_steps - stage_start >= PICK_LIMIT:
                    finish(False, "pick_timeout")
                if returning is not None:
                    motor = returning.action(q)
                else:
                    obs_id, motor, prior_q = pick(obs, q, prior_q, prior_action)
                    append(pending, obs_id, scene_steps)
                    pending = (
                        obs_id,
                        ((motor - CENTER) / SCALE).astype(np.float32),
                        scene_steps,
                    )
                prior_action = ((motor - CENTER) / SCALE).astype(np.float32)
                duplex.send_action(peer, np.repeat(motor[None], 2, axis=0))
                timing = duplex.receive_feedback(peer)
                assert set(timing) <= {
                    "query",
                    "seed",
                    "episode_index",
                    "phase",
                    "steps",
                    "terminated",
                    "truncated",
                    "terminal_joint_motor",
                }
                assert timing["query"] == query and timing["seed"] == seed
                query += 1
                scene_steps += timing["steps"]
                stats_count["decisions"] += 1
                stats_count["physics_steps"] += timing["steps"]
                if stats_count["decisions"] == 1 or stats_count["decisions"] % 50 == 0:
                    log("return_home" if returning is not None else "pick")
                if learner.updates >= last_milestone + 2000:
                    last_milestone = learner.updates
                    save(f"updates-{last_milestone:06d}")
                if timing["terminated"] or timing["truncated"]:
                    if pending is not None:
                        reached = returning is not None and at_home(
                            np.asarray(timing["terminal_joint_motor"])
                        )
                        success = bool(reached and seen == release_target)
                        finish(success, "release_and_home" if success else "scene_end")
                    break
            stats_count["scenes"] += 1
            scene_score = scoring.ledger.components(scene_steps)
            progress.update(
                scene_return=sum(scene_score.values()),
                retained_count=scene_score["retained"] / 20,
                native_success=int(bool(scene_score["completion"])),
            )
            log("scene_end")
            scoring.close()
            scoring = None
            publish(
                out / "episodes" / f"{index:06d}.json",
                dict(
                    seed=seed,
                    steps=scene_steps,
                    visual_successes=scene_successes,
                    successes=sum((row["success"] for row in completed_skills)),
                    skills=completed_skills,
                    **dict(
                        reward_parts=scene_score,
                        scene_return=sum(scene_score.values()),
                        retained=int(scene_score["retained"] / 20),
                        native_success=bool(scene_score["completion"]),
                    ),
                ),
            )
            save("latest")
            if learner.updates >= args.updates or (out / "stop-request").exists():
                publish(out / f"episode-{index + 1:06d}.json", dict(stop=True))
                publish(
                    out / "status.json",
                    dict(
                        status="completed"
                        if learner.updates >= args.updates
                        else "paused",
                        **stats_count,
                    ),
                )
                return
        raise RuntimeError("Seed list exhausted")
    finally:
        if scoring is not None:
            scoring.close()
        if peer is not None:
            peer.close()
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=PROFILE["actor_updates"])
    parser.add_argument("--batch", type=int, default=PROFILE["batch"])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            publish(
                args.out / "model-error.json", dict(traceback=traceback.format_exc())
            )
        raise
