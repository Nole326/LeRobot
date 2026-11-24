"""Frozen GR00T + spatial visual Residual SAC with asymmetric critics."""

import argparse, hashlib, json, os, random
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .sac_core import SAC, state
from .rewards import feedback_reward, VERSION, PARAMETERS
from .memory_rgb import receive_observation
from .run_files import run_directory

from ..common import (
    load_protocol,
    predict_groot,
    publish,
    wait_file,
    save_arrays,
    sha256,
)
from ..service import load_policy, verify_model


class VisualHook:
    def __init__(self, policy):
        self.value = None
        self.handle = (
            policy.model.backbone.eagle_model.vision_model.register_forward_hook(
                self.capture
            )
        )

    def capture(self, module, args, result):
        x = result.last_hidden_state if hasattr(result, "last_hidden_state") else result
        assert isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 2, tuple(
            x.shape
        )
        n = x.shape[1]
        side = int(n**0.5)
        if side * side != n:
            assert int((n - 1) ** 0.5) ** 2 == n - 1
            x = x[:, 1:]
            side = int((n - 1) ** 0.5)
        x = x.reshape(2, side, side, -1).permute(0, 3, 1, 2)
        self.value = (
            F.adaptive_avg_pool2d(x.float(), (4, 4))
            .permute(0, 2, 3, 1)
            .reshape(1, 32, -1)
            .detach()
        )


class FeatureFile:
    def __init__(self, path, prior=None):
        self.path = path
        self.file = path.open("xb+")
        self.width = None
        self.count = 0
        self.digest = hashlib.sha256()
        self.segments = []
        self.prior_count = 0
        if prior:
            self.width = prior["width"]
            self.segments = prior.get("segments", [prior])
            for segment in self.segments:
                digest = hashlib.sha256()
                left = segment["bytes"]
                with open(segment["path"], "rb") as f:
                    while left:
                        block = f.read(min(left, 4 * 1024**2))
                        assert block
                        digest.update(block)
                        left -= len(block)
                assert digest.hexdigest() == segment["prefix_sha256"]
                self.prior_count += segment["records"]
            self.count = self.prior_count

    def append(self, x):
        a = x.detach().cpu().numpy().astype("<f2").reshape(-1)
        assert np.isfinite(a).all()
        if self.width is None:
            self.width = len(a)
        assert len(a) == self.width
        data = a.tobytes()
        self.file.seek(0, 2)
        self.file.write(data)
        self.digest.update(data)
        index = self.count
        self.count += 1
        return torch.tensor([[index]], dtype=torch.float32)

    def gather(self, indices):
        self.file.flush()
        result = []
        for i in indices:
            i = int(i)
            if i >= self.prior_count:
                self.file.seek((i - self.prior_count) * self.width * 2)
                data = self.file.read(self.width * 2)
            else:
                for segment in self.segments:
                    if i < segment["records"]:
                        with open(segment["path"], "rb") as f:
                            f.seek(i * self.width * 2)
                            data = f.read(self.width * 2)
                        break
                    i -= segment["records"]
            assert len(data) == self.width * 2
            result.append(np.frombuffer(data, dtype="<f2").astype(np.float32))
        return torch.from_numpy(np.stack(result)).cuda()

    def manifest(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        segment = dict(
            path=str(self.path),
            records=self.count - self.prior_count,
            width=self.width,
            bytes=(self.count - self.prior_count) * self.width * 2,
            prefix_sha256=self.digest.hexdigest(),
            dtype="float16",
            contains_pixels=False,
        )
        return dict(
            **segment, total_records=self.count, segments=self.segments + [segment]
        )


def run(a):
    with run_directory(a.out):
        train(a)


def train(a):
    cfg = load_protocol()
    (a.out / "ipc").mkdir()
    (a.out / "episodes").mkdir()
    cp = a.out / "checkpoints"
    cp.mkdir()
    config = dict(
        target_updates=8 if a.qa else a.target_updates,
        replay_capacity=a.replay_capacity,
        batch_size=32,
        collect=32 if a.qa else 10000,
        critic_warmup_chunks=32 if a.qa else 10000,
        update_every=4,
        warmup_policy="frozen_base",
        gamma_per_16_steps=0.99,
        bootstrap="truncated",
        time_input=False,
        actor_lr=3e-5,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        tau=0.005,
        seed=1704,
        feature_grid=[4, 4],
        feature_source="frozen_vision_model_spatial",
        privileged_dim=30,
        privileged_fields=[
            "orange_minus_plate_xyz",
            "orange_minus_tcp_xyz",
            "orange_linear_velocity",
            "inside_plate",
        ],
    )
    identity = dict(
        config=config,
        protocol=cfg,
        reward_variant=a.reward,
        reward_version=VERSION,
        reward_parameters=PARAMETERS,
        model=verify_model(a.model),
        sources={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )
    policy = load_policy(a.model, a.backend, "cuda:0")
    policy.model.eval().requires_grad_(False)
    hook = VisualHook(policy)
    random.seed(1704)
    np.random.seed(1704)
    torch.manual_seed(1704)
    restored = (
        torch.load(a.resume, map_location="cpu", weights_only=False)
        if a.resume
        else None
    )
    if restored:
        previous = restored["identity"]
        mutable = {"target_updates", "replay_capacity"}
        assert {k: v for k, v in previous["config"].items() if k not in mutable} == {
            k: v for k, v in config.items() if k not in mutable
        }
        for key in (
            "protocol",
            "reward_variant",
            "reward_version",
            "reward_parameters",
            "model",
        ):
            assert previous[key] == identity[key], key
        assert config["target_updates"] >= previous["config"]["target_updates"]
        assert config["replay_capacity"] >= len(restored["replay"])
    feature_root = a.out.resolve()
    segment = a.out.name
    sampler = random.Random(1704)
    store = FeatureFile(
        feature_root / (segment + ".bin"),
        restored["schedule"]["features"] if restored else None,
    )
    replay = deque(maxlen=config["replay_capacity"])
    learner = None
    query = episode = chunks = physics = all_steps = 0
    last_save = 0
    eval_pending = []
    eval_rows = []
    next_eval = 1000
    phase = "train"
    train_episodes = 0
    scale = np.array([100] * 5 + [50], np.float32)
    center = np.array([0] * 5 + [50], np.float32)
    if restored:
        schedule = restored["schedule"]
        episode = schedule["next_episode"]
        chunks = schedule["train_chunks"]
        physics = schedule["train_steps"]
        all_steps = schedule.get("all_steps", physics)
        next_eval = schedule["next_eval"]
        eval_pending = schedule["eval_pending"]
        eval_rows = schedule["eval_rows"]
        train_episodes = schedule.get("train_episodes", episode)
        learner = SAC(
            store.width,
            96,
            "cuda",
            visual_shape=((store.width - 12) // 1152, 1152),
            privileged_dim=30,
        )
        replay_rows, _ = learner.restore(
            a.resume, identity=restored["identity"], sampler=sampler
        )
        replay = deque(replay_rows, maxlen=config["replay_capacity"])
        last_save = learner.actor_updates
    if not a.periodic_validation:
        eval_pending = []
        eval_rows = []
    bound = torch.tensor(
        np.tile(np.array([2] * 5 + [1], np.float32) / scale, 16), device="cuda"
    )[None]
    journal = a.out / "scalars.jsonl"

    def log(values, step):
        assert all(type(v) in (int, float) and np.isfinite(v) for v in values.values())
        with journal.open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=values)) + "\n")

    def observe(k):
        data = receive_observation(a.out, k, timeout=900)
        context = json.loads((a.out / "ipc" / f"context-{k:06d}.json").read_text())
        assert context["query"] == k
        with torch.no_grad():
            base = predict_groot(policy, data, cfg["instruction"]).astype(np.float32)
        assert base.shape == (16, 6) and hook.value is not None
        q = torch.tensor(
            (data["joint_motor"] - center) / scale, device="cuda", dtype=torch.float32
        )
        vel = torch.tensor(context["velocity"], device="cuda", dtype=torch.float32) / 5
        features = state(hook.value.flatten(1), q, vel).half().float()
        return (
            features,
            base,
            torch.tensor(context["privileged"], dtype=torch.float32).reshape(1, 30),
            store.append(features) if phase == "train" else None,
        )

    publish(
        a.out / "ready.json",
        dict(
            protocol=cfg,
            mode="train",
            gamma=0.99,
            reward_variant=a.reward,
            config=config,
            identity=identity,
            start_cursor=episode,
            image_storage="none_memory_socket",
            periodic_validation=a.periodic_validation,
        ),
    )
    publish(a.out / "status.json", dict(status="running"))
    while True:
        phase = "validation" if eval_pending else "train"
        seed = eval_pending.pop(0) if eval_pending else 10001 + train_episodes % 100
        if a.qa:
            seed = 9301 + episode
        job = dict(
            index=episode, seed=seed, phase=phase, train=phase == "train", stop=False
        )
        publish(a.out / f"episode-{episode:06d}.json", job)
        rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        if phase == "validation":
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        current = observe(query)
        if learner is None:
            learner = SAC(
                current[0].shape[-1],
                96,
                "cuda",
                visual_shape=tuple(hook.value.shape[1:]),
                privileged_dim=30,
            )
            publish(
                a.out / "architecture.json",
                dict(
                    visual_shape=list(hook.value.shape),
                    state_width=current[0].shape[-1],
                    actor_params=sum(p.numel() for p in learner.actor.parameters()),
                    q_params=sum(p.numel() for p in learner.q.parameters()),
                ),
            )
        totals = {}
        episode_steps = 0
        updates_before = learner.actor_updates
        while True:
            s, base, priv, sid = current
            nb = torch.tensor((base - center) / scale, device="cuda").reshape(1, 96)
            warm = chunks < config["collect"] + config["critic_warmup_chunks"]
            with torch.no_grad():
                actual, lp, _ = learner.actor(
                    s, nb, bound, deterministic=warm or phase != "train"
                )
            if warm:
                torch.testing.assert_close(
                    actual, nb.clamp(-1 + 1e-5, 1 - 1e-5), atol=2e-6, rtol=1e-5
                )
            actions = actual.reshape(16, 6).cpu().numpy() * scale + center
            save_arrays(a.out / "ipc" / f"action-{query:06d}.npz", actions=actions)
            wait_file(
                a.out / "ipc" / f"feedback-{query:06d}.json",
                900,
                a.out / "sim-error.json",
            )
            fb = json.loads((a.out / "ipc" / f"feedback-{query:06d}.json").read_text())
            parts, discount = feedback_reward(fb, a.reward)
            assert parts == fb["reward_components"]
            ended = fb["terminated"] or fb["truncated"]
            following = observe(query + 1)
            ns, nbase, npriv, nsid = following
            if fb["truncated"]:
                assert json.loads(
                    (a.out / "ipc" / f"context-{query + 1:06d}.json").read_text()
                )["pre_reset"]
            all_steps += fb["steps"]
            episode_steps += fb["steps"]
            if phase == "train":
                chunks += 1
                physics += fb["steps"]
                replay.append(
                    dict(
                        state=sid,
                        next_state=nsid,
                        action=actual.detach().cpu(),
                        reward=torch.tensor([parts["total"]]),
                        discount=torch.tensor([discount]),
                        base=nb.cpu(),
                        next_base=torch.tensor((nbase - center) / scale).reshape(1, 96),
                        privileged=priv,
                        next_privileged=npriv,
                    )
                )
                if (
                    chunks >= config["collect"]
                    and chunks % 4 == 0
                    and learner.actor_updates < config["target_updates"]
                ):
                    indices = sampler.sample(range(len(replay)), 32)
                    rows = [replay[i] for i in indices]
                    batch = {
                        k: torch.cat([r[k] for r in rows], 0).cuda() for k in rows[0]
                    }
                    batch["state"] = store.gather(
                        batch["state"].flatten().cpu().tolist()
                    )
                    batch["next_state"] = store.gather(
                        batch["next_state"].flatten().cpu().tolist()
                    )
                    metrics = learner.update(
                        tuple(
                            batch[k]
                            for k in (
                                "state",
                                "action",
                                "reward",
                                "next_state",
                                "discount",
                                "base",
                                "next_base",
                            )
                        ),
                        bound,
                        critic_only=warm,
                        privileged=batch["privileged"],
                        next_privileged=batch["next_privileged"],
                    )
                    log(
                        {"update/" + k: v for k, v in metrics.items()}
                        | {
                            "update/actor_updates": learner.actor_updates,
                            "update/critic_updates": learner.critic_updates,
                        },
                        all_steps,
                    )
            delta = (actual - nb.clamp(-1 + 1e-5, 1 - 1e-5)).abs() / bound
            log(
                dict(
                    action_residual_fraction=float(delta.mean()),
                    action_residual_near_limit=float((delta > 0.95).float().mean()),
                    base_outside_motor_range=float((nb.abs() > 1).float().mean()),
                    action_motor_boundary_fraction=float(
                        (actual.abs() > 0.999).float().mean()
                    ),
                    train_actor_updates=learner.actor_updates,
                    train_critic_updates=learner.critic_updates,
                ),
                all_steps,
            )
            for k, v in parts.items():
                totals[k] = totals.get(k, 0) + v
            if not a.qa:
                # These numeric IPC messages have already entered replay/metrics.
                # Do not retain millions of redundant physics-level trace samples.
                for label, suffix in (
                    ("action", "npz"),
                    ("feedback", "json"),
                    ("context", "json"),
                ):
                    (a.out / "ipc" / f"{label}-{query:06d}.{suffix}").unlink()
                if ended:
                    (a.out / "ipc" / f"context-{query + 1:06d}.json").unlink()
            query += 1
            publish(
                a.out / "progress.json",
                dict(
                    status="running",
                    phase=phase,
                    episode=episode,
                    actor_updates=learner.actor_updates,
                    critic_updates=learner.critic_updates,
                    train_chunks=chunks,
                    train_steps=physics,
                    all_steps=all_steps,
                    replay_size=len(replay),
                ),
            )
            if ended:
                query += 1
                break
            current = following
        if phase != "train":
            assert learner.actor_updates == updates_before
        record = dict(
            **job,
            steps=episode_steps,
            success=fb["success"],
            updates=learner.actor_updates,
            components=totals,
        )
        publish(a.out / "episodes" / f"{episode:06d}.json", record)
        log(
            {
                phase + "/success": int(fb["success"]),
                phase + "/episode_steps": episode_steps,
                phase + "/return": totals["total"],
            },
            all_steps,
        )
        if phase == "validation":
            eval_rows.append(record)
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
            if not eval_pending:
                log(
                    {
                        "validation/success_rate": sum(r["success"] for r in eval_rows)
                        / len(eval_rows)
                    },
                    all_steps,
                )
        else:
            train_episodes += 1
        episode += 1
        if (
            a.periodic_validation
            and phase == "train"
            and learner.actor_updates >= min(next_eval, config["target_updates"])
            and not a.qa
        ):
            eval_pending = list(range(20001, 20031))
            eval_rows = []
            next_eval = 5000 if next_eval == 1000 else next_eval + 5000
        due = (
            learner.actor_updates // 500 > last_save // 500
            or (learner.actor_updates >= config["target_updates"] and not eval_pending)
            or a.qa
        )
        if due:
            target = (
                cp / f"updates-{learner.actor_updates:06d}-episode-{episode:06d}.pt"
            )
            learner.save(
                target,
                identity=identity,
                replay=list(replay),
                sampler=sampler,
                schedule=dict(
                    episode_complete=True,
                    next_episode=episode,
                    features=store.manifest(),
                    train_chunks=chunks,
                    train_steps=physics,
                    all_steps=all_steps,
                    query=query,
                    next_eval=next_eval,
                    eval_pending=eval_pending,
                    eval_rows=eval_rows,
                    train_episodes=train_episodes,
                ),
            )
            publish(
                a.out / "latest.json",
                dict(
                    path=str(target),
                    sha256=sha256(target),
                    feature_dependency=store.manifest(),
                ),
            )
            if not a.qa:
                if learner.actor_updates // 4000 > last_save // 4000:
                    bundle = torch.load(target, map_location="cpu", weights_only=False)
                    bundle.pop("replay")
                    bundle["purpose"] = "evaluation_only_not_full_resume"
                    milestone = cp / f"milestone-{learner.actor_updates:06d}.pt"
                    torch.save(bundle, milestone)
                    del bundle
                for old in sorted(cp.glob("updates-*.pt"))[:-2]:
                    old.unlink()
            last_save = learner.actor_updates
        if learner.actor_updates >= config["target_updates"] and not eval_pending:
            break
    assert all(
        p.grad is None and not p.requires_grad for p in policy.model.parameters()
    )
    publish(a.out / f"episode-{episode:06d}.json", dict(stop=True))
    publish(
        a.out / "result.json",
        dict(
            status="completed",
            actor_updates=learner.actor_updates,
            critic_updates=learner.critic_updates,
        ),
    )
    publish(a.out / "status.json", dict(status="completed"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reward", choices=("count", "hybrid"), required=True)
    p.add_argument("--qa", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--target-updates", type=int, default=20000)
    p.add_argument("--replay-capacity", type=int, default=100000)
    p.add_argument("--periodic-validation", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--backend", type=Path, required=True)
    a = p.parse_args()
    if a.target_updates < 1 or a.replay_capacity < (32 if a.qa else 10000):
        p.error("invalid update or replay budget")
    run(a)
