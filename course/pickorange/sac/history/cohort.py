"""RGB and edge-aware residual SAC on a frozen ACT-1 80k policy."""

import argparse
import io
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ...common import load_protocol, publish, wait_file
from .. import transport as duplex
from .learner import Learner
from ...groot_sac import rewards as reward_module
from ...act.act_policy import QposPolicy

BUDGET = 30000
CAPACITY = 120000
PRECOLLECT = 5000
CRITIC_WARMUP = 5000
INTERVAL = 4
CHECKPOINT_INTERVAL = 2000
ARM_BOUNDS = [1, 1, 1, 1, 0.6, 1]


def encode_observation(images, numeric, base):
    encoded = []
    for image in images:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=80)
        encoded.append(buffer.getvalue())
    return encoded, numeric, base


def sample_tensors(rows):
    def states(which):
        pixels, numbers, base = [], [], []
        for row in rows:
            files, numeric, action = row[which]
            frames = []
            for payload in files:
                with Image.open(io.BytesIO(payload)) as picture:
                    frames.append(np.asarray(picture).transpose(2, 0, 1))
            pixels.append(frames)
            numbers.append(numeric)
            base.append(action)
        return (
            torch.as_tensor(np.asarray(pixels), device="cuda"),
            torch.as_tensor(np.asarray(numbers, np.float32), device="cuda"),
            torch.as_tensor(np.asarray(base, np.float32), device="cuda"),
        )

    return (
        states(0),
        torch.as_tensor(np.asarray([r[1] for r in rows], np.float32), device="cuda"),
        torch.as_tensor(np.asarray([r[2] for r in rows], np.float32), device="cuda"),
        states(3),
        torch.as_tensor(np.asarray([r[4] for r in rows], np.float32), device="cuda"),
    )


class Replay:
    """Fixed-size transition ring and reference-counted compressed observations."""

    def __init__(self):
        self.rows = []
        self.states = {}
        self.refs = {}
        self.next_id = 0
        self.offset = 0
        self.image_bytes = 0

    def observe(self, images, numeric, base):
        index = self.next_id
        self.next_id += 1
        encoded, numeric, base = encode_observation(images, numeric, base)
        self.states[index] = (encoded, numeric, base)
        self.image_bytes += sum(map(len, encoded))
        self.refs[index] = 0
        return index

    def append(self, row):
        for idx in (row[0], row[3]):
            self.refs[idx] += 1
        if len(self.rows) == CAPACITY:
            for idx in (self.rows[self.offset][0], self.rows[self.offset][3]):
                self.refs[idx] -= 1
                if self.refs[idx] == 0:
                    self.image_bytes -= sum(map(len, self.states.pop(idx)[0]))
                    del self.refs[idx]
            self.rows[self.offset] = row
            self.offset = (self.offset + 1) % CAPACITY
        else:
            self.rows.append(row)

    def sample(self, rng, batch):
        rows = [self.rows[i] for i in rng.choice(len(self.rows), batch, replace=False)]
        return sample_tensors(
            [
                (self.states[s], action, reward, self.states[ns], discount)
                for s, action, reward, ns, discount in rows
            ]
        )


def shared_observation(images, numeric, base):
    encoded, numeric, base = encode_observation(images, numeric, base)
    return encoded, numeric.tolist(), base.tolist()


def shared_batch(rows, reward):
    index = int(reward == "hybrid")
    return sample_tensors(
        [
            (old, action, rewards[index], new, discount)
            for _, (_, old, new, action, rewards, discount, _, _) in rows
        ]
    )


def check_shared_batch(client, path, source, args, learner, decisions):
    from collections import Counter
    from .pool import CAPACITY, ReplayClient

    before = client.request("stats")
    client.close()
    client = ReplayClient(path, source)
    after = client.request("stats")
    assert before["capacity"] == after["capacity"] == CAPACITY
    assert before["next_sequence"] == after["next_sequence"] == decisions
    assert before["counts"][source] == after["counts"][source] == decisions
    deadline = time.monotonic() + 180
    while min(after["counts"].values()) < args.batch:
        if time.monotonic() >= deadline:
            raise RuntimeError("Shared replay sources did not reach the smoke budget")
        time.sleep(0.25)
        after = client.request("stats")
    rows = client.request("sample", batch=args.batch, seed=1704)
    sources = Counter(name for name, _ in rows)
    assert sources[source] == args.batch // 2 and len(sources) == 4
    assert (
        max(v for k, v in sources.items() if k != source)
        - min(v for k, v in sources.items() if k != source)
        <= 1
    )
    assert any(row[4][0] != row[4][1] for _, row in rows)
    assert len({(name, row[0]) for name, row in rows}) == args.batch
    sample = shared_batch(rows, args.reward)
    before_actor = [p.detach().clone() for p in learner.actor.parameters()]
    metrics = learner.update(sample)
    assert learner.updates == learner.q_updates == 1
    assert any(
        not torch.equal(p, q) for p, q in zip(before_actor, learner.actor.parameters())
    )
    assert all(np.isfinite(x) for x in metrics.values())
    client.close()
    return dict(
        status="passed",
        real_transitions=decisions,
        sources=dict(sources),
        reconnect=True,
        reward=args.reward,
        dual_reward_rows=sum(row[4][0] != row[4][1] for _, row in rows),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        **metrics,
    )


def memory():
    return (
        int(
            next(
                row.split()[1]
                for row in Path("/proc/meminfo").read_text().splitlines()
                if row.startswith("MemAvailable:")
            )
        )
        / 2**20
    )


def main(args):
    budget = args.actor_updates
    torch.set_num_threads(2)
    torch.manual_seed(1704)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(1704)
    out = args.out
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / "episodes").mkdir(exist_ok=args.resume)
    base = QposPolicy(args.base)
    if base.control["optimizer_updates"] != 80000:
        raise ValueError("This cohort requires ACT-1 80k")
    policy, stats = base.model, base.stats
    learner = Learner(edge_aware=args.encoder == "edges", vectorized_augment=True)
    learner.bound.copy_(
        torch.tensor(ARM_BOUNDS, device="cuda")
        / torch.tensor([100] * 5 + [50], device="cuda")
    )
    replay = Replay()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    scale = np.array([100] * 5 + [50], np.float32)
    center = np.array([0] * 5 + [50], np.float32)
    seeds = list(range(881000, 891000))
    shared = None
    collectors = {}
    counts = {}
    collected_at = {}
    collection_models = {args.encoder == "edges": learner}
    if args.shared_collect:
        from .pool import CAPACITY as SHARED_CAPACITY, SOURCES, ReplayClient

        seeds = list(
            range(
                args.collection_first_seed, args.collection_first_seed + SHARED_CAPACITY
            )
        )
        for source in SOURCES:
            client = ReplayClient(args.shared_collect, source)
            client.request("close_interrupted_episode")
            collectors[source] = client
        counts = client.request("stats")["counts"]
    if args.shared_smoke:
        from .pool import SOURCES, ReplayClient

        shared_path, source = args.shared_smoke
        first_seed = 921000 + 100 * SOURCES.index(source)
        seeds = list(range(first_seed, first_seed + 100))
        shared = ReplayClient(shared_path, source)
    shared_training = bool(args.shared_train)
    own_writes = 0
    if shared_training:
        from .pool import CAPACITY as SHARED_CAPACITY, ReplayClient

        shared_path, source = args.shared_train
        shared = ReplayClient(shared_path, source)
        snapshot = shared.request("stats")
        assert (
            snapshot["capacity"] == SHARED_CAPACITY
            and min(snapshot["counts"].values()) >= PRECOLLECT
        )
        sequence = shared.request("close_interrupted_episode")
    config = dict(
        base="ACT-1 80k",
        encoder=args.encoder,
        reward=args.reward,
        batch=args.batch,
        replay_capacity=CAPACITY,
        precollection=PRECOLLECT,
        critic_only_updates=CRITIC_WARMUP,
        actor_updates=budget,
        update_every=INTERVAL,
        step_lim=6000,
        bootstrap="truncated",
        execution_ticks=2,
        act_chunk=15,
        act_ensemble=0.001,
        residual_motor=ARM_BOUNDS,
        gamma_per_16_physics_ticks=0.99,
        replay_images="RAM-only JPEG80",
        symmetric_critic=True,
        warmup_target="fixed ACT",
        source_test=args.smoke,
    )
    decisions = ticks = episodes = successes = 0
    if args.resume:
        restored = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if shared_training and restored["config"].get("shared_replay"):
            config["shared_replay"] = restored["config"]["shared_replay"]
        previous_config = restored["config"]
        assert restored["episode_boundary"]
        if shared_training:
            assert budget >= previous_config["actor_updates"]
            previous_config = dict(previous_config, actor_updates=budget)
        assert previous_config == config and restored["actor_updates"] <= budget
        learner.restore(restored)
        rng.bit_generator.state = restored["sampler"]
        torch.set_rng_state(restored["torch_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        decisions, ticks, episodes = (
            restored[k] for k in ("decisions", "physics_steps", "episodes")
        )
        successes = sum(
            json.loads(p.read_text())["success"]
            for p in (out / "episodes").glob("*.json")
        )
        assert len(list((out / "episodes").glob("*.json"))) == episodes
        del restored
        old = out / f"previous-segment-{time.time_ns()}"
        old.mkdir()
        for name in (
            "ready.json",
            "rgb.sock",
            "sim-ready.json",
            "sim-error.json",
            "model-error.json",
            "stop-request",
            "sim-result.json",
            "simulation-progress.json",
            f"episode-{episodes:06d}.json",
        ):
            path = out / name
            if path.exists():
                path.rename(old / name)
    if shared_training:
        config["shared_replay"] = dict(
            capacity=SHARED_CAPACITY, source=source, own_fraction=0.5
        )
    last_logged = 0
    if args.resume and (out / "scalars.jsonl").is_file():
        with (out / "scalars.jsonl").open() as stream:
            last_logged = max((json.loads(line)["step"] for line in stream), default=0)
    publish(out / "config.json", config)
    server = duplex.listener(out)
    publish(
        out / "ready.json",
        dict(
            mode="train",
            policy="ACT-residual",
            protocol=load_protocol(),
            benchmark_seconds=100,
            execution={"steps": 2},
            seeds=seeds,
            start_cursor=episodes,
            reward_variant=args.reward,
            gamma=0.99,
            factorial_training=True,
        ),
    )
    publish(out / "status.json", dict(status="starting"))
    wait_file(out / "sim-ready.json", 900, out / "sim-error.json")
    peer, _ = server.accept()
    server.close()
    peer.settimeout(900)

    @torch.no_grad()
    def prepare(obs, previous_q, previous_action, elapsed_ticks=2):
        q = np.asarray(obs["joint_motor"], np.float32).reshape(6)
        qn = (q - center) / scale
        dq = (
            np.zeros(6, np.float32)
            if previous_q is None
            else ((qn - previous_q) * (12 / elapsed_ticks)).clip(-10, 10)
        )
        batch = {
            "observation.state": (
                torch.as_tensor(q, device="cuda")[None] - stats["qmu"]
            )
            / stats["qstd"]
        }
        pixels = []
        for camera in ("front", "wrist"):
            rgb = torch.as_tensor(obs[camera].copy(), device="cuda").permute(0, 3, 1, 2)
            batch["observation.images." + camera] = (rgb.float() / 255 - mean) / std
            pixels.append(
                F.interpolate(rgb.float(), (120, 160), mode="area")[0].round().byte()
            )
        base = (
            (policy.select_action(batch) * stats["astd"] + stats["amu"])
            .cpu()
            .numpy()
            .reshape(6)
        )
        numeric = np.r_[qn, dq, previous_action].astype(np.float32)
        bn = ((base - center) / scale).clip(-1 + 1e-5, 1 - 1e-5).astype(np.float32)
        return (
            torch.stack(pixels).permute(0, 2, 3, 1).cpu().numpy(),
            numeric,
            bn,
            qn,
            base,
        )

    query = 0
    metrics = {}
    started = time.monotonic()
    last_saved = -1

    def status(phase):
        shared_counts = shared.request("stats")["counts"] if shared_training else {}
        progress = dict(
            phase=phase,
            decisions=decisions,
            physics_steps=ticks,
            actor_updates=learner.updates,
            critic_updates=learner.q_updates,
            actor_update_target=budget,
            replay_size=sum(shared_counts.values())
            if shared_training
            else len(replay.rows),
            episodes=episodes,
            successes=successes,
            replay_jpeg_bytes=replay.image_bytes,
            available_ram_gib=memory(),
            elapsed_seconds=time.monotonic() - started,
            **metrics,
        )
        if shared_training:
            progress.update(
                shared_source_size=shared_counts[source], shared_own_writes=own_writes
            )
        publish(out / "progress.json", progress)
        if decisions > last_logged:
            with (out / "scalars.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            step=decisions,
                            metrics={
                                f"train/{k}": v
                                for k, v in progress.items()
                                if isinstance(v, (float, int))
                            },
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )

    def save(label, boundary):
        state = learner.checkpoint()
        state.update(
            config=config,
            decisions=decisions,
            physics_steps=ticks,
            episodes=episodes,
            sampler=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            episode_boundary=boundary,
            replay_saved=False,
        )
        path = out / (label + ".pt")
        pending = path.with_suffix(".partial")
        torch.save(state, pending)
        pending.replace(path)

    publish(out / "status.json", dict(status="running"))
    while True:
        if collectors:
            if (
                min(counts.values()) == SHARED_CAPACITY // len(collectors)
                or (out / "stop-request").exists()
            ):
                publish(out / f"episode-{episodes:06d}.json", dict(stop=True))
                publish(out / "status.json", dict(status="completed", counts=counts))
                peer.close()
                for client in collectors.values():
                    client.close()
                return
            source = min(counts, key=counts.get)
            shared = collectors[source]
            sequence = shared.request("stats")["next_sequence"]
            path = out.parent / f"residual-{source}" / "latest.pt"
            with path.open("rb") as stream:
                weights = torch.load(stream, map_location="cpu", weights_only=False)
            assert weights["episode_boundary"] and weights["actor_updates"] > 0
            assert weights["config"]["base"] == config["base"]
            edges = weights["config"]["encoder"] == "edges"
            if edges not in collection_models:
                collection_models[edges] = Learner(
                    edge_aware=edges, vectorized_augment=True
                )
                collection_models[edges].bound.copy_(learner.bound)
            learner = collection_models[edges]
            learner.actor.load_state_dict(weights["actor"])
            learner.encoder.load_state_dict(weights["encoder"])
            collected_at[source] = weights["actor_updates"]
            del weights
        policy.reset()
        seed = seeds[episodes]
        ep_ticks, returns = 0, 0.0
        publish(
            out / f"episode-{episodes:06d}.json",
            dict(index=episodes, seed=seed, phase="train", train=True, stop=False),
        )
        obs = duplex.receive_observation(peer, query)
        image, numeric, base, previous_q, native_motor = prepare(
            obs, None, np.zeros(6, np.float32)
        )
        if shared:
            first_observation = shared_observation(image, numeric, base)
        else:
            sid = replay.observe(image, numeric, base)
        while True:
            warm = learner.q_updates < CRITIC_WARMUP
            if not collectors and (
                warm
                or (not shared_training and len(replay.rows) < PRECOLLECT)
                or learner.updates >= budget
            ):
                action, motor = base, native_motor
            else:
                with torch.no_grad():
                    data = [
                        torch.as_tensor(
                            image[None].transpose(0, 1, 4, 2, 3), device="cuda"
                        ),
                        torch.as_tensor(numeric[None], device="cuda"),
                        torch.as_tensor(base[None], device="cuda"),
                    ]
                    features, base_tensor = learner.features(data)
                    action = learner.sample(features, base_tensor)[0][0].cpu().numpy()
                motor = action * scale + center
            duplex.send_action(peer, np.repeat(motor[None], 2, axis=0))
            row = duplex.receive_feedback(peer)
            assert row["query"] == query and row["seed"] == seed
            query += 1
            decisions += 1
            ticks += row["steps"]
            ep_ticks += row["steps"]
            row.update(reward_variant=args.reward, reward_dt=1 / 60)
            reward, discount = reward_module.feedback_reward(row, args.reward)
            returns += reward["total"]
            done = row["terminated"] or row["truncated"]
            if row["truncated"]:
                assert ep_ticks == 6000 and row["pre_reset"] and discount > 0
            obs = duplex.receive_observation(peer, query)
            image, numeric, base, previous_q, native_motor = prepare(
                obs, previous_q, action, row["steps"]
            )
            encoded_action = ((motor - center) / scale).astype(np.float32)
            if shared:
                reward_pair = [
                    reward_module.feedback_reward(
                        dict(row, reward_variant=variant), variant
                    )
                    for variant in ("count", "hybrid")
                ]
                assert all(value[1] == discount for value in reward_pair)
                observed = shared_observation(image, numeric, base)
                seq = sequence if collectors or shared_training else decisions - 1
                append = dict(
                    sequence=seq,
                    previous=first_observation,
                    following=observed,
                    action=encoded_action.tolist(),
                    rewards=[value[0]["total"] for value in reward_pair],
                    discount=discount,
                    terminated=row["terminated"],
                    truncated=row["truncated"],
                )
                try:
                    committed = shared.request("append", **append)
                except (EOFError, OSError):
                    if not shared_training:
                        raise
                    shared.close()
                    shared = ReplayClient(shared_path, source)
                    assert shared.request("stats")["next_sequence"] in (seq, seq + 1)
                    committed = shared.request("append", **append)
                assert committed == seq + 1
                first_observation = None
                if collectors:
                    sequence = committed
                    counts[source] = min(SHARED_CAPACITY // len(collectors), sequence)
                    if decisions == 1 or decisions % 25 == 0 or done:
                        publish(
                            out / "progress.json",
                            dict(
                                source=source,
                                seed=seed,
                                counts=counts,
                                policy_updates=collected_at,
                                decisions=decisions,
                                episodes=episodes + int(done),
                                available_ram_gib=memory(),
                            ),
                        )
                elif shared_training:
                    sequence = committed
                    own_writes += 1
                elif decisions == args.batch:
                    result = check_shared_batch(
                        shared, shared_path, source, args, learner, decisions
                    )
                    publish(out / "smoke.json", result)
                    peer.close()
                    return
                if not shared_training:
                    if done:
                        episodes += 1
                        break
                    continue
            else:
                next_id = replay.observe(image, numeric, base)
                replay.append(
                    (
                        sid,
                        encoded_action,
                        np.float32(reward["total"]),
                        next_id,
                        np.float32(discount),
                    )
                )
                sid = next_id
            if args.smoke and decisions == args.batch:
                sample = replay.sample(rng, args.batch)
                before = [p.detach().clone() for p in learner.actor.parameters()]
                first = learner.update(sample, critic_only=True, fixed_base_target=True)
                assert learner.q_updates == 1 and learner.updates == 0
                assert all(
                    torch.equal(p, q)
                    for p, q in zip(before, learner.actor.parameters())
                )
                second = learner.update(sample)
                assert learner.q_updates == 2 and learner.updates == 1
                assert any(
                    not torch.equal(p, q)
                    for p, q in zip(before, learner.actor.parameters())
                )
                assert all(np.isfinite(x) for x in (*first.values(), *second.values()))
                publish(
                    out / "smoke.json",
                    dict(
                        status="passed",
                        batch=args.batch,
                        real_transitions=decisions,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        process_peak_rss_mib=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        / 1024,
                        replay_jpeg_bytes=replay.image_bytes,
                        available_ram_gib=memory(),
                        **second,
                    ),
                )
            if (
                not args.smoke
                and (shared_training or len(replay.rows) >= PRECOLLECT)
                and decisions % INTERVAL == 0
                and learner.updates < budget
            ):
                if shared_training:
                    sample = shared_batch(
                        shared.request(
                            "sample", batch=args.batch, seed=int(rng.integers(2**32))
                        ),
                        args.reward,
                    )
                else:
                    sample = replay.sample(rng, args.batch)
                metrics = learner.update(
                    sample, critic_only=warm, fixed_base_target=warm
                )
            if (
                decisions == 1
                or decisions % 50 == 0
                or (shared_training and own_writes == INTERVAL)
            ):
                status(
                    "precollection"
                    if not shared_training and len(replay.rows) < PRECOLLECT
                    else ("critic_warmup" if warm else "sac")
                )
            if (
                not args.smoke
                and learner.updates
                and learner.updates % CHECKPOINT_INTERVAL == 0
                and learner.updates != last_saved
            ):
                save(f"updates-{learner.updates:06d}", done)
                last_saved = learner.updates
            if done:
                successes += int(row["success"])
                publish(
                    out / "episodes" / f"{episodes:06d}.json",
                    dict(
                        seed=seed,
                        success=row["success"],
                        physics_steps=ep_ticks,
                        return_=returns,
                        placed=row["placed_after"],
                        truncated=row["truncated"],
                    ),
                )
                episodes += 1
                break
        if collectors:
            continue
        if not args.smoke:
            save("latest", True)
        handoff = (
            not shared_training
            and not args.smoke
            and learner.updates >= args.private_until
        )
        stop = (
            handoff
            or (out / "stop-request").exists()
            or (args.smoke and decisions >= args.batch)
            or learner.updates >= budget
        )
        if stop:
            publish(out / f"episode-{episodes:06d}.json", dict(stop=True))
            complete = args.smoke or learner.updates >= budget
            publish(
                out / "status.json",
                dict(
                    status="completed" if complete else "paused",
                    decisions=decisions,
                    actor_updates=learner.updates,
                    critic_updates=learner.q_updates,
                ),
            )
            peer.close()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    profiles = json.loads(Path(__file__).with_name("cohort_profiles.json").read_text())
    parser.add_argument("--profile", choices=profiles, required=True)
    parser.add_argument("--batch", type=int, choices=(128, 256), default=256)
    parser.add_argument("--actor-updates", type=int, default=BUDGET)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shared-smoke", nargs=2, metavar=("SOCKET", "SOURCE"))
    parser.add_argument("--shared-collect", type=Path, metavar="SOCKET")
    parser.add_argument("--shared-train", nargs=2, metavar=("SOCKET", "SOURCE"))
    parser.add_argument("--collection-first-seed", type=int, default=931000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = profiles[args.profile]
    args.encoder, args.reward = profile["encoder"], profile["reward"]
    args.private_until = profile["private_actor_updates"]
    if args.shared_train and args.shared_train[1] != args.profile:
        parser.error("The replay source must match the training profile")
    if args.shared_smoke and (not args.smoke or args.resume):
        parser.error("--shared-smoke requires a fresh --smoke run")
    if args.shared_collect and (args.smoke or args.resume):
        parser.error("--shared-collect requires a fresh collector output directory")
    if args.shared_train and (
        not args.resume or args.smoke or args.shared_collect or args.shared_smoke
    ):
        parser.error("--shared-train requires a formal --resume run")
    if args.actor_updates != BUDGET and not args.shared_train:
        parser.error("A custom Actor budget requires --shared-train")
    try:
        main(args)
    except BaseException:
        if args.out.is_dir():
            publish(
                args.out / "model-error.json", dict(traceback=traceback.format_exc())
            )
        raise
