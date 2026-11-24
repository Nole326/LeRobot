"""Shared evaluation simulator with independent seeds and action chunk clocks."""

import argparse
import json
import random
import time
import traceback
from pathlib import Path
import numpy as np
from ...common import load_protocol, publish, wait_file
from .. import transport as duplex


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--outs", type=Path, nargs="+", required=True)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = env = None
    peers = []
    try:
        ready = []
        for out in a.outs:
            wait_file(out / "ready.json", 900, out / "model-error.json")
            ready.append(json.loads((out / "ready.json").read_text()))
        n = len(ready)
        cfg = load_protocol()
        seconds = ready[0]["benchmark_seconds"]
        assert seconds == 100
        assert all(
            r["protocol"] == cfg
            and r["mode"] == "train"
            and r["benchmark_seconds"] == seconds
            and r["execution"]["steps"] == 2
            for r in ready
        )
        factorial = all(r.get("factorial_training") for r in ready)
        app = AppLauncher(a).app
        import torch
        import gymnasium as gym
        import leisaac.tasks  # noqa: F401 (environment registration)
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import (
            RecorderManagerBaseCfg,
            RecorderTerm,
        )
        from isaaclab.utils import configclass
        from leisaac.utils.robot_utils import (
            convert_leisaac_action_to_lerobot,
            convert_lerobot_action_to_leisaac,
        )
        from ...expert.joint_contract import JOINTS

        assert factorial
        from ...groot_sac.rewards import geometry, fixed_tcp
        from ...visual.reset_capture import (
            configure_front_reset,
            refresh_reset_observations,
        )

        final_states = {}
        final_observations = {}
        initial_z = {}

        def states(e):
            term = e.cfg.terminations.success
            origin = e.scene.env_origins
            plate = e.scene[term.params["plate_cfg"].name].data.root_pos_w - origin
            oranges = torch.stack(
                [
                    e.scene[o.name].data.root_pos_w - origin
                    for o in term.params["oranges_cfg"]
                ],
                dim=1,
            )
            rel = oranges - plate[:, None]
            bounds = rel.new_tensor(
                [
                    term.params.get("x_range", (-0.1, 0.1)),
                    term.params.get("y_range", (-0.1, 0.1)),
                    term.params.get("height_range", (-0.07, 0.07)),
                ]
            )
            inside = ((rel > bounds[:, 0]) & (rel < bounds[:, 1])).all(-1).tolist()
            success = term.func(e, **term.params).tolist()
            frames = e.scene["ee_frame"].data
            result = []
            for i in range(n):
                tcp = fixed_tcp(
                    (frames.target_pos_w[i, 0] - origin[i]).tolist(),
                    frames.target_quat_w[i, 0].tolist(),
                )
                fruit = oranges[i].tolist()
                dish = plate[i].tolist()
                value = geometry(
                    inside[i],
                    fruit,
                    dish,
                    tcp,
                    initial_z.get(i, oranges[i, :, 2].tolist()),
                    success[i],
                )
                value.update(oranges=fruit, plate=dish)
                result.append(value)
            return result

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if hasattr(self._env, "reset_buf") and bool(self._env.reset_buf.any()):
                    rows = states(self._env)
                    final_obs = self._env.observation_manager.compute()
                    for ix in env_ids:
                        i = int(ix)
                        final_states[i] = rows[i]
                        q = convert_leisaac_action_to_lerobot(
                            self._env.scene["robot"]
                            .data.joint_pos[i : i + 1, ids]
                            .cpu()
                            .numpy()
                        )
                        final_observations[i] = dict(
                            joint_motor=q,
                            **{
                                k: final_obs["policy"][k][i : i + 1]
                                .cpu()
                                .numpy()
                                .copy()
                                for k in ("front", "wrist")
                            },
                        )
                return None, None

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        ec = parse_env_cfg(cfg["task_id"], device=a.device, num_envs=n)
        ec.use_teleop_device("so101leader")
        ec.seed = ready[0]["seeds"][0]
        ec.episode_length_s = seconds
        ec.scene.robot.spawn.rigid_props.disable_gravity = False
        ec.recorders = Recorder()
        configure_front_reset(ec)
        env = gym.make(cfg["task_id"], cfg=ec).unwrapped
        robot = env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt - 1 / 60) < 1e-7
        terms = [
            env.action_manager.get_term(k) for k in env.action_manager.active_terms
        ]
        offsets = torch.cat(
            [
                torch.as_tensor(t._offset, device=env.device).expand(n, t.action_dim)
                for t in terms
            ],
            1,
        )
        scales = torch.cat(
            [
                torch.as_tensor(t._scale, device=env.device).expand(n, t.action_dim)
                for t in terms
            ],
            1,
        )
        # The upstream repeated joint_ids=[5,...] is single-environment oriented.
        # Preserve the exact mass-based rule, using one joint column and per-env rows.
        objects = list(env.scene._rigid_objects.values())

        def gripper():
            pos = torch.stack([o.data.body_link_pos_w[:, 0] for o in objects])
            mass = torch.stack([o.data.default_mass for o in objects])
            nearest = (
                ((pos - robot.data.body_link_pos_w[None, :, -1]) ** 2).sum(-1).argmin(0)
            )
            effort = (mass[nearest.cpu(), torch.arange(n), 0] / 0.15).to(env.device)
            threshold = 0.1
            current = robot._data.joint_effort_limits[:, -1]
            changed = (effort - current).abs() > threshold
            limits = torch.where(changed, effort, current)
            if bool(changed.any()):
                robot.write_joint_effort_limit_to_sim(
                    limits=limits[:, None], joint_ids=[5]
                )

        for out in a.outs:
            publish(out / "sim-ready.json", dict(num_envs=n))
            peer = duplex.connect(out)
            peer.settimeout(180)
            peers.append(peer)
        strides = [r["execution"]["steps"] for r in ready]
        cursors = [r.get("start_cursor", 0) for r in ready]
        query = [0] * n
        totals = [0] * n
        counts = [0] * n
        jobs = [None] * n
        active = [True] * n
        pending = [True] * n
        chunks = [None] * n
        used = [0] * n
        samples = [None] * n
        prior = [None] * n
        round_started = [0.0] * n
        target = robot.data.joint_pos[:, ids].clone()
        ticks = 0

        def pack(i):
            q = convert_leisaac_action_to_lerobot(
                robot.data.joint_pos[i : i + 1, ids].detach().cpu().numpy()
            )
            return dict(
                joint_motor=q,
                **{
                    k: obs["policy"][k][i : i + 1].detach().cpu().numpy().copy()
                    for k in ("front", "wrist")
                },
            )

        while any(active):
            reset_indices = []
            for i, out in enumerate(a.outs):
                if not active[i] or not pending[i]:
                    continue
                path = out / f"episode-{cursors[i]:06d}.json"
                wait_file(path, 180, out / "model-error.json")
                job = json.loads(path.read_text())
                if job["stop"]:
                    active[i] = False
                    publish(
                        out / "sim-result.json",
                        dict(
                            status="completed",
                            steps=totals[i],
                            chunks=query[i],
                            num_envs=n,
                        ),
                    )
                    continue
                assert (
                    job["index"] == cursors[i]
                    and job["seed"] == ready[i]["seeds"][cursors[i]]
                )
                jobs[i] = job
                seed = job["seed"]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                obs, _ = env.reset(
                    seed=seed, env_ids=torch.tensor([i], device=env.device)
                )
                reset_indices.append(i)
                final_states.pop(i, None)
                counts[i] = 0
                used[i] = 0
                chunks[i] = None
                initial_z[i] = [
                    float(
                        env.scene[f"Orange00{j}"].data.root_pos_w[i, 2]
                        - env.scene.env_origins[i, 2]
                    )
                    for j in (1, 2, 3)
                ]
                prior[i] = states(env)[i]
                pending[i] = False
            if reset_indices:
                obs = refresh_reset_observations(env, reset_indices)
            due = [i for i in range(n) if active[i] and chunks[i] is None]
            for i in due:
                round_started[i] = time.perf_counter()
                duplex.send_observation(peers[i], query[i], pack(i))
            for i in (i for i in due if active[i]):
                motor = duplex.receive_action(peers[i], strides[i])
                assert motor.shape == (strides[i], 6) and np.isfinite(motor).all()
                chunks[i] = torch.as_tensor(
                    convert_lerobot_action_to_leisaac(motor), device=env.device
                )
                used[i] = 0
                samples[i] = [prior[i]]
            if not any(active):
                break
            for i in range(n):
                if active[i]:
                    target[i] = chunks[i][used[i]]
            gripper()
            obs, _, term, trunc, _ = env.step((target - offsets) / scales)
            rows = states(env)
            ticks += 1
            for i, out in enumerate(a.outs):
                if not active[i]:
                    continue
                used[i] += 1
                counts[i] += 1
                totals[i] += 1
                terminal = bool(term[i])
                timeout = bool(trunc[i])
                done = terminal or timeout
                row = final_states[i] if done else rows[i]
                samples[i].append(row)
                if done or used[i] == strides[i]:
                    if terminal:
                        timeout = False
                    duplex.send_feedback(
                        peers[i],
                        dict(
                            query=query[i],
                            seed=jobs[i]["seed"],
                            episode_index=cursors[i],
                            phase=jobs[i]["phase"],
                            steps=used[i],
                            terminated=terminal,
                            truncated=timeout,
                            success=row["success"],
                            pre_reset=done,
                            placed_before=sum(prior[i]["inside"]),
                            placed_after=sum(row["inside"]),
                            reward_samples=samples[i],
                            reward=0.0,
                            native_reward=0.0,
                            wall_seconds=time.perf_counter() - round_started[i],
                        ),
                    )
                    if done:
                        duplex.send_observation(
                            peers[i], query[i] + 1, final_observations.pop(i)
                        )
                    query[i] += 1
                    chunks[i] = None
                    prior[i] = row
                    publish(
                        out / "simulation-progress.json",
                        dict(
                            seed=jobs[i]["seed"],
                            steps=totals[i],
                            chunks=query[i],
                            episode_steps=counts[i],
                            num_envs=n,
                            shared_simulation=True,
                        ),
                    )
                    if done:
                        cursors[i] += 1
                        pending[i] = True
    except BaseException:
        for out in a.outs:
            publish(out / "sim-error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        for peer in peers:
            peer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
