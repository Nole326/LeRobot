"""Shared evaluation simulator with independent seeds and action chunk clocks."""

import argparse
import json
import random
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
    score_streams = {}
    try:
        ready = []
        for out in a.outs:
            wait_file(out / "ready.json", 900, out / "model-error.json")
            ready.append(json.loads((out / "ready.json").read_text()))
        n = len(ready)
        cfg = load_protocol()
        horizon = ready[0]["benchmark_seconds"]
        assert all(
            (
                r["protocol"] == cfg
                and r["mode"] == "eval"
                and (r["benchmark_seconds"] == horizon)
                and (r["execution"]["steps"] == 2)
                for r in ready
            )
        )
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

        final_states = {}
        reward_wrist_frame = any((r.get("score_wrist_frame", False) for r in ready))
        trace_joints = trace_links = None

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
            rows = [
                dict(
                    inside=inside[i],
                    oranges=oranges[i].tolist(),
                    plate=plate[i].tolist(),
                    success=success[i],
                )
                for i in range(n)
            ]
            if trace_joints is not None:
                robot_data = e.scene["robot"].data
                links = robot_data.body_link_pos_w[:, trace_links] - origin[:, None]
                velocity = torch.stack(
                    [
                        e.scene[o.name].data.root_lin_vel_w
                        for o in term.params["oranges_cfg"]
                    ],
                    dim=1,
                )
                joints = robot_data.joint_pos[:, trace_joints]
                qvel = robot_data.joint_vel[:, trace_joints]
                effort = robot_data.applied_torque[:, trace_joints[-1:]]
                features = torch.cat(
                    (joints, qvel, effort, links.flatten(1), velocity.flatten(1)), dim=1
                ).tolist()
                if reward_wrist_frame:
                    quaternions = robot_data.body_link_quat_w[
                        :, trace_links[0]
                    ].tolist()
                    for row, quat in zip(rows, quaternions):
                        row["gripper_quat"] = quat
                for row, values in zip(rows, features):
                    row.update(
                        q_rad=values[:6],
                        qd_rad_s=values[6:12],
                        gripper_torque_nm=values[12],
                        gripper_xyz=values[13:16],
                        jaw_xyz=values[16:19],
                        orange_velocities=[values[s : s + 3] for s in (19, 22, 25)],
                    )
            return rows

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if hasattr(self._env, "reset_buf") and bool(self._env.reset_buf.any()):
                    rows = states(self._env)
                    for i in env_ids:
                        final_states[int(i)] = rows[int(i)]
                return (None, None)

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        ec = parse_env_cfg(cfg["task_id"], device=a.device, num_envs=n)
        ec.use_teleop_device("so101leader")
        ec.seed = ready[0]["seeds"][0]
        ec.episode_length_s = horizon
        ec.scene.robot.spawn.rigid_props.disable_gravity = False
        ec.dynamic_reset_gripper_effort_limit = False
        assert all(
            (
                r.get("policy") in ("placement-vision", "single-orange-sac")
                for r in ready
            )
        )
        from ...visual.reset_capture import (
            configure_front_reset,
            refresh_reset_observations,
        )

        configure_front_reset(ec)
        ec.recorders = Recorder()
        env = gym.make(cfg["task_id"], cfg=ec).unwrapped
        robot = env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt - 1 / 60) < 1e-07
        trace_joints = ids
        trace_links, link_names = robot.find_bodies(
            ["gripper", "jaw"], preserve_order=True
        )
        assert link_names == ["gripper", "jaw"]
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
        assert env.cfg.dynamic_reset_gripper_effort_limit is False
        effort = robot.data.joint_effort_limits[:, ids[-1]].clone()
        assert torch.allclose(effort, torch.full_like(effort, 10.0))
        for out in a.outs:
            publish(
                out / "sim-ready.json",
                dict(
                    num_envs=n,
                    synchronized_reset=True,
                    fixed_gripper=True,
                    gripper_effort_limits=effort.tolist(),
                ),
            )
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
        prior = [None] * n
        target = robot.data.joint_pos[:, ids].clone()

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

        ticks = 0
        while any(active):
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
                assert job["prefilled"] == 0
                prefilled = []
                obs = refresh_reset_observations(env, [i])
                final_states.pop(i, None)
                counts[i] = 0
                used[i] = 0
                chunks[i] = None
                prior[i] = states(env)[i]
                pending[i] = False
                front = env.scene["front"].data
                prior[i]["front_extrinsic"] = (
                    front.pos_w[i].tolist() + front.quat_w_ros[i].tolist()
                )
                stream = (out / f"score-{cursors[i]:02d}.jsonl").open("w", buffering=1)
                score_streams[i] = stream
                stream.write(
                    json.dumps(dict(step=0, prefilled=prefilled, **prior[i])) + "\n"
                )
            due = [i for i in range(n) if active[i] and chunks[i] is None]
            for i in due:
                duplex.send_observation(peers[i], query[i], pack(i))
            for i in due:
                motor = duplex.receive_action(peers[i], strides[i])
                assert motor.shape == (strides[i], 6) and np.isfinite(motor).all()
                chunks[i] = torch.as_tensor(
                    convert_lerobot_action_to_leisaac(motor), device=env.device
                )
                used[i] = 0
            if not any(active):
                break
            for i in range(n):
                if active[i]:
                    target[i] = chunks[i][used[i]]
            if ticks % 60 == 0:
                assert torch.equal(effort, robot.data.joint_effort_limits[:, ids[-1]])
            obs, _, term, trunc, _ = env.step((target - offsets) / scales)
            ticks += 1
            rows = states(env)
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
                score_streams[i].write(json.dumps(dict(step=counts[i], **row)) + "\n")
                if done:
                    score_streams.pop(i).close()
                if done or used[i] == strides[i]:
                    feedback = dict(
                        query=query[i],
                        seed=jobs[i]["seed"],
                        episode_index=cursors[i],
                        phase=jobs[i]["phase"],
                        steps=used[i],
                        terminated=terminal,
                        truncated=timeout,
                    )
                    if done and ready[i].get("policy") == "single-orange-sac":
                        feedback["terminal_joint_motor"] = (
                            convert_leisaac_action_to_lerobot(
                                np.asarray([row["q_rad"]], np.float32)
                            )[0].tolist()
                        )
                    duplex.send_feedback(peers[i], feedback)
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
        for stream in score_streams.values():
            stream.close()
        for peer in peers:
            peer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
