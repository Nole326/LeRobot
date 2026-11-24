"""Drive synchronized PickOrange training scenes in the Isaac Lab Python environment."""

import argparse
import json
import random
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from ..common import wait_file
from . import transport
from .reward import SceneLedger


def simulate(a):
    import gymnasium as gym
    import leisaac.tasks  # noqa: F401 -- register the PickOrange gym environment
    import torch
    from isaaclab.managers import DatasetExportMode, RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import parse_env_cfg
    from leisaac.utils.robot_utils import (
        convert_leisaac_action_to_lerobot,
        convert_lerobot_action_to_leisaac,
    )

    from ..expert.joint_contract import JOINTS
    from ..visual.reset_capture import configure_front_reset, refresh_reset_observations

    ready = []
    for out in a.runs:
        wait_file(out / "ready.json", 900, out / "model-error.json")
        ready.append(json.loads((out / "ready.json").read_text()))
    n = len(ready)
    ee = ready[0]["control"] == "ee"
    testing = ready[0].get("evaluation", False)
    if any(
        (c["control"], c.get("evaluation", False)) != (ready[0]["control"], testing)
        for c in ready
    ) or (ee and n != 1 and not testing):
        raise ValueError("Use a single EE scene or one or more qpos scenes")
    if ee:
        from .ik import EEAction

        if a.robot_usd is None or not a.robot_usd.is_file():
            raise ValueError(
                "EE training requires --robot-usd pointing to the calibrated SO-101 asset"
            )
        adapter = EEAction(a.robot_usd)
    final_states = {}
    trace_joints = trace_links = None
    reward_wrist_frame = True

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
                quaternions = robot_data.body_link_quat_w[:, trace_links[0]].tolist()
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
            return None, None

    @configclass
    class Recorder(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset = False
        final_state = RecorderTermCfg(class_type=Capture)

    ec = parse_env_cfg("LeIsaac-SO101-PickOrange-v0", device=a.device, num_envs=n)
    ec.use_teleop_device("so101leader")
    ec.seed = ready[0]["first_scene_seed"]
    limits = torch.tensor([c["physics_limit"] for c in ready], device=a.device)
    ec.terminations.time_out.func = lambda env: env.episode_length_buf >= limits
    ec.episode_length_s = max(c["physics_limit"] for c in ready) / 60
    ec.scene.robot.spawn.rigid_props.disable_gravity = False
    dynamic = (
        ready[0].get("gripper_effort", "dynamic" if ee or testing else "fixed-10Nm")
        == "dynamic"
    )
    ec.dynamic_reset_gripper_effort_limit = dynamic
    sync = ready[0].get("synchronized_reset", ee or not testing)
    if testing and any(
        c.get("gripper_effort") != ready[0].get("gripper_effort")
        or c.get("synchronized_reset") != sync
        or c.get("required_slots") not in (None, n)
        for c in ready
    ):
        raise ValueError(
            "Scene count, reset and gripper settings must match the evaluation profiles"
        )
    if sync:
        configure_front_reset(ec)
    ec.recorders = Recorder()
    with ExitStack() as resources:
        env = gym.make("LeIsaac-SO101-PickOrange-v0", cfg=ec).unwrapped
        resources.callback(env.close)
        robot = env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt - 1 / 60) < 1e-7
        trace_joints = ids
        trace_links, _ = robot.find_bodies(["gripper", "jaw"], preserve_order=True)
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

        def gripper():
            objects = list(env.scene._rigid_objects.values())
            pos = torch.stack([o.data.body_link_pos_w[:, 0] for o in objects])
            mass = torch.stack([o.data.default_mass for o in objects])
            nearest = (
                ((pos - robot.data.body_link_pos_w[None, :, -1]) ** 2).sum(-1).argmin(0)
            )
            target = (mass[nearest.cpu(), torch.arange(n), 0] / 0.15).to(env.device)
            current = robot._data.joint_effort_limits[:, -1]
            limits = torch.where((target - current).abs() > 0.1, target, current)
            if bool(((target - current).abs() > 0.1).any()):
                robot.write_joint_effort_limit_to_sim(
                    limits=limits[:, None], joint_ids=[5]
                )

        peers = [resources.enter_context(transport.connect(out)) for out in a.runs]
        counters = [0] * n
        query = [0] * n
        ledgers = [None] * n
        running = [True] * n
        seeds = [None] * n
        targets = robot.data.joint_pos[:, ids].clone()
        while any(running):
            for i, peer in enumerate(peers):
                if not running[i] or ledgers[i] is not None:
                    continue
                job = transport.receive_feedback(peer)
                if job.get("stop"):
                    running[i] = False
                    continue
                seed = seeds[i] = job["seed"]
                limit = job.get("physics_limit", ready[i]["physics_limit"])
                if limit != ready[i]["physics_limit"] and not (
                    not testing and ready[i]["physics_limit"] == 400 and limit == 700
                ):
                    raise ValueError("Unexpected episode horizon")
                limits[i] = limit
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                obs, _ = env.reset(
                    seed=seed, env_ids=torch.tensor([i], device=env.device)
                )
                if sync:
                    obs = refresh_reset_observations(env, [i])
                final_states.pop(i, None)
                counters[i] = 0
                ledgers[i] = (
                    True if testing else SceneLedger(dict(step=0, **states(env)[i]))
                )
            if not any(running):
                break
            chunks = {}
            totals = [0.0] * n
            for i, peer in enumerate(peers):
                if not running[i]:
                    continue
                q = robot.data.joint_pos[i : i + 1, ids].detach().cpu().numpy()
                views = {
                    k: obs["policy"][k][i : i + 1].detach().cpu().numpy().copy()
                    for k in ("front", "wrist")
                }
                transport.send_observation(
                    peer,
                    query[i],
                    dict(joint_motor=convert_leisaac_action_to_lerobot(q), **views),
                )
                if ee:
                    packet = transport.receive_feedback(peer)
                    label = np.asarray(packet["ee"], dtype=float)
                    assert (
                        packet["query"] == query[i]
                        and label.shape == (7,)
                        and np.isfinite(label).all()
                    )
                    base_q, _ = adapter.command(label, q[0])
                    base_motor = convert_leisaac_action_to_lerobot(base_q[None])[0]
                    transport.send_feedback(
                        peer, dict(query=query[i], motor=base_motor.tolist())
                    )
                command = transport.receive_action(peer, 2)
                assert np.isfinite(command).all()
                if ee:
                    bounds = np.array([1, 1, 1, 1, 0.6, 1]) * np.array(
                        [ready[i]["arm_fraction"]] * 5 + [1]
                    )
                    assert (np.abs(command) <= bounds + 2e-4).all()
                    values = convert_lerobot_action_to_leisaac(
                        base_motor[None] + command
                    )
                    values[:, :5] = np.clip(values[:, :5], adapter.low, adapter.high)
                    values = np.where(command == 0, base_q[None], values)
                else:
                    values = convert_lerobot_action_to_leisaac(command)
                chunks[i] = torch.as_tensor(
                    values, device=env.device, dtype=targets.dtype
                )
            for tick in range(2):
                for i, chunk in chunks.items():
                    targets[i] = chunk[tick]
                if dynamic:
                    gripper()
                else:
                    assert torch.allclose(
                        robot.data.joint_effort_limits[:, ids[-1]],
                        torch.full_like(targets[:, 0], 10.0),
                    )
                obs, _, terms, truncs, _ = env.step((targets - offsets) / scales)
                rows = states(env)
                for i in list(chunks):
                    counters[i] += 1
                    done = bool(terms[i] or truncs[i])
                    row = final_states[i] if done else rows[i]
                    parts = (
                        {}
                        if testing
                        else ledgers[i].advance(
                            dict(step=counters[i], **row),
                            counters[i] >= limits[i].item(),
                        )
                    )
                    totals[i] += (0.99 ** (tick / 16)) * sum(parts.values())
                    if done or tick == 1:
                        feedback = dict(
                            query=query[i],
                            seed=seeds[i],
                            reward=totals[i],
                            steps=tick + 1,
                            done=done,
                            discount=0.0 if done else 0.99 ** ((tick + 1) / 16),
                        )
                        if testing and ready[i].get("check_actions") == query[i] + 1:
                            feedback["check_complete"] = True
                            ledgers[i] = None
                        if done:
                            if not testing:
                                assert bool(terms[i]) == bool(ledgers[i].native)
                            feedback.update(
                                native_success=bool(terms[i]),
                                retained=sum(row["inside"])
                                if testing
                                else sum(ledgers[i].stable),
                                scene_return=0.0
                                if testing
                                else sum(ledgers[i].components(counters[i]).values()),
                            )
                            ledgers[i] = None
                        transport.send_feedback(peers[i], feedback)
                        query[i] += 1
                        del chunks[i]
                if not chunks:
                    break


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--robot-usd", type=Path, help="Calibrated SO-101 follower USD for EE IK"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        simulate(args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
