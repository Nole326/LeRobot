"""Training-only reward wrapper; the paired evaluator remains unchanged."""

import argparse
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
from .memory_rgb import send_observation
from .rewards import geometry, feedback_reward, VARIANTS, fixed_tcp, PARAMETERS

from ..common import load_protocol, publish, rng_state, sha256, wait_file
from ..evaluate import check_imported_sources

SEEDS = (127, 128, 129, 130)
GAMMA = 0.99


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--robot-usd", type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    out = a.out
    train = a.mode == "train"
    app = env = None
    try:
        wait_file(out / "ready.json", 900, out / "model-error.json")
        ready = json.loads((out / "ready.json").read_text())
        variant = ready["reward_variant"]
        assert variant in VARIANTS
        cfg = load_protocol()
        assert (
            cfg == ready["protocol"]
            and ready["mode"] == a.mode
            and ready["gamma"] == GAMMA
        )
        (out / "simulation.claim").open("x").close()
        app = AppLauncher(a).app
        import gymnasium as gym
        import torch
        import leisaac.tasks  # noqa: F401 -- register the environment
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import (
            RecorderManagerBaseCfg,
            RecorderTerm,
        )
        from isaaclab.utils import configclass
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from leisaac.utils.robot_utils import (
            convert_leisaac_action_to_lerobot,
            convert_lerobot_action_to_leisaac,
        )
        from ..expert.joint_contract import JOINTS, motor_to_rad, rad_to_motor

        initial_z = None

        def state(e):
            term = e.cfg.terminations.success
            plate = e.scene[term.params["plate_cfg"].name].data.root_pos_w[0]
            oranges = torch.stack(
                [e.scene[o.name].data.root_pos_w[0] for o in term.params["oranges_cfg"]]
            )
            rel = oranges - plate
            bounds = rel.new_tensor(
                [
                    term.params.get("x_range", (-0.10, 0.10)),
                    term.params.get("y_range", (-0.10, 0.10)),
                    term.params.get("height_range", (-0.07, 0.07)),
                ]
            )
            inside = ((rel > bounds[:, 0]) & (rel < bounds[:, 1])).all(-1).tolist()
            frames = e.scene["ee_frame"].data
            pos, quat = (
                frames.target_pos_w[0, 0, :].tolist(),
                frames.target_quat_w[0, 0, :].tolist(),
            )
            tcp = fixed_tcp(pos, quat)
            jaw = frames.target_pos_w[0, 1, :].tolist()
            z0 = initial_z if initial_z is not None else oranges[:, 2].tolist()
            success = bool(term.func(e, **term.params)[0])
            sample = geometry(
                inside, oranges.tolist(), plate.tolist(), tcp, z0, success
            )
            sample.update(
                reference_pos_w=pos,
                reference_quat_w=quat,
                fixed_tcp=tcp,
                jaw_tcp=jaw,
                old_jaw_potential=geometry(
                    inside, oranges.tolist(), plate.tolist(), jaw, z0, success
                )["potential"],
                gripper_rad=float(e.scene["robot"].data.joint_pos[0, -1]),
                oranges=oranges.tolist(),
                plate=plate.tolist(),
                initial_z=z0,
            )
            velocities = torch.stack(
                [
                    e.scene[o.name].data.root_lin_vel_w[0]
                    for o in term.params["oranges_cfg"]
                ]
            )
            sample["privileged"] = torch.cat(
                (
                    rel.flatten() / 0.3,
                    (oranges - oranges.new_tensor(tcp)).flatten() / 0.3,
                    velocities.flatten(),
                    oranges.new_tensor(inside),
                )
            ).tolist()
            return sample

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if hasattr(self._env, "reset_buf") and bool(self._env.reset_buf.any()):
                    self._env.sac_final = state(self._env)
                    # IsaacLab computes obs_buf before record_pre_reset; copy now,
                    # never bootstrap from the automatically reset next episode.
                    self._env.sac_final_observation = pack(self._env.obs_buf)
                    self._env.sac_final_velocity = velocity()
                return None, None

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        sources = check_imported_sources()
        assert sha256(a.robot_usd) == PARAMETERS["usd_sha256"]
        ec = parse_env_cfg(cfg["task_id"], device=a.device, num_envs=1)
        ec.use_teleop_device("so101leader")
        seeds = SEEDS if train else ready["seeds"]
        ec.seed = seeds[0]
        ec.episode_length_s = cfg["episode_seconds"] if train else 120
        ec.scene.robot.spawn.rigid_props.disable_gravity = False
        ec.recorders = Recorder()
        assert ec.dynamic_reset_gripper_effort_limit is True
        env = gym.make(cfg["task_id"], cfg=ec).unwrapped
        robot = env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt - 1 / cfg["physics_hz"]) < 1e-7
        assert env.max_episode_length == ec.episode_length_s * cfg["physics_hz"]
        terms = [
            env.action_manager.get_term(k) for k in env.action_manager.active_terms
        ]
        assert sum([list(t._joint_names) for t in terms], []) == JOINTS
        offsets = torch.cat(
            [
                torch.as_tensor(t._offset, device=env.device).expand(1, t.action_dim)
                for t in terms
            ],
            1,
        )
        scales = torch.cat(
            [
                torch.as_tensor(t._scale, device=env.device).expand(1, t.action_dim)
                for t in terms
            ],
            1,
        )
        assert bool(
            torch.isfinite(offsets).all()
            & torch.isfinite(scales).all()
            & (scales != 0).all()
        )
        hold = cfg["physics_hz"] // cfg["action_hz"]
        stride = hold * cfg["execute_steps"]
        n = steps = 0
        used_seeds = []
        publish(
            out / "sim-ready.json",
            dict(
                mode=a.mode,
                sources=sources,
                seeds=used_seeds,
                hold=hold,
                stride=stride,
                native_assisted=True,
                eval_protocol_unchanged=True,
                training_reward=variant,
                reward_sampling="every_env_step_60hz",
                object_truth_to_critic_only=True,
                simulator_sha256=sha256(__file__),
            ),
        )

        def velocity():
            return robot.data.joint_vel[:, ids].detach().cpu().numpy().copy().tolist()

        def pack(obs):
            measured = robot.data.joint_pos[:, ids].detach().cpu().numpy().copy()
            q = convert_leisaac_action_to_lerobot(measured)
            np.testing.assert_allclose(q, rad_to_motor(measured), atol=1e-4)
            return dict(
                joint_motor=q,
                **{
                    cam: obs["policy"][cam].detach().cpu().numpy().copy()
                    for cam in ("front", "wrist")
                },
            )

        def request(obs, k, *, final=False):
            numeric = env.sac_final if final else state(env)
            publish(
                out / "ipc" / f"context-{k:06d}.json",
                dict(
                    query=k,
                    pre_reset=final,
                    privileged=numeric["privileged"] if train else [],
                    velocity=env.sac_final_velocity if final else velocity(),
                ),
            )
            send_observation(
                out,
                k,
                env.sac_final_observation if final else pack(obs),
                timeout=900 if k == 0 else 180,
            )

        def jobs():
            index = ready.get("start_cursor", 0)
            while True:
                path = out / f"episode-{index:06d}.json"
                wait_file(path, 900, out / "model-error.json")
                job = json.loads(path.read_text())
                if job["stop"]:
                    return
                assert job["index"] == index
                yield job
                index += 1

        for job in jobs():
            episode, seed = job["index"], job["seed"]
            used_seeds.append(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            env.sac_final = None
            term_cfg = env.cfg.terminations.success
            initial_z = [
                float(env.scene[o.name].data.root_pos_w[0, 2])
                for o in term_cfg.params["oranges_cfg"]
            ]
            prior = state(env)
            local_steps = 0
            while True:
                if shutil.disk_usage(out).free < 20 * 1024**3:
                    raise RuntimeError("Output storage reserve reached")
                started = time.perf_counter()
                request(obs, n)
                path = out / "ipc" / f"action-{n:06d}.npz"
                wait_file(path, 180 if n else 900, out / "model-error.json")
                with np.load(path, allow_pickle=False) as d:
                    motor = d["actions"].copy()
                assert (
                    motor.shape == (cfg["execute_steps"], 6)
                    and np.isfinite(motor).all()
                )
                target = convert_lerobot_action_to_leisaac(motor)
                np.testing.assert_allclose(target, motor_to_rad(motor), atol=1e-4)
                native = 0.0
                samples = [prior]
                for j in range(stride):
                    command = (
                        torch.as_tensor(target[j // hold][None], device=env.device)
                        - offsets
                    ) / scales
                    dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                    obs, reward, term, trunc, _ = env.step(command)
                    native += float(reward[0])
                    final = bool(term[0] | trunc[0])
                    samples.append(env.sac_final if final else state(env))
                    if final:
                        break
                count = j + 1
                terminal = bool(term[0] | trunc[0])
                if terminal:
                    assert env.sac_final is not None
                final_sample = samples[-1]
                placed, success = sum(final_sample["inside"]), final_sample["success"]
                trace = dict(
                    steps=count,
                    reward_samples=samples,
                    reward_dt=env.step_dt,
                    reward_variant=variant,
                    success=success,
                    terminated=bool(term[0]),
                    truncated=bool(trunc[0]),
                    placed_before=sum(prior["inside"]),
                    placed_after=placed,
                )
                components, discount = feedback_reward(trace, variant)
                reward = components["total"]
                local_steps += count
                steps += count
                if train and terminal:
                    # Final data are from the recorder; this RNG snapshot is the
                    # post-terminal boundary, before the next explicit seeded reset.
                    tmp = out / f"sim-boundary-{episode}.new"
                    torch.save(rng_state(torch, random), tmp)
                    tmp.replace(out / f"sim-boundary-{episode}.pt")
                publish(
                    out / "ipc" / f"feedback-{n:06d}.json",
                    dict(
                        seed=seed,
                        episode_index=episode,
                        phase=job["phase"],
                        query=n,
                        steps=count,
                        terminated=bool(term[0]),
                        truncated=bool(trunc[0]),
                        success=success,
                        reward=reward,
                        native_reward=native,
                        placed_before=sum(prior["inside"]),
                        placed_after=placed,
                        reward_samples=samples,
                        reward_dt=env.step_dt,
                        reward_variant=variant,
                        reward_components=components,
                        wall_seconds=time.perf_counter() - started,
                    ),
                )
                n += 1
                prior = final_sample
                publish(
                    out / "simulation-progress.json",
                    dict(seed=seed, chunks=n, steps=steps, episode_steps=local_steps),
                )
                if terminal:
                    if train:
                        request(None, n, final=True)
                        n += 1
                        env.sac_final_observation = None
                    break
            assert terminal and local_steps <= env.max_episode_length
        wait_file(out / "result.json", 600, out / "model-error.json")
        publish(
            out / "sim-result.json",
            dict(
                status="completed", mode=a.mode, steps=steps, chunks=n, seeds=used_seeds
            ),
        )
    except BaseException:
        publish(out / "sim-error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
