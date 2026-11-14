"""Nominal front-camera reset and capture without advancing the physics clock."""
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_mul, sample_uniform
import torch


def _reset_front_from_nominal(env, env_ids, asset_cfg, pose_range, convention="ros"):
    camera = env.scene[asset_cfg.name]
    robot = env.scene["robot"].data
    root_position = robot.default_root_state[env_ids, :3] + env.scene.env_origins[env_ids]
    root_orientation = robot.default_root_state[env_ids, 3:7]
    nominal_position, nominal_orientation = combine_frame_transforms(
        root_position, root_orientation,
        root_position.new_tensor(camera.cfg.offset.pos).expand_as(root_position),
        root_orientation.new_tensor(camera.cfg.offset.rot).expand_as(root_orientation),
    )
    bounds = root_position.new_tensor(
        [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
    )
    delta = sample_uniform(bounds[:, 0], bounds[:, 1], (len(env_ids), 6), device=env.device)
    orientation = quat_mul(nominal_orientation, quat_from_euler_xyz(*delta[:, 3:].unbind(1)))
    camera.set_world_poses(nominal_position + delta[:, :3], orientation, env_ids, convention)


def configure_front_reset(cfg):
    """Keep the original event order and random draws for the seeded resets."""
    front_events = [
        term for term in vars(cfg.events).values()
        if getattr(getattr(term, "params", {}).get("asset_cfg"), "name", None) == "front"
        and "pose_range" in term.params
    ]
    if len(front_events) != 1 or cfg.scene.front.offset.convention != "ros":
        raise ValueError("Expected one ROS front-camera pose reset event")
    front_events[0].func = _reset_front_from_nominal
    cfg.scene.front.update_latest_camera_pose = True
    cfg.scene.wrist.update_latest_camera_pose = True


def refresh_reset_observations(env, indices):
    """Render the reset poses and refresh both observation buffers before actions."""
    physics_step = env._sim_step_counter
    env_ids = torch.as_tensor(indices, device=env.device, dtype=torch.long)
    # Four render-only updates align Isaac Lab's tiled RGB buffer with the reset poses.
    for _ in range(4):
        env.sim.render()
    for name in ("front", "wrist"):
        camera = env.scene[name]
        camera.reset(env_ids)
        _ = camera.data
    if env._sim_step_counter != physics_step:
        raise RuntimeError("Physics advanced during reset image acquisition")
    return env.observation_manager.compute(update_history=False)
