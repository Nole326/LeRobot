"""Adapter for LeIsaac LiftCube. No object GT in policy observations.

Local course adaptation (2026-10-20): camera/readiness, SO101 padding,
actual control frequency, causal command history and pre-reset terminal capture.
2026-10-22: reject missing feedback, require explicit alias units, and expose
requested/sent/measured joint values separately for interface diagnostics.
Original archive retained unchanged.

Created inside the course container as ``ManagerBasedRLEnv(LiftCubeEnvCfg)``.
LeIsaac v0.1.2 registers this task as ``LeIsaac-SO101-LiftCube-v0``.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from starter.interfaces.action import (
    ActionSpec,
    clip_action,
    motor_to_radians,
    radians_to_motor,
)
from starter.interfaces.observation import Observation, ObservationSpec, pack_proprio, proprio_dim
from starter.envs.lift_cube_privileged import (
    CUBE_NAME,
    ROBOT_BASE_NAME,
    ROBOT_NAME,
    LiftCubePrivilegedState,
    LiftCubePrivilegedStateSpec,
)
from starter.interfaces.vision import IdentityProcessor

TASK_ID = "LeIsaac-SO101-LiftCube-v0"
SPARSE_SUCCESS_REWARD = 1.0
LEISAAC_SOURCE = "/leisaac/source/leisaac"
_SIM_APP: Any | None = None


class IsaacUnavailableError(ImportError):
    """Raised when Isaac Sim / LeIsaac / EnvHub cannot be imported."""


def describe_install() -> str:
    return (
        "LiftCube is created with Isaac Lab and LeIsaac, either in the course image "
        "isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src or in a local interpreter that can import "
        "isaaclab and leisaac. The task is ManagerBasedRLEnv(LiftCubeEnvCfg) "
        "(LeIsaac-SO101-LiftCube-v0); there is no separate LiftCube-Direct task. "
        "From a host Python without those packages, `python -m starter.scripts.smoke_test` "
        "re-execs into the course image. "
        "The image has no lerobot package, so the env is not created with lerobot.envs.make_env. "
        "Inside that image, Python must see /leisaac/source/leisaac before the repo root, "
        "or `leisaac.assets` resolves to the USD folder instead of the Python package."
    )


def try_import_leisaac() -> tuple[bool, str]:
    try:
        import isaaclab  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional
        return False, f"isaaclab import failed: {exc}"
    try:
        import leisaac  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"leisaac import failed: {exc}"
    return True, "ok"


def _launch_sim_app(device: str) -> Any:
    """Start Isaac before any isaaclab.envs import. Headless, with cameras."""
    global _SIM_APP
    if _SIM_APP is not None:
        return _SIM_APP
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args, _unknown = parser.parse_known_args()
    app_args.headless = True
    app_args.enable_cameras = True
    app_args.device = device if str(device).startswith("cuda") else "cuda:0"
    _SIM_APP = AppLauncher(app_args).app
    return _SIM_APP


def _import_lift_cube():
    """Use the image's LeIsaac source only inside the course image."""
    from starter.envs.container import in_course_container

    if in_course_container():
        if LEISAAC_SOURCE not in sys.path:
            sys.path.insert(0, LEISAAC_SOURCE)
        # The image also exposes a USD directory named leisaac. Reload the
        # Python package from its source directory after correcting sys.path.
        for name in list(sys.modules):
            if name == "leisaac" or name.startswith("leisaac."):
                del sys.modules[name]
    from isaaclab.envs import ManagerBasedRLEnv
    from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg

    return LiftCubeEnvCfg, ManagerBasedRLEnv


def shutdown_sim_app() -> None:
    """Close Kit. Isaac's app.close() ends the process, so call this last."""
    global _SIM_APP
    app = _SIM_APP
    _SIM_APP = None
    if app is None:
        return
    app.close()


class LiftCubeEnvAdapter:
    """Wrap a live LeIsaac / EnvHub env into the starter Gym-like schema.

    Conversion of batch dim, dtype, image layout, action clip, and
    terminated/truncated is centralized here so BC/RL/eval never parse
    Isaac dicts directly.
    """

    name = TASK_ID

    def __init__(
        self,
        raw_env: Any | None = None,
        *,
        camera: str = "front",
        image_height: int = 64,
        image_width: int = 64,
        include_ee_in_proprio: bool = True,
        control_hz: float = 30.0,
        n_envs: int = 1,
        device: str = "cuda:0",
        max_episode_steps: int = 640,
        raw_joint_units: str | None = None,
    ) -> None:
        if n_envs != 1:
            raise ValueError("This scalar adapter supports n_envs=1 only.")
        self.camera = camera
        self.include_ee = include_ee_in_proprio
        self.device = device
        self.n_envs = n_envs
        self.max_episode_steps = int(max_episode_steps)
        self._owns_env = raw_env is None
        if raw_joint_units not in (None, "radians", "motor"):
            raise ValueError("raw_joint_units must be radians or motor")
        if self._owns_env and raw_joint_units == "motor":
            raise ValueError("The course Isaac environment emits joint_pos in radians")
        self.raw_joint_units = raw_joint_units
        self._vector_env = None
        self._app = None
        self._raw = raw_env if raw_env is not None else self._make_raw_env()
        self._unwrapped = self._unwrap(self._raw)
        step_dt = getattr(self._unwrapped, "step_dt", None)
        if step_dt is not None:
            if not np.isfinite(step_dt) or float(step_dt) <= 0:
                raise ValueError("Environment step_dt must be finite and positive")
            control_hz = 1.0 / float(step_dt)
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("Environment frequency must be finite and positive")
        self.action_spec = ActionSpec(frequency_hz=control_hz)
        self.observation_spec = ObservationSpec(
            image_shape=(image_height, image_width, 3),
            camera=camera,
            proprio_dim=proprio_dim(include_ee_in_proprio),
            include_ee_in_proprio=include_ee_in_proprio,
            control_hz=control_hz,
        )
        self.privileged_spec = LiftCubePrivilegedStateSpec()
        self._resize = IdentityProcessor(
            input_shape=(image_height, image_width, 3),
            output_size=(image_height, image_width),
        )
        self._episode_id = -1
        self._step_count = 0
        self._last_action = np.zeros(self.action_spec.dim, dtype=np.float32)
        self._terminal_obs = None
        self._terminal_privileged = None

    def _make_raw_env(self) -> Any:
        from starter.envs.container import isaac_runtime_ready

        if not isaac_runtime_ready():
            raise IsaacUnavailableError(describe_install())
        try:
            self._app = _launch_sim_app(self.device)
            env_cls, manager_cls = _import_lift_cube()
            env_cfg = env_cls()
            # LeIsaac enables a demonstration recorder by default. It appends
            # full-resolution RGB on the GPU with growing torch.cat buffers.
            # This adapter uses explicit external recording, like upstream
            # policy_inference.py; physics, sensors and scoring stay unchanged.
            env_cfg.recorders = None
            env_cfg.scene.num_envs = self.n_envs
            env_cfg.sim.device = self.device if str(self.device).startswith("cuda") else "cuda:0"
            # so101leader is absolute joint position. keyboard is a relative delta and
            # cannot execute the dataset's absolute motor targets.
            env_cfg.use_teleop_device("so101leader")
            env_cfg.actions.arm_action.use_default_offset = False
            env_cfg.actions.gripper_action.use_default_offset = False
            env_cfg.actions.arm_action.preserve_order = True
            env_cfg.actions.gripper_action.preserve_order = True
            dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
            env_cfg.episode_length_s = float(self.max_episode_steps) * dt
            class TerminalCaptureEnv(manager_cls):
                def _reset_idx(raw, env_ids):
                    callback = getattr(raw, "_course_before_reset", None)
                    if callback is not None:
                        callback()
                    super()._reset_idx(env_ids)

            return TerminalCaptureEnv(cfg=env_cfg)
        except IsaacUnavailableError:
            raise
        except Exception as exc:
            raise IsaacUnavailableError(f"LiftCube creation failed: {exc}\n{describe_install()}") from exc

    def _unwrap(self, env: Any) -> Any:
        cur = env
        for attr in ("envs", "unwrapped"):
            if attr == "envs" and hasattr(cur, "envs"):
                try:
                    cur = cur.envs[0]
                except Exception:
                    pass
            if hasattr(cur, "unwrapped"):
                try:
                    cur = cur.unwrapped
                except Exception:
                    pass
        return cur

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        self._episode_id += 1
        self._step_count = 0
        self._last_action = np.zeros(self.action_spec.dim, dtype=np.float32)
        self._terminal_obs = None
        self._terminal_privileged = None
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        try:
            result = self._raw.reset(**kwargs)
        except TypeError:
            result = self._raw.reset()
        obs_raw, info_raw = self._split_reset(result)
        if self._owns_env:
            from starter.envs.render_ready import prepare_front_rgb

            obs_raw, self.render_readiness = prepare_front_rgb(self._unwrapped, obs_raw, self.camera)
        obs = self._convert_obs(obs_raw)
        info = self._public_info(success=False, reason="", reward=0.0)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape not in ((6,), (1, 6)):
            raise ValueError(f"One SO101 command requires six ordered targets, got {requested.shape}")
        requested = requested.reshape(6).copy()
        action = clip_action(requested, self.action_spec)
        isaac_action = self._to_isaac_action(action)
        self._terminal_obs = None
        self._terminal_privileged = None
        self._last_action = action.copy()
        if self._owns_env:
            self._unwrapped._course_before_reset = self._capture_terminal
        try:
            result = self._raw.step(isaac_action)
        finally:
            if self._owns_env:
                self._unwrapped._course_before_reset = None
        obs_raw, _reward, terminated, truncated, info_raw = self._split_step(result)
        obs = self._terminal_obs if self._terminal_obs is not None else self._convert_obs(obs_raw)
        terminated_b = bool(self._first_bool(terminated))
        truncated_b = bool(self._first_bool(truncated))
        success = self._read_success()
        used_reward = SPARSE_SUCCESS_REWARD if success else 0.0
        reason = self._infer_reason(success, terminated_b, truncated_b, info_raw)
        self._step_count += 1
        info = self._public_info(success=success, reason=reason, reward=used_reward)
        info["command"] = {
            "requested_motor": requested.tolist(),
            "applied_motor": action.tolist(),
            "applied_radians": motor_to_radians(action).tolist(),
            "clipped": (requested != action).tolist(),
            "measured_motor": obs.proprio[:6].tolist(),
        }
        return obs, used_reward, terminated_b, truncated_b, info

    def _capture_terminal(self) -> None:
        """Capture before Isaac's automatic reset; never expose GT to the policy."""
        raw = self._unwrapped.observation_manager.compute(update_history=False)
        self._terminal_obs = self._convert_obs(raw)
        self._terminal_obs.image = self._terminal_obs.image.copy()
        self._terminal_privileged = self.get_privileged_state()

    def close(self) -> None:
        for obj in (self._raw, self._vector_env, self._unwrapped):
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
        self._app = None

    def _split_reset(self, result: Any) -> tuple[Any, dict]:
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1] if isinstance(result[1], dict) else {}
        return result, {}

    def _split_step(self, result: Any) -> tuple[Any, Any, Any, Any, dict]:
        if not isinstance(result, tuple):
            raise RuntimeError(f"Unexpected step return: {type(result)}")
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info if isinstance(info, dict) else {}
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info if isinstance(info, dict) else {}
        raise RuntimeError(f"Unexpected step tuple length {len(result)}")

    def _convert_obs(self, obs_raw: Any) -> Observation:
        policy = self._policy_dict(obs_raw)
        image = self._extract_image(policy)
        key = "joint_pos" if "joint_pos" in policy else "observation.state"
        joint_pos = self._extract_vec(policy, (key,), default=6)
        units = self.raw_joint_units
        if units is None:
            if key != "joint_pos":
                raise ValueError("observation.state units are ambiguous; declare raw_joint_units")
            units = "radians"  # Known Isaac joint_pos contract, not a generic dataset inference.
        if units == "radians":
            joint_pos = radians_to_motor(joint_pos)
        last_action = self._last_action
        # Match the TA-confirmed SO101 data/real-robot contract. Simulator-only
        # measurements must not become inputs that disappear on the real robot.
        joint_vel = np.zeros(6, dtype=np.float32)
        ee = np.zeros(3, dtype=np.float32)
        proprio = pack_proprio(joint_pos, joint_vel, last_action, ee, include_ee=self.include_ee)
        return Observation(
            image=image,
            proprio=proprio,
            metadata={
                "camera": self.camera,
                "env": self.name,
                "episode_id": self._episode_id,
            },
        )

    def _policy_dict(self, obs_raw: Any) -> dict[str, Any]:
        if not isinstance(obs_raw, dict):
            raise ValueError("Isaac observation is not a dict; cannot convert safely.")
        if "policy" in obs_raw and isinstance(obs_raw["policy"], dict):
            policy = dict(obs_raw["policy"])
        else:
            policy = dict(obs_raw)
        policy.pop("subtask_terms", None)
        return policy

    def _extract_image(self, policy: dict[str, Any]) -> np.ndarray:
        candidates = [self.camera]
        raw = None
        for key in candidates:
            if key in policy:
                raw = policy[key]
                break
            nested = f"observation.images.{key}"
            if nested in policy:
                raw = policy[nested]
                break
        if raw is None:
            raise ValueError(f"Required camera {self.camera!r} is absent.")
        img = self._to_numpy(raw)
        img = np.asarray(img)
        if img.ndim == 4:
            if img.shape[0] != 1:
                raise ValueError("The scalar adapter requires one camera frame, not a batch")
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim != 3 or img.shape[-1] not in (3, 4) or 0 in img.shape:
            raise ValueError(f"Expected one RGB/RGBA camera frame, got {img.shape}")
        if not np.isfinite(img).all() or img.min() < 0 or img.max() > 255:
            raise ValueError("Camera pixels must be finite and inside [0,255]")
        if img.dtype != np.uint8:
            if img.max() <= 1.01:
                img = np.clip(img, 0, 1) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        target_h, target_w, _ = self.observation_spec.image_shape
        if img.shape[0] != target_h or img.shape[1] != target_w:
            tensor = self._resize(img)
            img = (
                tensor.permute(1, 2, 0).clamp(0, 1).mul(255.0).byte().cpu().numpy()
                if tensor.dim() == 3
                else tensor[0].permute(1, 2, 0).clamp(0, 1).mul(255.0).byte().cpu().numpy()
            )
        return np.ascontiguousarray(img)

    def _extract_vec(
        self,
        policy: dict[str, Any],
        keys: tuple[str, ...],
        default: int,
        *,
        required: bool = True,
    ) -> np.ndarray | None:
        for key in keys:
            if key in policy:
                arr = np.asarray(self._to_numpy(policy[key]), dtype=np.float32)
                if arr.shape not in ((default,), (1, default)) or not np.isfinite(arr).all():
                    raise ValueError(f"{key} requires exactly {default} finite joint positions")
                return arr.reshape(default).copy()
        if not required:
            return None
        raise ValueError(f"Required joint feedback is missing: {keys}")

    def _extract_ee(self, policy: dict[str, Any]) -> np.ndarray:
        if "ee_frame_state" in policy:
            arr = np.asarray(self._to_numpy(policy["ee_frame_state"]), dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
        return np.zeros(3, dtype=np.float32)

    def _to_isaac_action(self, action: np.ndarray) -> Any:
        import torch

        arr = motor_to_radians(np.asarray(action, dtype=np.float32)).reshape(1, -1)
        device = getattr(self._unwrapped, "device", self.device)
        try:
            return torch.as_tensor(arr, device=device)
        except Exception:
            return arr

    def get_privileged_state(self) -> LiftCubePrivilegedState:
        """LiftCube cube, base, and jaw state for reward code. Not a policy observation."""
        if self._terminal_privileged is not None:
            return self._terminal_privileged
        scene = getattr(self._unwrapped, "scene", None)
        if scene is None:
            raise RuntimeError("LiftCube scene is not available")
        keys = set(scene.keys()) if hasattr(scene, "keys") else set()
        missing = [name for name in (CUBE_NAME, ROBOT_NAME, "ee_frame") if name not in keys]
        if missing:
            raise KeyError(f"Privileged entities {missing} not in scene. Keys: {sorted(keys)}")
        origin = np.asarray(self._to_numpy(scene.env_origins[0, :3]), dtype=np.float32).reshape(3)
        cube_pos = self._root_pos_env(scene[CUBE_NAME], origin)
        robot = scene[ROBOT_NAME]
        body_names = [str(name) for name in list(robot.data.body_names)]
        base_index = body_names.index(ROBOT_BASE_NAME)
        base = np.asarray(self._to_numpy(robot.data.body_pos_w[0, base_index, :3]), dtype=np.float32).reshape(3)
        robot_base_pos = base - origin
        jaw = np.asarray(self._to_numpy(scene["ee_frame"].data.target_pos_w[0, 1, :3]), dtype=np.float32)
        ee_jaw_pos = jaw - origin
        joint_names = [str(name) for name in list(robot.data.joint_names)]
        gripper_joint_rad = float(self._to_numpy(robot.data.joint_pos[0, joint_names.index("gripper")]))
        state = LiftCubePrivilegedState(
            cube_pos=cube_pos,
            robot_base_pos=robot_base_pos,
            ee_jaw_pos=ee_jaw_pos,
            gripper_joint_rad=gripper_joint_rad,
        )
        state.validate(self.privileged_spec)
        return state

    def _root_pos_env(self, asset: Any, origin: np.ndarray) -> np.ndarray:
        pos = np.asarray(self._to_numpy(asset.data.root_pos_w[0, :3]), dtype=np.float32).reshape(3)
        return pos - origin

    def _read_success(self) -> bool:
        manager = getattr(self._unwrapped, "termination_manager", None)
        if manager is None or not hasattr(manager, "get_term"):
            return False
        try:
            return bool(self._first_bool(manager.get_term("success")))
        except KeyError:
            return False

    def _infer_reason(self, success: bool, terminated: bool, truncated: bool, info_raw: dict) -> str:
        if success:
            return "success"
        if truncated:
            return "timeout"
        if terminated:
            return str(info_raw.get("termination_reason", "failed"))
        return ""

    def _public_info(self, success: bool, reason: str, reward: float) -> dict[str, Any]:
        return {
            "success": bool(success),
            "termination_reason": reason,
            "episode_id": self._episode_id,
            "step_id": self._step_count,
            "env_name": self.name,
            "reward_total": float(reward),
        }

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _first_bool(value: Any) -> bool:
        arr = LiftCubeEnvAdapter._to_numpy(value).reshape(-1)
        if arr.size == 0:
            return bool(value)
        return bool(arr[0])

    @staticmethod
    def _first_float(value: Any) -> float:
        arr = LiftCubeEnvAdapter._to_numpy(value).reshape(-1)
        if arr.size == 0:
            return float(value)
        return float(arr[0])
