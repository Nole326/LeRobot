"""Read local LeRobot v2.1 and v3 datasets for BC training.

Local modification 2026-10-20: shared bilinear RGB resize; reject unapproved
action clipping before decoding videos or constructing training labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from starter.interfaces.action import ActionSpec, JOINT_NAMES, DEFAULT_LOW, DEFAULT_HIGH, radians_to_motor
from starter.interfaces.observation import ObservationSpec
from starter.interfaces.vision import IdentityProcessor

from .schema import EnvManifest


class HFAdapterError(ValueError):
    pass


def inspect_lerobot_dir(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    info = _read_info(root)
    features = info.get("features", {})
    version = str(info.get("codebase_version", ""))
    camera_keys = [k for k in features if "image" in k]
    supported = (version == "v2.1" or version.startswith("v3.")) and "action" in features and bool(camera_keys)
    if version == "v2.1":
        fmt = "lerobot_v2"
    elif version.startswith("v3"):
        fmt = "lerobot_v3"
    else:
        fmt = version or "unknown"
    report = {
        "path": str(root),
        "format": fmt,
        "codebase_version": version,
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fields": sorted(features.keys()),
        "feature_shapes": {k: v.get("shape") if isinstance(v, dict) else None for k, v in features.items()},
        "feature_dtypes": {k: v.get("dtype") if isinstance(v, dict) else None for k, v in features.items()},
        "action_names": (features.get("action") or {}).get("names") if isinstance(features.get("action"), dict) else None,
        "camera_keys": camera_keys,
        "compatibility": "supported" if supported else "needs_conversion",
        "license_file_present": (root / "README.md").exists(),
    }
    return report


def load_lerobot_dataset(
    path: str | Path,
    *,
    camera: str = "front",
    image_height: int = 64,
    image_width: int = 64,
    include_ee: bool = True,
    max_episodes: int | None = None,
    max_frames: int | None = None,
    action_units: str | None = None,
    allow_action_clipping: bool = False,
) -> tuple[dict[str, np.ndarray], EnvManifest, ObservationSpec, ActionSpec, dict[str, Any]]:
    """Load local LeRobot v2.1 or v3.0 data into canonical BC arrays.

    v2.1 stores one parquet per episode and RGB in sibling mp4 files. Actions
    stay in LeRobot motor units. Joint velocity and EE pose are absent, so
    those proprio slots are zeros. ``last_action`` is the previous step's
    action within the episode. The final loaded frame of a full episode is
    marked terminated. This function does not download.
    """
    root = Path(path)
    info = _read_info(root)
    version = str(info.get("codebase_version", ""))
    if version.startswith("v3."):
        return _load_lerobot_v3(
            root, info, camera=camera, image_height=image_height, image_width=image_width,
            include_ee=include_ee, max_episodes=max_episodes, max_frames=max_frames,
            action_units=action_units,
            allow_action_clipping=allow_action_clipping,
        )
    if version != "v2.1":
        raise HFAdapterError(
            f"Unsupported LeRobot version {version or 'unknown'} at {root}; "
            "only v2.1 and v3.x are supported."
        )
    if action_units not in (None, "motor"):
        raise HFAdapterError("LeRobot v2.1 data expects action_units: motor")
    report = inspect_lerobot_dir(root)
    video_key = _select_video_key(info, camera)
    episodes = _episode_records(root, info)
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    chunks_size = int(info.get("chunks_size", 1000))
    image_size = (int(image_height), int(image_width))
    proprio_dim = 21 if include_ee else 18
    parts: list[dict[str, np.ndarray]] = []
    remaining = None if max_frames is None else int(max_frames)
    for episode in episodes:
        if remaining is not None and remaining <= 0:
            break
        episode_index = int(episode["episode_index"])
        table = _read_episode_parquet(root, info, episode_index, chunks_size)
        n = int(table["action"].shape[0])
        take = n if remaining is None else min(n, remaining)
        images = _decode_rgb_frames(
            _feature_path(root, info, "video_path", episode_index, chunks_size, video_key=video_key),
            take,
            image_size,
        )
        proprio = np.zeros((take, proprio_dim), dtype=np.float32)
        proprio[:, :6] = table["state"][:take]
        if take > 1:
            proprio[1:, 12:18] = table["action"][: take - 1]
        terminated = np.zeros(take, dtype=np.bool_)
        if take == n:
            terminated[-1] = True
        parts.append(
            {
                "image": images,
                "proprio": proprio,
                "action": table["action"][:take],
                "timestamp": table["timestamp"][:take],
                "episode_id": np.full(take, episode_index, dtype=np.int64),
                "step_id": table["frame_index"][:take],
                "terminated": terminated,
                "truncated": np.zeros(take, dtype=np.bool_),
                "success": np.full(take, -1, dtype=np.int64),
            }
        )
        if remaining is not None:
            remaining -= take
    if not parts:
        raise HFAdapterError(f"No frames loaded from {root}")
    payload = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    obs_spec = ObservationSpec(
        image_shape=(image_size[0], image_size[1], 3),
        camera=camera,
        proprio_dim=proprio_dim,
        include_ee_in_proprio=include_ee,
        control_hz=float(info.get("fps", 30.0)),
    )
    action_spec = ActionSpec(frequency_hz=float(info.get("fps", 30.0)))
    manifest = EnvManifest(
        env_id=str(root.name),
        backend="lerobot_v2",
        robot=str(info.get("robot_type", "so101_follower")),
        control_hz=float(info.get("fps", 30.0)),
        camera=camera,
        extra={
            "codebase_version": version,
            "video_key": video_key,
            "loaded_episodes": int(len(np.unique(payload["episode_id"]))),
            "loaded_frames": int(payload["action"].shape[0]),
        },
    )
    report = dict(report)
    report["loaded_frames"] = int(payload["action"].shape[0])
    report["loaded_episodes"] = int(len(np.unique(payload["episode_id"])))
    report["video_key"] = video_key
    return payload, manifest, obs_spec, action_spec, report


def _load_lerobot_v3(
    root: Path, info: dict[str, Any], *, camera: str, image_height: int,
    image_width: int, include_ee: bool, max_episodes: int | None,
    max_frames: int | None, action_units: str | None, allow_action_clipping: bool,
) -> tuple[dict[str, np.ndarray], EnvManifest, ObservationSpec, ActionSpec, dict[str, Any]]:
    """Join v3 episode metadata, packed parquet rows, and packed MP4 frames.

    Dataset units are not encoded in LeRobot's feature schema, so the caller
    must explicitly declare them. The LiftCube demonstrations use radians for
    arm positions and [0, 1] for gripper. Actions beyond the physical joint
    limits require explicit diagnostic approval before clipping. The source
    zero point/unit mapping must be established before using these as BC labels.
    """
    if action_units != "radians_gripper_0_1":
        raise HFAdapterError(
            "LeRobot v3 action units are not in meta/info.json; set "
            "data.action_units: radians_gripper_0_1 only after verifying the dataset."
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise HFAdapterError("pyarrow is required to read LeRobot v3 parquet files") from exc
    records_path = root / "meta" / "episodes"
    files = sorted(records_path.glob("chunk-*/*.parquet"))
    if not files:
        raise HFAdapterError(f"No v3 episode metadata under {records_path}")
    records = sorted(
        (rec for path in files for rec in pq.read_table(path).to_pylist()),
        key=lambda rec: int(rec["episode_index"]),
    )
    if max_episodes is not None:
        records = records[:int(max_episodes)]
    if not records or max_frames is not None and int(max_frames) <= 0:
        raise HFAdapterError("No episodes or frames selected")
    feature = _select_video_key(info, camera)
    names = (info.get("features", {}).get("action") or {}).get("names")
    state_names = (info.get("features", {}).get("observation.state") or {}).get("names")
    if list(names or []) != list(JOINT_NAMES) or list(state_names or []) != list(JOINT_NAMES):
        raise HFAdapterError("LeRobot v3 action/state joint names or order differ from SO-101")
    hz = float(info.get("fps", 0))
    if hz <= 0:
        raise HFAdapterError("meta/info.json must specify a positive fps")
    data_cache: dict[Path, dict[str, list]] = {}
    episode_parts: list[dict[str, np.ndarray]] = []
    selected_video: dict[Path, list[tuple[int, int]]] = {}
    remaining = None if max_frames is None else int(max_frames)
    total = 0
    for rec in records:
        if remaining is not None and remaining == 0:
            break
        episode_id = int(rec["episode_index"])
        data_path = root / str(info["data_path"]).format(
            chunk_index=int(rec["data_path.chunk_index"]),
            file_index=int(rec["data_path.file_index"]),
        )
        if data_path not in data_cache:
            if not data_path.is_file():
                raise HFAdapterError(f"Missing v3 parquet: {data_path}")
            columns = ["action", "observation.state", "timestamp",
                       "frame_index", "episode_index"]
            data_cache[data_path] = pq.read_table(data_path, columns=columns).to_pydict()
        table = data_cache[data_path]
        episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
        rows = np.flatnonzero(episode_ids == episode_id)
        length = int(rec["length"])
        if len(rows) != length or not np.array_equal(
            np.asarray(table["frame_index"], dtype=np.int64)[rows], np.arange(length)
        ) or not np.array_equal(rows, np.arange(rows[0], rows[0] + length)):
            raise HFAdapterError(f"Episode {episode_id} parquet rows do not match metadata/frames")
        take = length if remaining is None else min(length, remaining)
        rows = rows[:take]
        raw_action = np.asarray([table["action"][i] for i in rows], dtype=np.float32)
        raw_state = np.asarray([table["observation.state"][i] for i in rows], dtype=np.float32)
        if raw_action.shape != (take, 6) or raw_state.shape != (take, 6):
            raise HFAdapterError(f"Episode {episode_id} requires 6-D state and action")
        timestamps = np.asarray([table["timestamp"][i] for i in rows], dtype=np.float32)
        if not np.all(np.isfinite(raw_action)) or not np.all(np.isfinite(raw_state)):
            raise HFAdapterError(f"Episode {episode_id} contains nonfinite state/action")
        video_path = root / str(info["video_path"]).format(
            video_key=feature, chunk_index=int(rec[f"videos/{feature}/chunk_index"]),
            file_index=int(rec[f"videos/{feature}/file_index"]),
        )
        video_start = int(round(float(rec[f"videos/{feature}/from_timestamp"]) * hz))
        # v3 frame_index restarts at each episode; video frames do not.
        selected_video.setdefault(video_path, []).extend(
            (video_start + step, total + step) for step in range(take)
        )
        episode_parts.append({
            "action": raw_action, "state": raw_state, "timestamp": timestamps,
            "episode_id": np.full(take, episode_id, dtype=np.int64),
            "step_id": np.arange(take, dtype=np.int64),
            "terminated": np.array([False] * (take - 1) + [take == length]),
        })
        total += take
        if remaining is not None:
            remaining -= take
    if total == 0:
        raise HFAdapterError("No v3 frames loaded")
    actions_raw = np.concatenate([part["action"] for part in episode_parts])
    states_raw = np.concatenate([part["state"] for part in episode_parts])
    action = _lift_cube_radians_to_motor(actions_raw)
    state = _lift_cube_radians_to_motor(states_raw)
    clipped = (action < DEFAULT_LOW) | (action > DEFAULT_HIGH)
    clip_count = np.count_nonzero(clipped, axis=0).astype(int).tolist()
    if any(clip_count) and not allow_action_clipping:
        raise HFAdapterError(
            f"Action mapping exceeds SO101 limits; per-joint counts={clip_count}. "
            "Verify source units/zero points before BC; allow_action_clipping=True "
            "is for explicit diagnostics, not dataset acceptance."
        )
    action = np.clip(action, DEFAULT_LOW, DEFAULT_HIGH).astype(np.float32)
    images = np.empty((total, image_height, image_width, 3), dtype=np.uint8)
    for video_path, targets in selected_video.items():
        _decode_selected_rgb_frames(video_path, targets, images, (image_height, image_width))
    proprio = np.zeros((total, 21 if include_ee else 18), dtype=np.float32)
    proprio[:, :6] = state
    for part in episode_parts:
        ids = part["episode_id"]
        positions = np.flatnonzero(np.concatenate([p["episode_id"] for p in episode_parts]) == ids[0])
        if len(positions) > 1:
            proprio[positions[1:], 12:18] = action[positions[:-1]]
    payload = {
        "image": images, "proprio": proprio, "action": action,
        "timestamp": np.concatenate([part["timestamp"] for part in episode_parts]),
        "episode_id": np.concatenate([part["episode_id"] for part in episode_parts]),
        "step_id": np.concatenate([part["step_id"] for part in episode_parts]),
        "terminated": np.concatenate([part["terminated"] for part in episode_parts]),
        "truncated": np.zeros(total, dtype=np.bool_),
        "success": np.full(total, -1, dtype=np.int64),
    }
    spec = ObservationSpec(
        image_shape=(image_height, image_width, 3), camera=camera,
        proprio_dim=proprio.shape[1], include_ee_in_proprio=include_ee, control_hz=hz,
    )
    action_spec = ActionSpec(frequency_hz=hz)
    extra = {"codebase_version": str(info["codebase_version"]), "video_key": feature,
             "source_action_units": action_units, "clipped_action_counts": clip_count,
             "loaded_episodes": len(episode_parts), "loaded_frames": total}
    manifest = EnvManifest(
        env_id=root.name, backend="lerobot_v3", robot=str(info.get("robot_type", "so101")),
        control_hz=hz, camera=camera, extra=extra,
    )
    report = {**inspect_lerobot_dir(root), **extra}
    return payload, manifest, spec, action_spec, report


def _lift_cube_radians_to_motor(values: np.ndarray) -> np.ndarray:
    result = radians_to_motor(values)
    # LeRobot v3 LiftCube gripper is normalized [0, 1], not radians.
    result[:, 5] = values[:, 5] * 100.0
    return result.astype(np.float32)


def _decode_selected_rgb_frames(
    path: Path, targets: list[tuple[int, int]], out: np.ndarray, size: tuple[int, int],
) -> None:
    try:
        import av
    except ImportError as exc:
        raise HFAdapterError("PyAV (package 'av') is required for LeRobot v3 video") from exc
    if not path.is_file():
        raise HFAdapterError(f"Missing v3 video: {path}")
    targets = sorted(targets)
    if len({frame for frame, _ in targets}) != len(targets):
        raise HFAdapterError(f"Duplicate frame positions requested from {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames and targets[-1][0] >= stream.frames:
            raise HFAdapterError(f"Video {path} has {stream.frames} frames; need {targets[-1][0]}")
        found = 0
        for frame_number, frame in enumerate(container.decode(stream)):
            if frame_number < targets[found][0]:
                continue
            if frame_number != targets[found][0]:
                raise HFAdapterError(f"Video frame alignment failed at {path}:{frame_number}")
            out[targets[found][1]] = _as_uint8_hwc(frame.to_ndarray(format="rgb24"), size)
            found += 1
            if found == len(targets):
                break
        if found != len(targets):
            raise HFAdapterError(f"Video {path} decoded {found}/{len(targets)} requested frames")


def _read_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise HFAdapterError(
            f"{root} is not a local LeRobot dataset (missing meta/info.json). "
            "Pass a local copy; this adapter never downloads from the Hub."
        )
    return json.loads(info_path.read_text(encoding="utf-8"))


def _episode_records(root: Path, info: dict[str, Any]) -> list[dict[str, Any]]:
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        records = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return sorted(records, key=lambda row: int(row["episode_index"]))
    chunks_size = int(info.get("chunks_size", 1000))
    records = []
    for parquet in sorted((root / "data").glob("chunk-*/*.parquet")):
        stem = parquet.stem
        if not stem.startswith("episode_"):
            continue
        episode_index = int(stem.split("_")[-1])
        records.append({"episode_index": episode_index, "chunk": episode_index // chunks_size})
    if not records:
        raise HFAdapterError(f"No episode parquet files under {root / 'data'}")
    return records


def _feature_path(
    root: Path,
    info: dict[str, Any],
    path_key: str,
    episode_index: int,
    chunks_size: int,
    *,
    video_key: str | None = None,
) -> Path:
    template = info.get(path_key)
    if not template:
        raise HFAdapterError(f"meta/info.json missing {path_key}")
    rel = str(template).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key or "",
    )
    return root / rel


def _select_video_key(info: dict[str, Any], camera: str) -> str:
    features = info.get("features", {})
    preferred = f"observation.images.{camera}"
    if preferred in features:
        return preferred
    cameras = [key for key in features if key.startswith("observation.images.")]
    raise HFAdapterError(f"Camera {camera!r} not in dataset. Available: {cameras}")


def _read_episode_parquet(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    chunks_size: int,
) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise HFAdapterError("pyarrow is required to read LeRobot v2.1 parquet episodes.") from exc
    path = _feature_path(root, info, "data_path", episode_index, chunks_size)
    if not path.exists():
        raise HFAdapterError(f"Missing episode parquet: {path}")
    columns = ["action", "observation.state", "timestamp", "frame_index", "episode_index"]
    table = pq.read_table(path, columns=columns).to_pydict()
    action = np.asarray(table["action"], dtype=np.float32)
    state = np.asarray(table["observation.state"], dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 6 or state.shape != action.shape:
        raise HFAdapterError(f"{path} action/state shape {action.shape} / {state.shape}, expected (N, 6)")
    names = (info.get("features", {}).get("action") or {}).get("names")
    if names is not None and list(names) != list(JOINT_NAMES):
        raise HFAdapterError(f"Action names {names} != {list(JOINT_NAMES)}")
    return {
        "action": action,
        "state": state,
        "timestamp": np.asarray(table["timestamp"], dtype=np.float32),
        "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
    }


def _decode_rgb_frames(path: Path, n_frames: int, size: tuple[int, int]) -> np.ndarray:
    if n_frames <= 0:
        raise HFAdapterError(f"Refusing to decode {n_frames} frames from {path}")
    if not path.exists():
        raise HFAdapterError(f"Missing episode video: {path}")
    try:
        import av
    except ImportError as exc:
        raise HFAdapterError("PyAV (package 'av') is required to decode LeRobot v2.1 videos.") from exc
    height, width = size
    out = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    container = av.open(str(path))
    written = 0
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            out[written] = _as_uint8_hwc(frame.to_ndarray(format="rgb24"), size)
            written += 1
            if written >= n_frames:
                break
    finally:
        container.close()
    if written < n_frames:
        raise HFAdapterError(f"{path} decoded {written} frames, parquet expects {n_frames}")
    return out


def _as_uint8_hwc(image: Any, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.01:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = size
    if arr.shape[0] != h or arr.shape[1] != w:
        tensor = IdentityProcessor(output_size=(h, w))(arr)
        arr = tensor.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    if arr.shape[-1] != 3:
        raise HFAdapterError(f"image channels {arr.shape}")
    return arr
