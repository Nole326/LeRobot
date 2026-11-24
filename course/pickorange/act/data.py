"""Read the published PickOrange archives at their original action rates."""

import json
import zlib
from pathlib import Path

import numpy as np


def rotation_columns(vectors):
    # Rodrigues' formula for the first two columns, including zero rotation.
    vectors = np.asarray(vectors, dtype=np.float64)
    angle = np.linalg.norm(vectors, axis=1)[:, None, None]
    axes = np.broadcast_to(np.eye(3)[:2], (len(vectors), 2, 3))
    cross = np.cross(vectors[:, None], axes)
    columns = axes + np.sinc(angle / np.pi) * cross
    columns += 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * np.cross(vectors[:, None], cross)
    return columns.reshape(-1, 6)


def episode_files(root, entry):
    files = entry["files"]
    if isinstance(files, list):
        paths = {Path(item["path"]).name: root / item["path"] for item in files}
    else:
        paths = {name: root / entry["path"] / name for name in files}
    required = [paths[name].resolve() for name in ("demonstration.npz", "front.mkv", "wrist.mkv")]
    if any(not path.is_relative_to(root.resolve()) for path in required):
        raise ValueError("Episode path outside dataset directory")
    return required


def decode(path, count, *, cache=True):
    import av

    frames = []
    previous = None
    n = 0
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (480, 640, 3):
                raise ValueError(f"Unexpected frame shape in {path}")
            n += 1
            if not cache:
                continue
            raw = rgb.tobytes()
            frames.append(frames[-1] if raw == previous else zlib.compress(raw, 1))
            previous = raw
    if n != count:
        raise ValueError(f"{path}: expected {count} frames, decoded {n}")
    return frames


class Episodes:
    def __init__(self, manifest, mode, limit=None):
        self.mode = mode
        self.single = mode == "ee"
        self.rows = []
        self.intervals = []
        spec = json.loads(manifest.read_text("utf-8"))
        if self.single and (spec.get("control"), spec.get("physics_hz"), spec.get("action_hz")) != (
            "absolute_ee_robot_base",
            60,
            30,
        ):
            raise ValueError("Expected 30 Hz absolute EE targets in the robot base frame")
        entries = spec["episodes"]
        if not entries or len({e["seed"] for e in entries}) != len(entries):
            raise ValueError("Expected a nonempty manifest with unique episode seeds")
        for index, entry in enumerate(entries[:limit]):
            archive, front, wrist = episode_files(manifest.parent, entry)
            with np.load(archive, allow_pickle=False) as data:
                q = data["joints"].astype(np.float32)
                action = data["action"].astype(np.float32)
                if not self.single:
                    times = data.get("timestamp")
                    if (
                        times is None
                        or times.shape != (len(q),)
                        or not np.allclose(np.diff(times), 1 / 60, rtol=1e-3, atol=1e-6)
                    ):
                        raise ValueError(f"Stride-two qpos requires 60 Hz timestamps at {archive}")
            n = entry["records"]
            if q.shape != (n, 6) or action.shape != (n, 7 if self.single else 6):
                raise ValueError(f"Invalid {mode} state or action shape at {archive}")
            if not np.isfinite(q).all() or not np.isfinite(action).all() or n < 3:
                raise ValueError(f"Invalid trajectory at {archive}")
            if self.single:
                segments = entry["segments"]
                ends = [b for _, b in segments]
                if (
                    len(ends) != 3
                    or ends[-1] != n
                    or [a for a, _ in segments] != [0, *ends[:-1]]
                    or any(a >= b for a, b in segments)
                ):
                    raise ValueError(f"Invalid segment boundaries at {archive}")
                stop = np.zeros(n, dtype=np.float32)
                for a, b in segments:
                    stop[max(a, b - 2) : b] = 1
                action = np.column_stack(
                    (action[:, :3], rotation_columns(action[:, 3:6]), action[:, 6], stop)
                ).astype(np.float32)
            else:
                segments = [(0, n)]
            self.intervals.extend((index, a, b) for a, b in segments)
            self.rows.append((q, action, (front, wrist)))

        self.views = []
        q = np.concatenate([row[0] for row in self.rows])
        a = np.concatenate([row[1] for row in self.rows])
        self.qmu, self.qstd = q.mean(0), q.std(0).clip(1e-3)
        self.amu, self.astd = a.mean(0), a.std(0).clip(1e-3)
        if self.single:
            positive = a[:, -1].sum()
            self.stop_weight = min(32.0, (len(a) - positive) / positive)
            self.amu[-1], self.astd[-1] = 0.0, 1.0

    def cache(self):
        for q, _, cameras in self.rows:
            self.views.append(tuple(decode(path, len(q)) for path in cameras))

    def choose(self, rng, count):
        samples = []
        for i in rng.integers(len(self.intervals), size=count):
            ep, start, end = self.intervals[i]
            if self.single:
                t = int(rng.integers(start, end))
            else:
                t = 2 * int(rng.integers((end - 1) // 2))
            samples.append((ep, t, end))
        return samples

    def sample(self, sample):
        ep, t, end = sample
        q, a, _ = self.rows[ep]
        offsets = np.arange(15) if self.single else 2 * np.arange(1, 16)
        targets = t + offsets
        images = [
            np.frombuffer(zlib.decompress(view[t]), dtype=np.uint8).reshape(480, 640, 3)
            for view in self.views[ep]
        ]
        return (
            (q[t] - self.qmu) / self.qstd,
            (a[targets.clip(max=end - 1)] - self.amu) / self.astd,
            targets >= end,
            *images,
        )
