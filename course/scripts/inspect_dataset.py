"""Read-only LeRobot metadata report. Does not import or start a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inspect(root: Path) -> dict:
    path = root / "meta" / "info.json"
    data = path.read_bytes()
    info = json.loads(data)
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("meta/info.json features must be an object")
    images = [key for key, value in features.items()
              if isinstance(value, dict) and
              (value.get("dtype") in {"image", "video"} or
               key.startswith("observation.images."))]
    notes = []
    if not images:
        notes.append("No image feature found; verify course visual-input requirement.")
    if "action" not in features:
        notes.append("Missing action feature.")
    if "observation.state" not in features:
        notes.append("No observation.state key; inspect alternative proprioception keys.")
    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        notes.append("FPS is missing or invalid.")
    notes.append("Metadata cannot establish units, actual action semantics, timestamp alignment, "
                 "privileged-state leakage, or replay success; inspect episodes separately.")
    return {
        "info_path": str(path.resolve()),
        "info_sha256": hashlib.sha256(data).hexdigest(),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": fps,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "splits": info.get("splits"),
        "features": features,
        "image_features": images,
        "notes": notes,
        "training_ready": "not determined by metadata inspection",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.dataset_root)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Metadata inspection failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
