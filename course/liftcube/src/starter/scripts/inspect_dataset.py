"""Inspect a local dataset: python -m starter.scripts.inspect_dataset --path DATASET_PATH"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starter.data.hf_adapter import HFAdapterError, inspect_lerobot_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local LeRobot v2.1/v3 dataset without downloading."
    )
    parser.add_argument("--path", type=str, required=True, help="Local LeRobot v2.1/v3 dataset directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Local path does not exist: {path}")
    report: dict[str, Any] = {"auto_download": False}
    if (path / "meta" / "info.json").exists():
        try:
            report["local_lerobot"] = inspect_lerobot_dir(path)
        except HFAdapterError as exc:
            report["local_lerobot_error"] = str(exc)
    if "local_lerobot" not in report:
        report["error"] = f"Unrecognized dataset layout at {path}"
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
