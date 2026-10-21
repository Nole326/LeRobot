"""YAML config loading and CLI helpers."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        alt = DEFAULT_CONFIG_DIR / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    data["_config_path"] = str(path.resolve())
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def add_common_args(parser: argparse.ArgumentParser, *, default_config: str) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=default_config, help="YAML config path.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:N")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log/output directory.")
    return parser


def apply_common_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.checkpoint is not None:
        cfg.setdefault("checkpoint", {})
        if isinstance(cfg["checkpoint"], dict):
            cfg["checkpoint"]["path"] = args.checkpoint
        else:
            cfg["checkpoint"] = {"path": args.checkpoint}
    if args.log_dir is not None:
        cfg.setdefault("logging", {})
        cfg["logging"]["dir"] = args.log_dir
    return cfg


def get_device(name: str | None) -> str:
    import torch

    if name is None or name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
