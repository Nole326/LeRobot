"""Trusted local training checkpoints at epoch / completed-rollout boundaries.

Inference exports remain separate. No simulator snapshot or DataLoader worker
state is claimed; resumable BC uses num_workers=0 and PPO starts seeded episodes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    if state["cuda"] is not None and len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG device count differs from checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def training_identity(kind, config, payload=None):
    """Freeze effective options and source; permit only budget/output changes.

    BC hashes the already loaded canonical arrays once, not source videos again.
    No private generators or persistent worker pools are used by these trainers.
    """
    options = json.loads(json.dumps(config, default=str))
    for key in ("_config_path", "logging", "checkpoint", "device"):
        options.pop(key, None)
    section = "train" if kind == "bc" else "ppo"
    for key in ("epochs", "num_updates", "log_every", "bc_checkpoint_path"):
        options.get(section, {}).pop(key, None)
    source = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*.py"))}
    data = {}
    if payload is not None:
        for name, array in sorted(payload.items()):
            array = np.ascontiguousarray(array)
            data[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                          "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    return {"kind": kind, "options": options, "source": code, "data": data}


def _atomic_bytes(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training(path, model, identity, progress):
    payload = {"format": "liftcube-training-v1", "policy": model.checkpoint_payload(),
               "optimizer": model.opt.state_dict(), "identity": identity,
               "progress": progress, "rng": capture_rng()}
    _atomic_bytes(path, lambda stream: torch.save(payload, stream))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    _atomic_bytes(str(path) + ".sha256", lambda stream: stream.write((digest + "\n").encode()))


def load_training(path, model, identity):
    path = Path(path)
    digest = Path(str(path) + ".sha256").read_text().strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Training checkpoint SHA256 mismatch")
    # Local checkpoints contain Python/NumPy RNG objects: never load untrusted files.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "liftcube-training-v1" or saved["identity"] != identity:
        raise ValueError("Training identity differs; use weights-only initialization for a new task")
    policy = saved["policy"]
    for key in ("observation_spec", "action_spec", "hidden_dim", "feature_mode"):
        if policy[key] != model.checkpoint_payload()[key]:
            raise ValueError(f"Policy contract differs: {key}")
    model.load_state_dict(policy["state_dict"], strict=True)
    model.opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    return saved["progress"]


def training_path(config):
    export = config.get("checkpoint", {}).get("path", "outputs/checkpoint.pt")
    return config.get("checkpoint", {}).get("training_path", str(export) + ".training.pt")
