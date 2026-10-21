"""Canonical dataset loader and episode-aware splits."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from starter.interfaces.action import ActionSpec
from starter.interfaces.observation import ObservationSpec
from starter.interfaces.vision import IdentityProcessor

from .schema import EnvManifest
from .validator import validate_arrays


class CanonicalDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        env_manifest: EnvManifest | None = None,
        *,
        indices: np.ndarray | None = None,
        validate: bool = True,
    ) -> None:
        if validate:
            validate_arrays(payload, observation_spec, action_spec)
        self.payload = payload
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.env_manifest = env_manifest
        n = int(payload["action"].shape[0])
        self.indices = np.arange(n) if indices is None else np.asarray(indices, dtype=np.int64)
        self.processor = IdentityProcessor(input_shape=observation_spec.image_shape)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        image = self.processor(self.payload["image"][i])
        proprio = torch.from_numpy(np.asarray(self.payload["proprio"][i], dtype=np.float32))
        action = torch.from_numpy(np.asarray(self.payload["action"][i], dtype=np.float32))
        return {
            "image": image,
            "proprio": proprio,
            "action": action,
            "episode_id": torch.tensor(int(self.payload["episode_id"][i])),
            "step_id": torch.tensor(int(self.payload["step_id"][i])),
            "terminated": torch.tensor(bool(self.payload["terminated"][i])),
            "truncated": torch.tensor(bool(self.payload["truncated"][i])),
        }


def episode_aware_split(
    payload: dict[str, np.ndarray],
    val_ratio: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.unique(payload["episode_id"])
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_eps = set(shuffled[:n_val].tolist())
    train_eps = set(shuffled[n_val:].tolist()) or set(shuffled.tolist())
    all_idx = np.arange(payload["episode_id"].shape[0])
    train_idx = all_idx[np.isin(payload["episode_id"], list(train_eps))]
    val_idx = all_idx[np.isin(payload["episode_id"], list(val_eps))] if val_eps else np.array([], dtype=np.int64)
    return train_idx, val_idx


def make_dataloader(
    dataset: CanonicalDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def iter_episode_ids(payload: dict[str, np.ndarray]) -> Iterator[int]:
    yield from (int(x) for x in np.unique(payload["episode_id"]))
