"""Minimal behavior-cloning trainer on canonical datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from starter.interfaces.model import SmallCNNPolicy
from starter.interfaces.policy import PolicyModel


class BCTrainer:
    def __init__(self, model: PolicyModel, log_every: int = 1) -> None:
        self.model = model
        self.log_every = log_every

    def fit(
        self,
        train_loader: DataLoader,
        *,
        epochs: int,
        max_batches_per_epoch: int | None = None,
        val_loader: DataLoader | None = None,
        start_epoch: int = 0,
        on_epoch_end: Any = None,
        start_step: int = 0,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        updates = start_step
        if epochs < start_epoch:
            raise ValueError("epochs is a total budget, smaller than restored progress")
        if train_loader.num_workers != 0 or (val_loader is not None and val_loader.num_workers != 0):
            raise ValueError("Resumable BC requires num_workers=0 (worker RNG state is not serialized)")
        for epoch in range(start_epoch, epochs):
            losses = []
            for batch_idx, batch in enumerate(train_loader):
                stats = self.model.update(batch)
                updates += 1
                losses.append(float(stats["loss"]))
                if max_batches_per_epoch is not None and (batch_idx + 1) >= max_batches_per_epoch:
                    break
            row = {
                "epoch": float(epoch),
                "train_loss": float(sum(losses) / max(len(losses), 1)),
                "optimizer_steps": updates,
            }
            if val_loader is not None:
                row["val_loss"] = self.evaluate_loss(val_loader)
            history.append(row)
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, row)
            if epoch % self.log_every == 0:
                print(f"[bc] epoch={epoch} train_loss={row['train_loss']:.4f}" + (
                    f" val_loss={row['val_loss']:.4f}" if "val_loss" in row else ""
                ))
        return history

    def evaluate_loss(self, loader: DataLoader) -> float:
        model = self.model
        was_training = getattr(model, "training", False)
        if isinstance(model, torch.nn.Module):
            model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(model, SmallCNNPolicy):
                    tensors = model._as_tensors(batch)
                    pred = model.forward(tensors)
                    target = batch["action"].float().to(model.device_name)
                    low = torch.as_tensor(model.action_spec.low, device=model.device_name)
                    high = torch.as_tensor(model.action_spec.high, device=model.device_name)
                    target_norm = 2.0 * (target - low) / torch.clamp(high - low, min=1e-6) - 1.0
                    losses.append(float(torch.mean((pred - target_norm) ** 2).cpu()))
                else:
                    break
        if isinstance(model, torch.nn.Module) and was_training:
            model.train()
        return float(sum(losses) / max(len(losses), 1)) if losses else 0.0

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))
