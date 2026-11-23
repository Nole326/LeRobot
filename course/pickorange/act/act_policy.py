"""Load published ACT checkpoints and prepare their dual-view observations."""

import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file


class ACTInputs:
    def __init__(self, checkpoint, device="cuda"):
        folder = Path(checkpoint)
        control = json.loads((folder / "execution.json").read_text())
        normalization = json.loads((folder / "normalization.json").read_text())
        self.control = control
        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(folder)
        config.pretrained_backbone_weights = None
        self.model = ACTPolicy(config).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        order = ["front", "wrist"]
        if control["image_order"] != order or list(config.image_features) != [
            "observation.images." + name for name in order
        ]:
            raise ValueError("Expected front then wrist RGB camera features")
        if (
            control["physics_hz"],
            control["replan_physics_ticks"],
            control["chunk"],
            control["ensemble"],
        ) != (60, 2, config.chunk_size, config.temporal_ensemble_coeff):
            raise ValueError("Incompatible ACT execution cadence or chunk configuration")
        self.stats = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in normalization.items()
        }
        action_dim = config.output_features["action"].shape[0]
        for name, size in (("qmu", 6), ("qstd", 6), ("amu", action_dim), ("astd", action_dim)):
            value = self.stats[name]
            if (
                value.shape != (size,)
                or not value.isfinite().all()
                or (name.endswith("std") and (value <= 0).any())
            ):
                raise ValueError(f"Invalid checkpoint normalization: {name}")
        self.mean = torch.tensor(control["rgb_mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(control["rgb_std"], device=self.device).view(1, 3, 1, 1)

    def observation(self, front, wrist, joint_motor):
        q = np.asarray(joint_motor, dtype=np.float32)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six measured SO-101 motor positions")
        state = torch.as_tensor(q, device=self.device).view(1, 6)
        batch = {"observation.state": (state - self.stats["qmu"]) / self.stats["qstd"]}
        for name, image in (("front", front), ("wrist", wrist)):
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f"Invalid {name} RGB observation")
            pixels = torch.from_numpy(np.array(image, copy=True)).to(self.device)
            rgb = pixels.permute(2, 0, 1).unsqueeze(0).float().div_(255)
            batch["observation.images." + name] = (rgb - self.mean) / self.std
        return batch


class QposPolicy(ACTInputs):
    """Predict absolute joint targets in calibrated motor units at 30 Hz."""

    def __init__(self, checkpoint, device="cuda"):
        super().__init__(checkpoint, device)
        if self.control["mode"] != "stride2" or self.control["action_hz"] != 30:
            raise ValueError("Expected a stride-two qpos checkpoint")
        self.reset_scene()

    def reset_scene(self):
        self.model.reset()

    @torch.inference_mode()
    def predict(self, front, wrist, joint_motor):
        batch = self.observation(front, wrist, joint_motor)
        target = self.model.select_action(batch)[0]
        return (target * self.stats["astd"] + self.stats["amu"]).cpu().numpy()
