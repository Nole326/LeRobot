"""Inference components for the dual-camera PickOrange residual policy."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
        )

    def forward(self, x):
        return (x + self.layers(x)).relu()


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.views = nn.ModuleList()
        for _ in range(2):
            layers = []
            for channels in (3, 32, 32, 32):
                layers.extend((nn.Conv2d(channels, 32, 3, 2, 1), nn.ReLU()))
            self.views.append(nn.Sequential(*layers, ResidualBlock()))
        self.readout = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 6 * 8 * 2, 128), nn.LayerNorm(128), nn.Tanh()
        )

    def forward(self, images):
        x = images.float() / 255
        return self.readout(
            torch.cat(
                [tower(x[:, i] - 0.5).flatten(1) for i, tower in enumerate(self.views)], dim=1
            )
        )


def mlp(inputs, outputs):
    layers = []
    for _ in range(3):
        layers += [nn.Linear(inputs, 256), nn.LayerNorm(256), nn.ReLU()]
        inputs = 256
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class ResidualActor(nn.Module):
    """The deterministic action of the released bounded SAC distribution."""

    def __init__(self, features, action_dim):
        super().__init__()
        self.mean = mlp(features + action_dim, action_dim)
        # Both trained heads are present in the published checkpoint.
        self.log_std = mlp(features + action_dim, action_dim)

    def forward(self, features, base, bound):
        base = base.flatten(1).clamp(-1 + 1e-5, 1 - 1e-5)
        low, high = (base - bound).clamp(min=-1), (base + bound).clamp(max=1)
        center, scale = (low + high) / 2, (high - low) / 2
        offset = ((base - center) / scale).clamp(-1 + 1e-6, 1 - 1e-6).atanh()
        inputs = torch.cat((features, base), -1)
        return center + scale * (offset + self.mean(inputs)).tanh()


SCALE = np.array([100.0] * 5 + [50.0], np.float32)
CENTER = np.array([0.0] * 5 + [50.0], np.float32)
BOUNDS = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 1.0], np.float32)


def motor_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("Expected six finite SO-101 motor coordinates")
    return vector


def check_views(front, wrist):
    for image in (front, wrist):
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError("Expected two HWC 640 by 480 uint8 RGB images")


def checkpoint_config(directory, expected_mode):
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    mode = config.get("action_mode")
    if mode != expected_mode:
        raise ValueError(f"{path}: expected action_mode {expected_mode!r}, got {mode!r}")
    return config


def make_policy(frozen, normalization, weights, device="cpu", arm_fraction=0.5):
    """Attach the residual to a loaded ACT-1 100k policy. Returns a per-episode controller."""
    checkpoint_config(Path(weights).parent, "absolute_joint_motor")
    if arm_fraction not in (0.0, 0.25, 0.5, 1.0):
        raise ValueError("Expected an arm fraction of 0.0, 0.25, 0.5, or 1.0")
    frozen = frozen.to(device).eval().requires_grad_(False)
    if frozen.config.temporal_ensemble_coeff != 0.001:
        raise ValueError("Expected the released ACT temporal ensemble coefficient (0.001)")
    stats = {
        k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in normalization.items()
    }
    layers = nn.ModuleDict({"encoder": SpatialEncoder(), "actor": ResidualActor(146, 6)})
    layers.load_state_dict(load_file(str(weights)), strict=True)
    layers.to(device).eval().requires_grad_(False)
    encoder, actor = layers["encoder"], layers["actor"]
    image_size = (96, 128)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    bound = torch.as_tensor(BOUNDS / SCALE, device=device)[None]

    @torch.inference_mode()
    def action(obs, q, last_q, last_action):
        qn = (q - CENTER) / SCALE
        dq = np.zeros(6, np.float32) if last_q is None else ((qn - last_q) * 6).clip(-10, 10)
        numeric = torch.as_tensor(np.r_[qn, dq, last_action], device=device, dtype=torch.float32)[
            None
        ]
        q_gpu = torch.as_tensor(q, device=device, dtype=torch.float32)
        batch = {"observation.state": (q_gpu[None] - stats["qmu"]) / stats["qstd"]}
        small = []
        for cam in ("front", "wrist"):
            rgb = torch.as_tensor(obs[cam][0].copy(), device=device).permute(2, 0, 1)[None].float()
            small.append(F.interpolate(rgb, image_size, mode="area")[0].round().byte())
            batch["observation.images." + cam] = (rgb.div_(255) - mean) / std
        base = (frozen.select_action(batch) * stats["astd"] + stats["amu"]).reshape(6).cpu().numpy()
        base = torch.as_tensor(
            ((base - CENTER) / SCALE).clip(-1 + 1e-5, 1 - 1e-5), device=device, dtype=torch.float32
        )[None]
        features = torch.cat((encoder(torch.stack(small)[None]), numeric), dim=1)
        full_motor = actor(features, base, bound)[0].cpu().numpy() * SCALE + CENTER
        base_motor = base[0].cpu().numpy() * SCALE + CENTER
        motor = full_motor.copy()
        if arm_fraction != 1.0:
            motor[:5] = base_motor[:5] + arm_fraction * (full_motor[:5] - base_motor[:5])
        return motor, qn

    class Controller:
        def reset(self):
            frozen.reset()
            self.previous_q = None
            self.last_action = np.zeros(6, np.float32)

        def act(self, front_rgb, wrist_rgb, joint_motor):
            """Call once per two physics ticks, with synchronized HWC uint8 RGB."""
            check_views(front_rgb, wrist_rgb)
            q = motor_vector(joint_motor)
            obs = {"front": front_rgb[None], "wrist": wrist_rgb[None]}
            motor, self.previous_q = action(obs, q, self.previous_q, self.last_action)
            self.last_action = (motor - CENTER) / SCALE
            return motor

    controller = Controller()
    controller.reset()
    return controller


class EEResidual:
    """Apply a learned correction to the EE controller's IK joint target, in motor coordinates."""

    def __init__(self, directory, device="cpu", deployment_arm_fraction=1.0):
        directory = Path(directory)
        cfg = checkpoint_config(directory, "joint_motor_correction_after_absolute_ee_ik")
        if deployment_arm_fraction not in cfg["deployment_arm_fractions"]:
            raise ValueError("Unsupported deployment amplitude for this checkpoint")
        self.device = torch.device(device)
        self.bound = torch.as_tensor(
            np.asarray(cfg["residual_limit_motor"], np.float32) / SCALE, device=device
        )
        self.factor = np.array([deployment_arm_fraction] * 5 + [1.0], np.float32)
        self.model = nn.ModuleDict({"encoder": SpatialEncoder(), "actor": ResidualActor(146, 6)})
        self.model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.reset_scene()

    def reset_scene(self):
        self.previous_q = None
        self.last_action = np.zeros(6, np.float32)

    @torch.inference_mode()
    def correction(self, front_rgb, wrist_rgb, joint_motor, ik_motor):
        """Return six additive motor corrections after inverse kinematics."""
        check_views(front_rgb, wrist_rgb)
        views = []
        for rgb in (front_rgb, wrist_rgb):
            rgb = torch.as_tensor(rgb, device=self.device)
            views.append(
                F.interpolate(rgb.permute(2, 0, 1)[None].float(), (96, 128), mode="area")[0]
                .round()
                .byte()
            )
        q = (motor_vector(joint_motor) - CENTER) / SCALE
        dq = (
            np.zeros_like(q)
            if self.previous_q is None
            else ((q - self.previous_q) * 6).clip(-10, 10)
        )
        numeric = torch.as_tensor(np.r_[q, dq, self.last_action], device=self.device)[None]
        base = ((motor_vector(ik_motor) - CENTER) / SCALE).clip(-1 + 1e-5, 1 - 1e-5)
        target = torch.as_tensor(base, device=self.device)[None]
        features = torch.cat((self.model["encoder"](torch.stack(views)[None]), numeric), dim=1)
        action = self.model["actor"](features, target, self.bound)[0].cpu().numpy()
        delta = (action - base) * SCALE * self.factor
        self.previous_q = q
        self.last_action = (base * SCALE + CENTER + delta - CENTER) / SCALE
        return delta
