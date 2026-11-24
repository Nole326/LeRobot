"""Deterministic PickOrange ACT residual policy."""

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .residual import mlp


def add_edges(rgb):
    red, green, blue = rgb.unbind(dim=1)
    channels = torch.stack(
        (0.299 * red + 0.587 * green + 0.114 * blue, red - 0.5 * (green + blue)), dim=1
    )
    padded = F.pad(channels, (1, 1, 1, 1), mode="replicate")
    dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
    dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
    return torch.cat((rgb - 0.5, (dx.abs() + dy.abs()).clamp(0, 1)), dim=1)


def camera_tower(channels):
    layers = []
    for _ in range(4):
        layers.extend((nn.Conv2d(channels, 32, 3, stride=2, padding=1), nn.ReLU()))
        channels = 32
    return nn.Sequential(*layers)


class ACTResidualPolicy(nn.Module):
    def __init__(self, edge_aware):
        super().__init__()
        self.edge_aware = edge_aware
        self.encoder = nn.ModuleDict(
            {
                "views": nn.ModuleList(
                    camera_tower(5 if edge_aware else 3) for _ in range(2)
                ),
                "readout": nn.Sequential(
                    nn.Flatten(), nn.Linear(5120, 128), nn.LayerNorm(128), nn.Tanh()
                ),
            }
        )
        self.actor = nn.ModuleDict({"mean": mlp(152, 6), "log_std": mlp(152, 6)})
        self.register_buffer(
            "scale",
            torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 50.0]),
            persistent=False,
        )
        self.register_buffer(
            "center", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 50.0]), persistent=False
        )
        self.register_buffer(
            "limit", torch.tensor([1.0, 1.0, 1.0, 1.0, 0.6, 1.0]), persistent=False
        )
        self.reset()

    @classmethod
    def from_safetensors(cls, path, *, edge_aware, device="cpu"):
        policy = cls(edge_aware)
        policy.load_state_dict(load_file(path), strict=True)
        return policy.to(device).eval().requires_grad_(False)

    def reset(self):
        self.previous_q = None
        self.previous_action = None

    @torch.inference_mode()
    def forward(self, front_rgb, wrist_rgb, joint_motor, base_motor):
        """Correct a single time-ensembled ACT target, returning six motor coordinates."""
        device = self.scale.device
        images = []
        for pixels in (front_rgb, wrist_rgb):
            rgb = torch.as_tensor(pixels, device=device)
            if rgb.shape != (480, 640, 3) or rgb.dtype != torch.uint8:
                raise ValueError("Each camera must supply a 480×640×3 uint8 RGB image")
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None].float(), (120, 160), mode="area"
            )
            images.append(rgb.round().clamp(0, 255).byte().float() / 255)
        views = [
            tower(add_edges(rgb) if self.edge_aware else rgb - 0.5).flatten(1)
            for tower, rgb in zip(self.encoder["views"], images)
        ]
        visual = self.encoder["readout"](torch.cat(views, dim=1))

        q = torch.as_tensor(joint_motor, device=device, dtype=torch.float32).reshape(6)
        q = (q - self.center) / self.scale
        velocity = (
            torch.zeros_like(q)
            if self.previous_q is None
            else ((q - self.previous_q) * 30 / 5).clamp(-10, 10)
        )
        previous = (
            torch.zeros_like(q)
            if self.previous_action is None
            else self.previous_action
        )
        base = torch.as_tensor(base_motor, device=device, dtype=torch.float32).reshape(
            1, 6
        )
        base = ((base - self.center) / self.scale).clamp(-1 + 1e-5, 1 - 1e-5)
        bound = self.limit / self.scale
        low, high = (base - bound).clamp(min=-1), (base + bound).clamp(max=1)
        center, scale = (low + high) / 2, (high - low) / 2
        offset = ((base - center) / scale).clamp(-1 + 1e-6, 1 - 1e-6).atanh()
        features = torch.cat(
            (visual, torch.cat((q, velocity, previous))[None], base), dim=1
        )
        action = (
            center + scale * (offset + self.actor["mean"](features)).tanh()
        ).reshape(6)
        self.previous_q, self.previous_action = q, action
        return action * self.scale + self.center
