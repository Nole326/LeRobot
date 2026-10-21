"""Vision preprocessing extension point. No object-GT estimators here."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F


@runtime_checkable
class VisionProcessor(Protocol):
    """RGB uint8 HWC -> model tensor input."""

    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]  # C, H, W

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        ...


class IdentityProcessor:
    """Resize (optional) + convert HWC uint8 to CHW float32 in [0, 1]."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (64, 64, 3),
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self.input_shape = input_shape
        h, w = output_size if output_size is not None else input_shape[:2]
        self.output_size = (h, w)
        self.output_shape = (3, h, w)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        arr = np.asarray(image)
        if arr.ndim == 3:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Expected HWC or BHWC image, got {arr.shape}")
        if arr.dtype == np.uint8 or tensor.max() > 1.5:
            tensor = tensor / 255.0
        _, _, h, w = tensor.shape
        if (h, w) != self.output_size:
            tensor = F.interpolate(tensor, size=self.output_size, mode="bilinear", align_corners=False)
        if arr.ndim == 3:
            return tensor.squeeze(0)
        return tensor
