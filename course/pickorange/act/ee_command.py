"""Decode one denormalized ACT output into the absolute EE controller's units."""

import cv2
import numpy as np


def unpack(values):
    v = np.asarray(values, dtype=np.float64)
    if v.shape not in ((10,), (11,)) or not np.isfinite(v).all():
        raise ValueError("Invalid EE ACT output")
    norm = np.linalg.norm(v[3:6])
    if norm < 1e-6:
        raise ValueError("Degenerate first tool axis")
    x = v[3:6] / norm
    orthogonal = v[6:9] - x * np.dot(v[6:9], x)
    norm = np.linalg.norm(orthogonal)
    if norm < 1e-6:
        raise ValueError("Degenerate second tool axis")
    y = orthogonal / norm
    r = cv2.Rodrigues(np.column_stack((x, y, np.cross(x, y))))[0].reshape(3)
    stop = None if len(v) == 10 else float(1 / (1 + np.exp(-np.clip(v[10], -80, 80))))
    return np.r_[v[:3], r, v[9]], stop
