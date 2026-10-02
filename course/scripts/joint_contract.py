"""Frozen SO-101 mapping used by validation, never an EE controller.

Matches LeIsaac commit 24d3bcd's robot_utils.py and lerobot.py. Values named
motor are normalized calibration coordinates, NOT physical motor degrees.
"""
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
USD_LIMITS_DEG = np.array([[-110, 110], [-100, 100], [-100, 90], [-95, 95], [-160, 160], [-10, 100]], dtype=np.float32)
MOTOR_LIMITS = np.array([[-100, 100]] * 5 + [[0, 100]], dtype=np.float32)


def _check(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite ordered SO-101 six-joint values")
    return values


def rad_to_motor(values):
    values = _check(values) * (180.0 / np.pi)
    return ((values - USD_LIMITS_DEG[:, 0]) / np.diff(USD_LIMITS_DEG, axis=1).ravel()
            * np.diff(MOTOR_LIMITS, axis=1).ravel() + MOTOR_LIMITS[:, 0]).astype(np.float32)


def motor_to_rad(values):
    values = _check(values)
    return ((values - MOTOR_LIMITS[:, 0]) / np.diff(MOTOR_LIMITS, axis=1).ravel()
            * np.diff(USD_LIMITS_DEG, axis=1).ravel() + USD_LIMITS_DEG[:, 0]).astype(np.float32) * (np.pi / 180.0)
