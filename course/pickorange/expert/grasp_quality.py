"""Privileged grasp verification; never policy features or task success."""
import numpy as np


def check_closure(angles, command, target):
    values = np.asarray(angles, dtype=float)
    if len(values) < 12 or not np.isfinite(values).all() or not np.isfinite([command, target]).all():
        return {'verified': False, 'reason': 'insufficient_finite_samples'}
    recent = values[-12:]
    span = float(np.ptp(recent))
    verified = abs(command-target) <= 1e-4 and float(recent.max()) <= .85 and span <= .025
    return {'verified': bool(verified), 'measured_span_rad': span,
            'measured_gripper_rad': float(recent[-1]), 'command_rad': float(command),
            'target_rad': float(target), 'contact_sensor_verified': False}


def check_grasp(relative_positions, object_heights, closed_angles, initial_height):
    relative = np.asarray(relative_positions, dtype=float)
    heights = np.asarray(object_heights, dtype=float)
    angles = np.asarray(closed_angles, dtype=float)
    if (relative.ndim != 2 or relative.shape[1] != 3 or len(relative) < 12
            or heights.shape != (len(relative),) or angles.shape != (len(relative),)):
        return {'verified': False, 'reason': 'insufficient_aligned_samples'}
    if not all(np.isfinite(x).all() for x in (relative, heights, angles)) or not np.isfinite(initial_height):
        return {'verified': False, 'reason': 'non_finite'}
    relative, heights, angles = relative[-12:], heights[-12:], angles[-12:]
    drift = float(np.linalg.norm(relative-relative[-1], axis=1).max())
    lift = float(heights.min()-initial_height)
    # Contact can prevent reaching the requested angle. Requiring target match
    # would reject valid grasps; use achieved closing AND actual carried-object
    # evidence, not a motor command alone. These are diagnostic heuristics.
    closed = float(angles.max()) <= .85
    verified = closed and lift >= .03 and drift <= .025
    return {'verified': bool(verified), 'reason': 'verified' if verified else 'grasp_not_confirmed',
            'minimum_lift_m': lift, 'relative_drift_m': drift,
            'measured_gripper_rad_max': float(angles.max()),
            'gripper_closed_enough': bool(closed), 'samples': 12,
            'contact_sensor_verified': False}
