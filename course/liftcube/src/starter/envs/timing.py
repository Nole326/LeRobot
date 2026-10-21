"""Action holding without changing the simulator clock or episode horizon."""
from dataclasses import replace

import numpy as np


def policy_action_hold_steps(env, policy) -> int:
    return action_hold_steps(env.action_spec.frequency_hz, policy.action_spec.frequency_hz)


def action_hold_steps(env_hz, policy_hz) -> int:
    env_hz, policy_hz = float(env_hz), float(policy_hz)
    if not np.isfinite([env_hz, policy_hz]).all() or min(env_hz, policy_hz) <= 0:
        raise ValueError("Policy/environment frequency must be finite and positive")
    ratio = env_hz / policy_hz
    ticks = round(ratio)
    if ticks < 1 or not np.isclose(ratio, ticks, atol=1e-6, rtol=0):
        raise ValueError(f"Cannot hold {policy_hz} Hz policy at {env_hz} Hz environment ticks")
    return ticks


def decision_specs(env, frequency_hz):
    """Declare the actual decision clock; callers must execute the hold."""
    action = replace(env.action_spec, frequency_hz=float(frequency_hz))
    action_hold_steps(env.action_spec.frequency_hz, frequency_hz)
    return replace(env.observation_spec, control_hz=float(frequency_hz)), action
