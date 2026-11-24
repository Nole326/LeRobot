"""PickOrange reward ablation. Numeric inputs only; no simulator imports.

Inspired by IRMV open_laptop hybrid shaping; new task-specific implementation.
Success and the placement mask must come from the unchanged native evaluator.
"""

import math

VERSION = "pickorange-step-cost"
VARIANTS = ("count", "hybrid")
PARAMETERS = dict(
    arrival_by_count=[2.0, 3.0, 4.0],
    departure=5.0,
    completion=10.0,
    progress=3.0,
    quality_per_second=0.03,
    step_penalty=0.02 / 60,
    hybrid_departure=6.0,
    reference_offset_m=[
        0.04239292443776763,
        -0.00119974788459086,
        -0.09303105684230396,
    ],
    reference_jaw_rad=0.6,
    usd_sha256="64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58",
    reach_scale_m=0.10,
    lift_scale_m=0.04,
    transport_scale_m=0.20,
)


def geometry(inside, oranges, plate, tcp, initial_z, success):
    """Bounded progress; proximity/lift are geometric proxies, not grasp proof."""
    if not (len(inside) == len(oranges) == len(initial_z) == 3):
        raise ValueError("Exactly three consistently ordered oranges required")
    points = [*oranges, plate, tcp]
    if any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in points):
        raise ValueError("Expected finite xyz coordinates in metres")
    if any(not math.isfinite(z) for z in initial_z):
        raise ValueError("Invalid reset heights")
    candidates = []
    for i, orange in enumerate(oranges):
        if inside[i]:
            continue
        reach = 1 - math.tanh(math.dist(tcp, orange) / PARAMETERS["reach_scale_m"])
        lift = min(
            1.0, max(0.0, (orange[2] - initial_z[i]) / PARAMETERS["lift_scale_m"])
        )
        carry = 1 - math.tanh(
            math.dist(orange, plate) / PARAMETERS["transport_scale_m"]
        )
        candidates.append(0.2 * reach + 0.3 * reach * lift + 0.4 * reach * lift * carry)
    sample = dict(
        inside=list(inside),
        success=success,
        potential=(sum(inside) + max(candidates, default=0.0)) / 3.0,
    )
    validate(sample)
    return sample


def validate(sample):
    mask = sample["inside"]
    if len(mask) != 3 or any(type(v) is not bool for v in mask):
        raise ValueError("Placement mask must contain three booleans")
    if type(sample["success"]) is not bool or (sample["success"] and not all(mask)):
        raise ValueError("Native success requires all three oranges in plate")
    p = sample["potential"]
    if isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid potential")


def transition(before, after, dt, variant):
    if variant not in VARIANTS or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid reward variant or simulation duration")
    validate(before)
    validate(after)
    parts = dict(
        arrival=0.0,
        departure=0.0,
        completion=0.0,
        progress=0.0,
        quality=0.0,
        step=0.0,
        timeout=0.0,
    )
    if not before["success"]:
        previous, current = sum(before["inside"]), sum(after["inside"])
        parts["arrival"] = (
            sum(PARAMETERS["arrival_by_count"][previous:current])
            if current > previous
            else 0.0
        )
        parts["departure"] = -PARAMETERS["departure"] * max(previous - current, 0)
        parts["completion"] = PARAMETERS["completion"] * int(after["success"])
        if variant == "hybrid":
            # Object identity catches simultaneous swaps even when count is unchanged.
            losses = sum(
                was and not now for was, now in zip(before["inside"], after["inside"])
            )
            parts["departure"] = -PARAMETERS["hybrid_departure"] * losses
            # Progress is settled once per SAC transition in feedback_reward.
            # Summing un-discounted per-tick deltas would use the wrong time scale.
            # Trapezoidal, time-scaled nonpositive quality; standing still cannot farm reward.
            parts["quality"] = (
                PARAMETERS["quality_per_second"]
                * ((before["potential"] + after["potential"]) / 2 - 1)
                * dt
            )
            parts["step"] = -PARAMETERS["step_penalty"]
    parts["total"] = sum(parts.values())
    return parts


def feedback_reward(feedback, variant):
    """Recompute the chunk from per-physics-step numeric samples, never RGB."""
    steps = feedback["steps"]
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError("Expected 1..32 physical steps")
    samples = feedback["reward_samples"]
    if len(samples) != steps + 1 or feedback["reward_variant"] != variant:
        raise ValueError("Reward trace identity or length mismatch")
    if any(
        type(feedback[k]) is not bool for k in ("success", "terminated", "truncated")
    ):
        raise ValueError("Invalid terminal flags")
    if feedback["success"] != samples[-1]["success"] or (
        feedback["success"] and not feedback["terminated"]
    ):
        raise ValueError("Native success/termination mismatch")
    if any(s["success"] for s in samples[:-1]):
        raise ValueError("Transition after native success")
    if (feedback["placed_before"], feedback["placed_after"]) != (
        sum(samples[0]["inside"]),
        sum(samples[-1]["inside"]),
    ):
        raise ValueError("Placement count mismatch")
    if abs(feedback["reward_dt"] - 1 / 60) > 1e-10:
        raise ValueError("This entry uses the frozen 60 Hz protocol")
    parts = [
        transition(a, b, feedback["reward_dt"], variant)
        for a, b in zip(samples, samples[1:])
    ]
    totals = {key: sum(p[key] for p in parts) for key in parts[0]}
    # Preserve the learner's finite 60 s episode and action-chunk discount convention.
    discount = 0.0 if feedback["terminated"] else 0.99 ** (steps / 16)
    if variant == "hybrid":
        # User-selected external time-limit truncation retains the continuation
        # value and next potential. True task termination settles both to zero.
        totals["progress"] = PARAMETERS["progress"] * (
            discount * samples[-1]["potential"] - samples[0]["potential"]
        )
        # Success wins if success and the time limit coincide.
        totals["timeout"] = 0.0  # v7: no separate deadline penalty
        totals["total"] = sum(value for key, value in totals.items() if key != "total")
    return totals, discount


def fixed_tcp(position, quaternion):
    """Fixed virtual tool point in the wrist/gripper BODY frame, quaternion wxyz.

    Offset matches the old jaw detection point at 0.6 rad in the pinned USD.
    Actual jaw angle is deliberately not an input; no physical geometry changes.
    """
    if (
        len(position) != 3
        or len(quaternion) != 4
        or not all(math.isfinite(x) for x in [*position, *quaternion])
    ):
        raise ValueError("Invalid wrist pose")
    norm = math.sqrt(sum(x * x for x in quaternion))
    if norm < 1e-8:
        raise ValueError("Zero quaternion")
    w, x, y, z = [v / norm for v in quaternion]
    vx, vy, vz = PARAMETERS["reference_offset_m"]
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [
        position[0] + vx + w * tx + y * tz - z * ty,
        position[1] + vy + w * ty + z * tx - x * tz,
        position[2] + vz + w * tz + x * ty - y * tx,
    ]
