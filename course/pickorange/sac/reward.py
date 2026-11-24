"""Whole-scene reward ledger, evaluated inside the simulator at 60 Hz."""

import math
from collections import deque

BANDS = ((-20.0, -5.0), (-1.0, 3.0), (15.0, 25.0), (40.0, 50.0))


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
    vx, vy, vz = (0.04239292443776763, -0.00119974788459086, -0.09303105684230396)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [
        position[0] + vx + w * tx + y * tz - z * ty,
        position[1] + vy + w * ty + z * tx - x * tz,
        position[2] + vz + w * tz + x * ty - y * tx,
    ]


class SceneLedger:
    def __init__(self, first):
        assert first["step"] == 0 and not any(first["inside"])
        self.initial_z = [p[2] for p in first["oranges"]]
        self.held = None
        self.grasped = set()
        self.ever_placed = set()
        self.stable = [False] * 3
        self.ins = [0] * 3
        self.outs = [0] * 3
        self.confirmed_release = [False] * 3
        self.pending = None
        self.loss_ticks = 0
        self.approach = self.credit = self.carry = self.penalty = 0.0
        self.dist0 = self.best_d = self.plate_dist(first)
        self.last_d = self.dist0[:]
        self.best_reach = max(
            0.04,
            min(0.12, min(math.dist(self.tcp(first), p) for p in first["oranges"])),
        )
        self.native = 0
        self.history = deque(maxlen=37)
        self.record(first)

    @staticmethod
    def tcp(row):
        if "audit_tcp" in row:
            return row["audit_tcp"]
        return fixed_tcp(row["gripper_xyz"], row["gripper_quat"])

    @staticmethod
    def plate_dist(row):
        return [
            max(0.0, math.hypot(p[0] - row["plate"][0], p[1] - row["plate"][1]) - 0.045)
            for p in row["oranges"]
        ]

    def record(self, row):
        self.history.append(
            (self.tcp(row), row["oranges"], row["q_rad"][5], row["inside"])
        )

    def lift(self, i):
        tcp, fruit, angle, _ = self.history[-1]
        if angle > 0.85 or math.dist(tcp, fruit[i]) > 0.05:
            return False
        for w in (6, 12, 18, 24, 36):
            if len(self.history) <= w:
                break
            t0, f0, _, _ = self.history[-1 - w]
            if min(tcp[2] - t0[2], fruit[i][2] - f0[i][2]) < 0.015:
                continue
            relative = [v - u for v, u in zip(f0[i], t0)]
            for j in {round(w * k / 4) for k in range(5)}:
                t, f, a, inside = self.history[-1 - w + j]
                if (
                    inside[i]
                    or a > 0.85
                    or math.dist(t, f[i]) > 0.05
                    or math.dist(relative, [v - u for v, u in zip(f[i], t)]) > 0.008
                ):
                    break
            else:
                return True
        return False

    def components(self, step):
        # Book the initial scene score on the first physics transition, exactly once.
        # All later rewards are score differences, including across policy resets.
        n = sum(self.stable)
        lower, upper = BANDS[n]
        time_quality = 1.0 - min(1.0, max(0.0, step / 60.0 - 20.0) / 100.0)
        clean = 1.0 + self.penalty / 3.0
        time_weight, clean_weight = 0.625, 0.375
        # The first three grasps receive identical increments before placement.
        incomplete = n < 3
        quality_budget = upper - lower - 3.25 if incomplete else 4.0
        parts = dict(
            tier=lower,
            reach=self.approach if incomplete else 0.0,
            grasp=self.credit if incomplete else 0.0,
            carry=self.carry if incomplete else 0.0,
            timeliness=quality_budget * time_weight * time_quality,
            handling=quality_budget * clean_weight * clean,
            completion=6.0 * int(bool(self.native) and n == 3),
        )
        if step == 0:
            return dict.fromkeys(parts, 0.0)
        return parts

    def advance(self, row, timeout):
        tick = row["step"]
        before = self.components(tick - 1)
        self.record(row)
        tcp, fruits, jaw, inside = self.history[-1]
        dist = self.plate_dist(row)
        held_before = self.held
        if self.held is None and self.pending is None:
            available = [i for i in range(3) if not self.stable[i] and not inside[i]]
            if available and jaw > 0.35:
                near = max(
                    0.04, min(0.12, min(math.dist(tcp, fruits[i]) for i in available))
                )
                self.approach = min(
                    0.25, self.approach + 0.25 * max(0.0, self.best_reach - near) / 0.08
                )
                self.best_reach = min(self.best_reach, near)
            for i in sorted(available, key=lambda i: math.dist(tcp, fruits[i])):
                if self.lift(i):
                    self.held = i
                    self.credit = 0.0 if i in self.grasped else 2.0
                    self.grasped.add(i)
                    self.loss_ticks = 0
                    break
        if self.held is not None:
            i = self.held
            lost = math.dist(tcp, fruits[i]) > 0.065 or jaw > 0.90
            self.loss_ticks = self.loss_ticks + 1 if lost else 0
            if self.loss_ticks >= 3:
                self.pending = i
                self.held = None
                self.loss_ticks = 0
            elif held_before == i and not lost:
                gain = max(0.0, min(self.last_d[i], self.best_d[i]) - dist[i]) / max(
                    0.05, self.dist0[i]
                )
                self.carry = min(1.0, self.carry + gain)
        for i, in_plate in enumerate(inside):
            speed = math.sqrt(sum(v * v for v in row["orange_velocities"][i]))
            released = math.dist(tcp, fruits[i]) > 0.065 or (
                jaw > 0.90 and speed <= 0.12
            )
            self.confirmed_release[i] = in_plate and (
                self.confirmed_release[i] or released
            )
            self.ins[i] = (
                self.ins[i] + 1 if in_plate and self.confirmed_release[i] else 0
            )
            self.outs[i] = 0 if in_plate else self.outs[i] + 1
            if (
                in_plate
                and (self.ins[i] >= 12 or row["success"])
                and not self.stable[i]
            ):
                self.stable[i] = True
                if i == self.held or i == self.pending:
                    self.held = self.pending = None
                    self.credit = self.carry = 0.0
            if self.stable[i] and (self.outs[i] >= 12 or (timeout and not in_plate)):
                self.stable[i] = False
                self.penalty = max(-3.0, self.penalty - 1.0)
        if self.pending is not None:
            i = self.pending
            if (
                not inside[i]
                and fruits[i][2] <= self.initial_z[i] + 0.012
                and (math.dist(tcp, fruits[i]) > 0.065 or jaw > 0.90)
            ):
                self.credit = self.carry = 0.0
                self.penalty = max(-3.0, self.penalty - 0.25)
                self.pending = None
            elif not inside[i] and self.lift(i):
                self.held, self.pending = i, None
        new = {i for i in range(3) if self.stable[i]} - self.ever_placed
        if new:
            self.ever_placed.update(new)
            self.approach = 0.0
            self.best_reach = max(
                0.04,
                min(
                    [0.12]
                    + [math.dist(tcp, fruits[i]) for i in range(3) if not inside[i]]
                ),
            )
        self.native = int(bool(self.native or row["success"]))
        self.best_d = [min(a, b) for a, b in zip(self.best_d, dist)]
        self.last_d = dist
        after = self.components(tick)
        return {key: val - before[key] for key, val in after.items()}
