"""Scene-level reward bands for an ACT residual policy. Object state is used only for scoring."""

import json
import math
from collections import deque

from ...groot_sac.rewards import fixed_tcp


class TieredReward:
    def __init__(self, path, horizon=2700, step_discount=0.99 ** (1 / 16)):
        self.stream = path.open()
        self.row = json.loads(next(self.stream))
        self.horizon = horizon
        self.gamma = step_discount
        self.ledger = SceneLedger(self.row)

    def next_skill(self):
        self.ledger.skill_start = list(self.row["inside"])

    def close(self):
        self.stream.close()

    def transition(self, start, end, terminal, success):
        assert self.row["step"] == start and end > start
        parts = {}
        for tick in range(start + 1, end + 1):
            self.row = json.loads(next(self.stream))
            assert self.row["step"] == tick
            changes = self.ledger.advance(
                self.row, success and tick == end, tick >= self.horizon
            )
            for key, value in changes.items():
                parts[key] = (
                    parts.get(key, 0.0) + self.gamma ** (tick - start - 1) * value
                )
        return parts, 0.0 if terminal else self.gamma ** (end - start)


class SceneLedger:
    def __init__(self, first):
        assert first["step"] == 0 and not any(first["inside"])
        self.skill_start = list(first["inside"])
        self.initial_z = [p[2] for p in first["oranges"]]
        self.held = None
        self.grasped = set()
        self.home = set()
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
        self.previous_count = self.native = 0
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
        return dict(
            retained=20.0 * sum(self.stable),
            home=2.0 * len(self.home),
            completion=10.0 * self.native,
            reach=self.approach,
            grasp=self.credit,
            carry=self.carry,
            loss=self.penalty,
            time=-0.0005 * max(0, step - 1200),
        )

    def advance(self, row, home_success, timeout):
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
        if home_success:
            added = [i for i in range(3) if inside[i] and not self.skill_start[i]]
            if len(added) == 1:
                self.home.add(added[0])
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
