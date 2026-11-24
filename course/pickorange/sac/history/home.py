"""Single-orange termination and a joint-space return using measured joints."""

import json
from itertools import islice

import numpy as np


HOME = np.array([0.0, -86.0, 86.0, 53.0, 0.0, 6.0], np.float32)
REST_LOW = np.array(
    [
        -30 / 110 * 100,
        -130,
        200 * 160 / 190 - 100,
        20 / 95 * 100,
        -30 / 160 * 100,
        -30 / 110 * 100,
    ]
)
REST_HIGH = np.array(
    [
        30 / 110 * 100,
        -70,
        200 * 220 / 190 - 100,
        80 / 95 * 100,
        30 / 160 * 100,
        30 / 110 * 100,
    ]
)


def at_home(q):
    return bool(((q > REST_LOW) & (q < REST_HIGH)).all())


class ReturnHome:
    def __init__(self, q):
        opening = max(q[5], 40.0)
        retreat = q.copy()
        retreat[1] = min(60.0, q[1] + 18.0)
        retreat[2] = max(-70.0, q[2] - 12.0)
        retreat[5] = opening
        fold = HOME.copy()
        fold[[0, 4, 5]] = [q[0], q[4], opening]
        self.waypoints = [retreat, fold, HOME]
        self.index = 0
        self.command = q.copy()

    def action(self, q):
        target = self.waypoints[self.index]
        if self.index < 2 and np.abs(q - target).max() < 8.0:
            self.index += 1
            target = self.waypoints[self.index]
        self.command += np.clip(target - self.command, -4.0, 4.0)
        return self.command.copy()


def score_one_orange(path, start, end):
    """Read the two boundary records for the reward calculation, outside the policy."""
    with path.open() as stream:
        first = json.loads(next(islice(stream, start, start + 1)))
        last = json.loads(next(islice(stream, end - start - 1, end - start)))
    assert first["step"] == start and last["step"] == end
    before, after = first["inside"], last["inside"]
    kept = all(not old or new for old, new in zip(before, after))
    rad = np.asarray(last["q_rad"])
    native = np.deg2rad([30.0, -70.0, 60.0, 20.0, 30.0, 20.0])
    home = (
        abs(rad[0]) < native[0]
        and rad[1] < native[1]
        and rad[2] > native[2]
        and native[3] < rad[3] < np.deg2rad(80.0)
        and abs(rad[4]) < native[4]
        and rad[5] < native[5]
    )
    return (
        bool(kept and sum(after) == sum(before) + 1 and home),
        sum(before),
        sum(after),
    )
