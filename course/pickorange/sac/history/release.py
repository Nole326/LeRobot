"""Confirm the release of one orange from the observed plate count and gripper opening."""

from collections import deque


class ReleasePlacementTrigger:
    def __init__(self):
        self.initial = None
        self.confirmed = None
        self.history = deque(maxlen=32)
        self.armed = False
        self.opened = False

    def update(self, gripper_motor, count):
        delta = gripper_motor - self.history[-1] if self.history else 0.0
        self.history.append(gripper_motor)
        if gripper_motor < 32.0:
            self.armed = True
            self.opened = False
        if (
            self.armed
            and gripper_motor >= 37.0
            and gripper_motor - min(self.history) >= 4.0
        ):
            self.opened = True
        if count is None:
            return None
        if self.confirmed is None:
            self.initial = self.confirmed = count
            return None
        if (
            self.opened
            and gripper_motor >= 37.0
            and delta <= 0.3
            and count > self.confirmed
        ):
            self.confirmed += 1
            self.armed = self.opened = False
            return self.confirmed
        return None
