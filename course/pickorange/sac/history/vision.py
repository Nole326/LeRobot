"""Track a placed orange using only RGB pixels and measured joint positions."""

from collections import deque
from itertools import permutations

import cv2
import numpy as np


def hsv_small(rgb):
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    return cv2.cvtColor(
        cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV
    )


def orange_mask(hsv):
    mask = cv2.inRange(hsv, (2, 140, 66), (22, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def plate_ellipse(hsv):
    mask = cv2.inRange(hsv, (0, 0, 125), (179, 71, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if len(c) < 12 or not 1000 < area < 19500:
            continue
        hull = cv2.convexHull(c)
        if len(hull) < 5:
            continue
        center, axes, angle = cv2.fitEllipse(hull)
        minor, major = sorted(axes)
        ellipse_area = np.pi * minor * major / 4
        if not (
            29 < minor <= major < 173
            and major / minor < 2.2
            and 0.45 < area / ellipse_area < 1.18
        ):
            continue
        if best is None or area > best[0]:
            best = area, (center, axes, angle)
    return None if best is None else best[1]


def orange_centers(mask):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    centers = []
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] >= 0:
            continue
        area = cv2.contourArea(contour)
        if not 32 < area < 2800:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 6:
            continue
        shape = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(
            shape,
            [contour - np.array([x - 1, y - 1], dtype=contour.dtype)],
            0,
            255,
            cv2.FILLED,
        )
        distance = cv2.distanceTransform(shape, cv2.DIST_L2, 5)
        max_radius = float(distance.max())
        if max_radius < 3.3 or area > 6 * max_radius**2:
            continue
        if area < 4.0 * max_radius**2:
            moments = cv2.moments(contour)
            centers.append((moments["m10"] / area, moments["m01"] / area))
            continue
        peaks = (distance >= cv2.dilate(distance, np.ones((7, 7), np.uint8))) & (
            distance > max(3.5, 0.63 * max_radius)
        )
        candidates = sorted(zip(*np.nonzero(peaks)), key=lambda p: -distance[p])
        picked = []
        for py, px in candidates:
            if all(
                (px - cx) ** 2 + (py - cy) ** 2 > (1.3 * max_radius) ** 2
                for cy, cx in picked
            ):
                picked.append((py, px))
        centers.extend((float(x + px - 1), float(y + py - 1)) for py, px in picked[:3])
    return np.asarray(centers, dtype=float).reshape(-1, 2)


def in_plate(points, ellipse):
    center, axes, angle = ellipse
    theta = np.deg2rad(angle)
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    radius = np.linalg.norm(
        ((points - center) @ rotate) / (np.asarray(axes) / 2), axis=1
    )
    return points[radius <= 0.87]


def same_positions(first, last):
    if len(first) != len(last):
        return False
    return (
        not len(first)
        or min(
            np.max(np.linalg.norm(first - last[list(p)], axis=1))
            for p in permutations(range(len(last)))
        )
        <= 3.3
    )


class PlacementDetector:
    def __init__(self):
        self.plate = None
        self.frames = deque(maxlen=8)
        self.release_window = deque(maxlen=32)
        self.confirmed = None
        self.initial = None
        self.frame = 0

    def update(self, front_rgb, wrist_rgb, joints):
        self.frame += 1
        hsv = hsv_small(front_rgb)
        current_plate = (
            plate_ellipse(hsv) if self.plate is None or self.frame <= 4 else None
        )
        if current_plate is not None and (
            self.plate is None
            or np.linalg.norm(np.subtract(current_plate[0], self.plate[0])) < 7
        ):
            self.plate = current_plate
        wrist_orange = float(
            np.mean(orange_mask(hsv_small(wrist_rgb))[73:165, 110:218] > 0)
        )
        gripper = float(joints[-1])
        self.release_window.append((gripper, wrist_orange))
        opening = gripper - min(item[0] for item in self.release_window)
        cleared = max(item[1] for item in self.release_window) - wrist_orange
        released = (gripper >= 33 and opening >= 4.0) or (
            gripper >= 38 and cleared >= 0.025
        )
        if self.plate is None:
            return dict(
                count=None,
                confirmed=self.confirmed,
                initial=self.initial,
                event=None,
                plate=False,
            )
        points = in_plate(orange_centers(orange_mask(hsv)), self.plate)
        if len(points) > 3:
            self.frames.clear()
            return dict(
                count=None,
                confirmed=self.confirmed,
                initial=self.initial,
                event=None,
                plate=True,
            )
        self.frames.append(points)
        stable = len(self.frames) == 8 and all(
            same_positions(points, old) for old in self.frames
        )
        count = len(points)
        event = None
        if stable and self.confirmed is None and self.frame <= 25:
            self.initial = self.confirmed = count
        if (
            stable
            and self.confirmed is not None
            and count > self.confirmed
            and released
        ):
            self.confirmed = count
            event = count
        return dict(
            count=count,
            confirmed=self.confirmed,
            initial=self.initial,
            event=event,
            plate=True,
            stable=stable,
            released=released,
            wrist_orange=wrist_orange,
            opening=opening,
            centers=points.tolist(),
            plate_pose=self.plate,
        )


if __name__ == "__main__":
    front = np.full((480, 640, 3), (142, 92, 55), dtype=np.uint8)
    cv2.ellipse(front, (390, 270), (100, 80), 0, 0, 360, (238, 229, 219), -1)
    wrist = front.copy()
    detector = PlacementDetector()
    for _ in range(12):
        empty = detector.update(front, wrist, [0, 0, 0, 0, 0, 25])
    assert empty["initial"] == 0, empty
    cv2.circle(front, (385, 279), 25, (240, 117, 12), -1)
    events = [
        detector.update(front, wrist, [0, 0, 0, 0, 0, 44])["event"] for _ in range(14)
    ]
    assert events.count(1) == 1, (
        events,
        detector.update(front, wrist, [0, 0, 0, 0, 0, 44]),
    )
