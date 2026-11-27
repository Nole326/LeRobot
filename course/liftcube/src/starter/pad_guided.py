"""Front RGB contour localization, tracking and pad-aligned grasps for varied solids."""

from colorsys import rgb_to_hsv
from dataclasses import replace
from itertools import product

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .scripted_geometry import GeometricLiftCube, UnreachableGrasp, fit_cube_outline
from .scripted_liftcube import GraspStall, TargetNotVisible, pixel_to_base, red_mask


def tabletop_component(image, calibration, reference=None):
    """Find a compact foreground silhouette against the local tabletop appearance."""
    rgb = np.asarray(image, dtype=np.float32)
    h, w = rgb.shape[:2]
    # Learn the wood colour from table-like pixels. The front view also includes
    # a grey background and a yellow arm; neither is a suitable reference.
    r, g, b = rgb.transpose(2, 0, 1)
    wood = (r > g + 5) & (g > b + 7) & (r - b < 95) & (r > 110)
    wood[: h // 2] = False
    if wood.sum() < 100:
        return np.zeros((h, w), dtype=bool), None
    chroma = np.stack((r - g, g - b), axis=2)
    base = np.median(chroma[wood][::5], axis=0)
    dist = np.linalg.norm(chroma - base, axis=2)
    mask = ndimage.binary_closing(dist > 19, structure=np.ones((3, 3)))
    labels, n = ndimage.label(mask)
    candidates = []
    for i, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        sy, sx = region
        dy, dx = sy.stop - sy.start, sx.stop - sx.start
        if not (4 <= dx <= 0.15 * w and 4 <= dy <= 0.18 * h):
            continue
        area = ndimage.binary_fill_holes(labels[region] == i)
        yy, xx = np.nonzero(area)
        if len(yy) < max(12, 0.25 * dx * dy):
            continue
        yy, xx = yy + sy.start, xx + sx.start
        point = pixel_to_base([xx.mean(), yy.mean()], calibration)
        lo, hi = calibration["workspace"]
        if (
            not np.all((point[:2] >= lo[:2]) & (point[:2] <= hi[:2]))
            or np.linalg.norm(point[:2]) < 0.105
        ):
            continue
        # A free tabletop target should have wood around most of its boundary.
        ex = (
            slice(max(0, sy.start - 3), min(h, sy.stop + 3)),
            slice(max(0, sx.start - 3), min(w, sx.stop + 3)),
        )
        local = labels[ex] == i
        ring = ndimage.binary_dilation(local, iterations=3) & ~local
        surround = np.mean(wood[ex][ring]) if ring.any() else 0.0
        color = np.median(rgb[yy, xx], axis=0)
        if surround < 0.24 or (
            reference is not None and np.linalg.norm(color - reference) > 85
        ):
            continue
        confidence = (
            surround * np.sqrt(len(yy)) * min(2.0, np.median(dist[yy, xx]) / 25)
        )
        candidates.append((confidence, yy, xx, color))
    out = np.zeros((h, w), dtype=bool)
    if not candidates:
        return out, None
    _, yy, xx, color = max(candidates, key=lambda v: v[0])
    out[yy, xx] = True
    return out, color


def fit_solid(mask, c, *, dimensions=False):
    """Fit projected box, upright cylinder and sphere silhouettes on the calibrated plane."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise TargetNotVisible("No tabletop outline")
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dirs = np.array([np.cos(a), np.sin(a)])
    obs = (np.column_stack((xx, yy)) @ dirs).max(0) + 0.5
    xy = pixel_to_base([xx.mean(), yy.mean()], c)[:2] - c["grasp_xy_offset"]
    floor = c["visible_plane_z"] - 0.015
    eye = c["camera_to_base"]
    unit_box = np.array(list(product((-1.0, 1.0), repeat=3)))
    ring = dirs.T
    zring = np.array([-1.0, 1.0]).repeat(len(a))
    unit_cyl = np.column_stack((np.tile(ring, (2, 1)), zring))
    ph = np.linspace(-np.pi / 2, np.pi / 2, 13)
    sphere = np.column_stack(
        (
            np.repeat(np.cos(ph), len(a)) * np.tile(np.cos(a), len(ph)),
            np.repeat(np.cos(ph), len(a)) * np.tile(np.sin(a), len(ph)),
            np.repeat(np.sin(ph), len(a)),
        )
    )

    def residual(params, kind):
        x, y, rx = params[:3]
        if kind == "sphere":
            pts = sphere * rx
            half = rx
        elif kind == "cylinder":
            half = params[3]
            pts = unit_cyl * [rx, rx, half]
        else:
            ry, half, yaw = params[3:]
            pts = unit_box * [rx, ry, half]
            co, si = np.cos(yaw), np.sin(yaw)
            pts[:, :2] = pts[:, :2] @ np.array([[co, si], [-si, co]])
        base = pts + np.r_[xy + [x, y], floor + half]
        uv = (base - eye[:3, 3]) @ eye[:3, :3] @ c["K"].T
        outline = ((uv[:, :2] / uv[:, 2:]) @ dirs).max(0) - obs
        # A single oblique image couples height and range. Use the nominal
        # tabletop scale as a soft prior while fitting each silhouette.
        return np.r_[outline, (half - 0.015) / 0.0018]

    fits = []
    for kind in ("sphere", "cylinder", "box"):
        extra = {"sphere": [], "cylinder": [0.015], "box": [0.012, 0.015, 0.0]}[kind]
        low = [-0.025, -0.025, 0.007] + {
            "sphere": [],
            "cylinder": [0.008],
            "box": [0.007, 0.008, -np.pi / 2],
        }[kind]
        high = [0.025, 0.025, 0.024] + {
            "sphere": [],
            "cylinder": [0.023],
            "box": [0.024, 0.023, np.pi / 2],
        }[kind]
        for yaw in (-0.7, 0, 0.7) if kind == "box" else (0.0,):
            start = [0.0, 0.0, 0.014] + extra
            if kind == "box":
                start[-1] = yaw
            fit = least_squares(
                residual, start, args=(kind,), bounds=(low, high), max_nfev=55
            )
            # Rasterized boundaries have subpixel uncertainty. Penalize extra fit parameters.
            err = np.mean(fit.fun**2) + 0.06 * (len(start) - 3)
            fits.append((err, kind, fit.x))
    error, kind, p = min(fits, key=lambda item: item[0])
    height = p[2] if kind == "sphere" else p[3] if kind == "cylinder" else p[4]
    point = np.r_[xy + p[:2] + c["grasp_xy_offset"], floor + height]
    result = kind, point, float(p[-1]) if kind == "box" else None, float(np.sqrt(error))
    return (*result, p[2:4].copy() if kind == "box" else None) if dimensions else result


class _ContourGrasp(GeometricLiftCube):
    """Consume front RGB, measured joints and the same fixed arm/camera calibration."""

    def __init__(self, calibration, kinematics):
        super().__init__(dict(calibration, follow_retry_close=False), kinematics)

    def reset(self):
        super().reset()
        self.appearance = None
        self.kind = self.fit_error = None

    def act(self, observation, deterministic=False):
        mask, color = tabletop_component(observation.image, self.c, self.appearance)
        # Reuse the existing single-component motion and recovery checks.
        # The normalized mask is derived solely from the current RGB.
        binary = np.zeros_like(observation.image)
        binary[mask, 0] = 255
        before = self.phase
        cmd = super().act(replace(observation, image=binary), deterministic)
        if (
            before in ("open", "localize")
            and self.phase == "hover"
            and color is not None
        ):
            self.appearance = color
        self.telemetry.update(outline=self.kind, outline_error_px=self.fit_error)
        return cmd

    def _plan_from_image(self, image, pixel, q):
        self.kind, point, self.face_yaw, self.fit_error = fit_solid(
            image[:, :, 0] > 0, self.c
        )
        self.c["grasp_z"] = float(
            np.clip(
                point[2] - (0.001 if self.kind == "sphere" else 0.003),
                self.c["workspace"][0, 2] + 0.003,
                self.c["hover_z"] - 0.012,
            )
        )
        self.pad_active = self.kind == "box"
        try:
            self.select_pose(point, q)
        except UnreachableGrasp:
            if not self.pad_active:
                raise
            self.pad_active = False
            self.select_pose(point, q)
        self.approach_z = self.c["hover_z"]
        return point

    def _step(self, image, q):
        goal = super()._step(image, q)
        if self.phase in ("close", "lift", "hold") and self.jaw_blocked:
            # Bounded position preload follows measured aperture as an object settles.
            # It persists during ascent and hold with the native actuator torque cap.
            self.grasp_motor = max(
                self.c["close_motor"], min(float(self.last[5]), float(q[5]) - 3.0)
            )
            goal[5] = self.grasp_motor
        return goal


def tracked_component(image, pixel, reference, area):
    """Use the target's observed colour in a small image neighbourhood during approach."""
    u, v = np.rint(pixel).astype(int)
    radius = int(np.clip(np.sqrt(area) * 1.6, 16, 31))
    y0 = max(0, v - radius)
    x0 = max(0, u - radius)
    tile = image[y0 : v + radius + 1, x0 : u + radius + 1, :3].astype(np.float32)

    def chroma(c):
        return c / np.maximum(c.sum(axis=-1, keepdims=True), 45.0)

    diff = np.linalg.norm(chroma(tile) - chroma(reference), axis=-1)
    luma = np.abs(tile.mean(axis=2) - reference.mean())
    mask = (
        (diff < 0.15) & (luma < 72) & (np.linalg.norm(tile - reference, axis=2) < 100)
    )
    labels, n = ndimage.label(mask)
    best = None
    for i, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        yy, xx = np.nonzero(labels[slices] == i)
        if not max(9, area * 0.12) <= len(xx) <= area * 1.8:
            continue
        yy += slices[0].start + y0
        xx += slices[1].start + x0
        distance = np.linalg.norm([xx.mean() - u, yy.mean() - v])
        if distance > radius * 0.8:
            continue
        score = distance + 3 * abs(np.log(len(xx) / area))
        if best is None or score < best[0]:
            best = (score, yy, xx)
    if best is None:
        return None
    mask = np.zeros(image.shape[:2], bool)
    mask[best[1], best[2]] = True
    return mask


class _TrackedGrasp(_ContourGrasp):
    def __init__(self, calibration, kinematics):
        self.nominal_hover = calibration["hover_z"]
        super().__init__(calibration, kinematics)

    def reset(self):
        super().reset()
        self.occluded = self.shifted = 0
        self.area = None
        self.close_to_ray = False

    def _plan_from_image(self, image, pixel, q):
        mask = image[:, :, 0] > 0
        self.area = int(mask.sum())
        self.occluded = self.shifted = 0
        self.c["hover_z"] = self.nominal_hover
        kind, _, _, _ = fit_solid(mask, self.c)
        if kind == "sphere":
            self.c["hover_z"] = self.nominal_hover + 0.03
        try:
            return super()._plan_from_image(image, pixel, q)
        except UnreachableGrasp:
            if self.c["hover_z"] == self.nominal_hover:
                raise
            self.c["hover_z"] = self.nominal_hover
            return super()._plan_from_image(image, pixel, q)

    def act(self, observation, deterministic=False):
        self.tracked = None
        if (
            self.pixel is not None
            and self.appearance is not None
            and self.area
            and self.phase in ("hover", "descend", "close", "lift", "hold")
        ):
            self.tracked = tracked_component(
                observation.image, self.pixel, self.appearance, self.area
            )
            if self.phase == "hover":
                pos = self._pose(np.asarray(observation.proprio[:6]))[:3, 3]
                cam = self.c["camera_to_base"]
                uv = self.c["K"] @ ((pos - cam[:3, 3]) @ cam[:3, :3])
                self.close_to_ray = (
                    uv[2] > 0 and np.linalg.norm(uv[:2] / uv[2] - self.pixel) < 37
                )
        return super().act(observation, deterministic)

    def _check_target(self, image):
        mask = self.tracked if self.tracked is not None else image[:, :, 0] > 0
        y, x = np.nonzero(mask)
        if not len(x):
            self.occluded += 1
            if self.occluded <= (45 if self.close_to_ray else 5):
                return
            raise GraspStall("Target remained hidden beyond the approach window")
        self.occluded = 0
        if np.linalg.norm([x.mean() - self.pixel[0], y.mean() - self.pixel[1]]) <= 12:
            self.shifted = 0
            return
        self.shifted += 1
        if self.shifted >= 4:
            raise GraspStall("Target moved away during approach")


class _ShortSideGrasp(_TrackedGrasp):
    aligned_close_height_m = 0.008

    def __init__(self, calibration, kinematics):
        self.aperture_tcp = np.array(calibration["tcp_offset_m"])
        super().__init__(calibration, kinematics)

    def _plan_from_image(self, image, pixel, q):
        mask = image[:, :, 0] > 0
        kind, point, yaw, error, widths = fit_solid(mask, self.c, dimensions=True)
        self.pose_path, self.short_axis = "contour", None
        self.planned_tilt = self.next_face_pair = None
        if kind == "box" and max(widths) > 1.20 * min(widths):
            self.short_axis = float(yaw + (np.pi / 2 if widths[1] < widths[0] else 0.0))
            self.face_yaw = self.short_axis
            self.next_face_pair = 0
            self.kin.tcp = self.aperture_tcp.copy()
            self.c["grasp_z"] = float(
                np.clip(
                    point[2],
                    self.c["workspace"][0, 2] + 0.003,
                    self.nominal_hover - 0.012,
                )
            )
            self.c["hover_z"] = self.nominal_hover
            self._select_orientations(point)
            hover = self._face_pose(point, q)
            if hover is not None:
                try:
                    self._plan_lift(point, hover)
                except GraspStall:
                    pass
                else:
                    self.hover_motor = hover
                    self.transport_rotation = self.c["lift_rotation"].copy()
                    self.area, self.occluded, self.shifted = int(mask.sum()), 0, 0
                    self.approach_z = self.nominal_hover
                    self.pad_active = True
                    self.kind, self.fit_error = kind, error
                    self.pose_path = "short_face"
                    return point
        self.planned_tilt = self.next_face_pair = None
        return super()._plan_from_image(image, pixel, q)


class PadGuided(_ShortSideGrasp):
    def __init__(self, calibration, kinematics):
        self.start_open = calibration["open_motor"]
        self.front_rgb = None
        super().__init__(calibration, kinematics)

    def act(self, observation, deterministic=False):
        self.front_rgb = observation.image
        return super().act(observation, deterministic)

    def _red_cube_outline(self, image, pixel, mask):
        """Refine an already identified cube outline before planning the descent."""
        if self.front_rgb is None:
            return image, pixel
        yy, xx = np.nonzero(mask)
        rgb = self.front_rgb
        hue, saturation, value = rgb_to_hsv(
            *(np.median(rgb[yy, xx, :3], axis=0) / 255.0)
        )
        if min(hue, 1 - hue) >= 0.055 or saturation <= 0.45 or value <= 0.31:
            return image, pixel
        ys = slice(max(0, yy.min() - 2), min(mask.shape[0], yy.max() + 3))
        xs = slice(max(0, xx.min() - 2), min(mask.shape[1], xx.max() + 3))
        components, _ = ndimage.label(red_mask(rgb[ys, xs, :3]))
        overlap = np.bincount(components[mask[ys, xs]], minlength=2)
        label = 1 + np.argmax(overlap[1:])
        if overlap[label] < max(12, 0.6 * len(yy)):
            return image, pixel
        refined = np.zeros_like(image)
        region = components == label
        refined[ys, xs, 0] = region * 255
        y, x = np.nonzero(region)
        return refined, np.array([xs.start + x.mean(), ys.start + y.mean()])

    def _plan_from_image(self, image, pixel, q):
        mask = image[:, :, 0] > 0
        kind, point, yaw, error, widths = fit_solid(mask, self.c, dimensions=True)
        self.wide = None
        self.c["open_motor"] = self.start_open
        self.c["follow_retry_close"] = False
        self.c["hover_z"] = self.nominal_hover
        # Test the projected physical cube against the outline, including shapes
        # the free cylinder fit can explain as well at low image resolution.
        if kind in ("box", "cylinder") and (
            widths is None or max(widths) / min(widths) < 1.2
        ):
            center, angle = fit_cube_outline(image, pixel, self.c)
            if angle is not None and self._cube_error(mask, center, angle) <= 0.95:
                self.c["follow_retry_close"] = True
                self.kind, self.fit_error, self.short_axis = "cube", error, None
                self.pose_path = "cube_preset"
                self.area, self.occluded, self.shifted = int(mask.sum()), 0, 0
                image, pixel = self._red_cube_outline(image, pixel, mask)
                return GeometricLiftCube._plan_from_image(self, image, pixel, q)
        if kind == "box" and max(widths) / min(widths) >= 1.20:
            self.wide = (widths.copy(), yaw)
            self.c["open_motor"] = 65.0
        return super()._plan_from_image(image, pixel, q)

    def _cube_error(self, mask, center, yaw):
        y, x = np.nonzero(mask)
        a = np.arange(32) * np.pi / 16
        dirs = np.stack((np.cos(a), np.sin(a)))
        observed = (np.column_stack((x, y)) @ dirs).max(axis=0)
        corners = np.indices((2, 2, 2)).reshape(3, -1).T * 0.030 - 0.015
        cs, sn = np.cos(yaw), np.sin(yaw)
        corners[:, :2] = corners[:, :2] @ np.array([[cs, sn], [-sn, cs]])
        center = center.copy()
        center[:2] -= self.c["grasp_xy_offset"]
        camera = self.c["camera_to_base"]
        uv = (corners + center - camera[:3, 3]) @ camera[:3, :3] @ self.c["K"].T
        return np.sqrt(
            np.mean(((uv[:, :2] / uv[:, 2:]) @ dirs - observed).max(axis=0) ** 2)
        )

    def _pad_tcp(self, point, rotation, source):
        widths, yaw = self.wide
        axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        normal = rotation[:, 0]
        table = self.c["visible_plane_z"] - 0.015
        half = np.abs(axes @ normal[:2]) @ widths + abs(normal[2]) * (point[2] - table)
        tcp = source.copy()
        tcp[0] = -0.0079 + half + 0.006
        tcp[1] = 0.0
        vertices = np.asarray(self.c["pad_clearance_vertices_m"])
        tcp[2] += (
            ((vertices - tcp) @ rotation[2]).min() + self.c["grasp_z"] - table - 0.0025
        ) / rotation[2, 2]
        return tcp

    def _pad_grasps(self, point):
        choices = super()._pad_grasps(point)
        if self.wide is not None:
            radial = np.arctan2(point[0], -point[1])
            cs, sn = np.cos(radial), np.sin(radial)
            world = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
            for choice in choices:
                choice["tcp_offset_m"] = self._pad_tcp(
                    point, world @ choice["rotation"], choice["tcp_offset_m"]
                )
        return choices

    def _face_pose(self, point, measured):
        hover = super()._face_pose(point, measured)
        if hover is None or getattr(self, "wide", None) is None:
            return hover
        rotation = self.c["grasp_rotation"]
        # Project the fitted box onto the closing axis. The fixed pad's
        # inner face is x=-7.9 mm at 9 mm above its tip (measured from CAD).
        goal_z = self.c["grasp_z"]
        original = self.kin.tcp.copy()
        self.kin.tcp = self._pad_tcp(point, rotation, original)
        try:
            candidate = self._solve(
                np.r_[point[:2], self.c["hover_z"]],
                measured,
                self.c["open_motor"],
                rotation=rotation,
            )
            self._solve(
                np.r_[point[:2], goal_z],
                candidate,
                self.c["open_motor"],
                rotation=rotation,
            )
        except (GraspStall, ValueError):
            self.kin.tcp = original
            return None
        return candidate
