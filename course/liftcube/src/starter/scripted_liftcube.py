"""Front-RGB rule baseline. Calibration is required; no simulator state is read."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from starter.interfaces.action import ActionSpec, degrees_to_motor, motor_to_degrees, validate_action


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _rotation(value, name):
    result = _array(value, (3, 3), name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(result), 1):
        raise ValueError(f"{name} must be a proper rotation")
    return result


def validate_calibration(data):
    """Validate numbers, not the truth of an operator's calibration evidence."""
    c = dict(data)
    if c.get("schema") != "so101-front-rgb-rule-v1" or not c.get("evidence"):
        raise ValueError("Calibration needs schema and evidence reference")
    if c.get("image_shape") not in ([64,64,3], [128,128,3], [256,256,3]):
        raise ValueError("Calibrate the configured square front RGB (64, 128 or 256)")
    c["K"] = _array(c["K"], (3, 3), "K")
    if min(c["K"][0, 0], c["K"][1, 1]) <= 0 or not np.allclose(c["K"][2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    c["camera_to_base"] = _array(c["camera_to_base"], (4, 4), "camera_to_base")
    _rotation(c["camera_to_base"][:3, :3], "camera rotation")
    if not np.allclose(c["camera_to_base"][3], [0, 0, 0, 1]):
        raise ValueError("Invalid camera transform")
    c["grasp_rotation"] = _rotation(c["grasp_rotation"], "grasp_rotation")
    c["lift_rotation"] = _rotation(c.get("lift_rotation", c["grasp_rotation"]), "lift_rotation")
    if not isinstance(c.get("gravity_compensation", False), bool):
        raise ValueError("gravity_compensation must be boolean")
    if not isinstance(c.get("contact_close", False), bool):
        raise ValueError("contact_close must be boolean")
    for key in ("probe_loaded_close", "retry_dropped_cube", "joint_hover_approach", "clear_retry_view", "parallel_approach"):
        if not isinstance(c.get(key, False), bool):
            raise ValueError(f"{key} must be boolean")
    speed = c.get("arm_travel_speed", 1.)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 1 <= speed <= 2:
        raise ValueError("arm_travel_speed must be between 1 and 2")
    if c.get("probe_loaded_close") and not c.get("contact_close"):
        raise ValueError("probe_loaded_close requires contact_close")
    follow = c.get("follow_retry_close", False)
    if not isinstance(follow, bool) or (follow and not c.get("hold_contact_on_lift")):
        raise ValueError("follow_retry_close requires contact holding when enabled")
    if not isinstance(c.get("hold_contact_on_lift", False), bool) or (c.get("hold_contact_on_lift") and not c.get("contact_close")):
        raise ValueError("hold_contact_on_lift requires contact_close when enabled")
    extra = c.get("contact_preload_motor", 0.)
    if not isinstance(c.get("pace_loaded_lift", False), bool):
        raise ValueError("pace_loaded_lift must be boolean")
    if not isinstance(extra, (int, float)) or isinstance(extra, bool) or not 0. <= extra <= 1. or (extra and not c.get("hold_contact_on_lift")):
        raise ValueError("contact_preload_motor requires contact holding and a value from 0 to 1")
    if not isinstance(c.get("face_aligned_grasp", False), bool):
        raise ValueError("face_aligned_grasp must be boolean")
    # Preserve the earlier face-aligned preset while allowing the arrival test
    # to be varied independently of pose selection.
    if not isinstance(c.get("relaxed_close_window", False), bool):
        raise ValueError("relaxed_close_window must be boolean")
    if not isinstance(c.get("plan_while_opening", False), bool):
        raise ValueError("plan_while_opening must be boolean")
    if c.get("plan_while_opening") and not c.get("grasp_presets"):
        raise ValueError("plan_while_opening requires geometric grasp_presets")
    fast_clearance = c.get("fast_clearance", False)
    if not isinstance(fast_clearance, bool) or (fast_clearance and not c.get("grasp_presets")):
        raise ValueError("fast_clearance requires a boolean and geometric grasp presets when enabled")
    fast_loaded = c.get("fast_loaded_clearance", False)
    if not isinstance(fast_loaded, bool) or (fast_loaded and not (fast_clearance and c.get("probe_loaded_close"))):
        raise ValueError("fast_loaded_clearance requires fast_clearance and probe_loaded_close")
    c["joint_sign"] = _array(c["joint_sign"], (5,), "joint_sign")
    if not np.isin(c["joint_sign"], [-1, 1]).all():
        raise ValueError("joint_sign must contain only -1/+1")
    c["joint_offset_deg"] = _array(c["joint_offset_deg"], (5,), "joint_offset_deg")
    c["grasp_xy_offset"] = _array(c["grasp_xy_offset"], (2,), "grasp_xy_offset")
    c["workspace"] = _array(c["workspace"], (2, 3), "workspace")
    if not (c["workspace"][0] < c["workspace"][1]).all():
        raise ValueError("Workspace lower bounds must be below upper bounds")
    for name in ("visible_plane_z", "grasp_z", "hover_z", "lift_z", "open_motor", "close_motor"):
        c[name] = float(c[name])
        if not np.isfinite(c[name]):
            raise ValueError(f"Invalid {name}")
    if not (0 <= c["close_motor"] < c["open_motor"] <= 100):
        raise ValueError("Expected 0 <= close_motor < open_motor <= 100")
    if not (c["workspace"][0, 2] <= c["grasp_z"] < c["hover_z"] <= c["lift_z"] <= c["workspace"][1, 2]):
        raise ValueError("Grasp/hover/lift heights must be ordered inside workspace")
    names = c.get("joint_names", [])
    if len(names) != 5 or len(set(names)) != 5 or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Provide five ordered URDF arm joint names")
    if not c.get("tcp_frame"):
        raise ValueError("Provide the calibrated TCP frame")
    return c


def load_calibration(path):
    path = Path(path)
    c = validate_calibration(json.loads(path.read_text(encoding="utf-8")))
    backend = c.get("kinematics_backend", "urdf")
    if backend not in ("urdf", "joint_chain"):
        raise ValueError("Unknown kinematics_backend")
    key = "chain" if backend == "joint_chain" else "urdf"
    source = (path.parent / c[key+"_path"]).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != c[key+"_sha256"]:
        raise ValueError(f"{key} SHA256 mismatch")
    c[key+"_path"] = str(source)
    if backend == "joint_chain":
        c["tcp_offset_m"] = _array(c["tcp_offset_m"], (3,), "tcp_offset_m")
        if not np.array_equal(c["joint_sign"], np.ones(5)) or np.any(c["joint_offset_deg"] != 0):
            raise ValueError("joint_chain uses native USD axes and zero offsets")
    return c


def red_mask(image):
    r, g, b = image.astype(float).transpose(2, 0, 1)
    return (r >= 80) & (r > 1.5 * g) & (r > 1.5 * b) & ((r - np.maximum(g, b)) >= 35)


def red_centroid(image, expected_shape=(64,64,3)):
    """One isolated red component; reject ambiguity instead of guessing a target."""
    image = np.asarray(image)
    if image.shape != tuple(expected_shape) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 front RGB {tuple(expected_shape)}")
    height, width = image.shape[:2]
    mask = red_mask(image)
    components = []
    for y, x in np.argwhere(mask):
        if not mask[y, x]:
            continue
        stack, pixels = [(int(y), int(x))], []
        mask[y, x] = False
        while stack:
            yy, xx = stack.pop()
            pixels.append((xx, yy))
            for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
                    mask[ny, nx] = False
                    stack.append((ny, nx))
        if len(pixels) >= 3 * height * width / 4096:
            points = np.asarray(pixels)
            if (len(points) > height*width/4 or np.any(points[:,0] == 0)
                    or np.any(points[:,0] == width-1) or np.any(points[:,1] == 0)
                    or np.any(points[:,1] == height-1)):
                raise TargetNotVisible("Red region is oversized or touches image boundary")
            components.append(points.mean(axis=0))
    if len(components) != 1:
        raise TargetNotVisible(f"Expected one red cube, found {len(components)} components")
    return components[0]


def pixel_to_base(pixel, c):
    """Intersect an optical-camera ray with the calibrated visible object plane."""
    ray = c["camera_to_base"][:3, :3] @ np.linalg.solve(c["K"], np.r_[pixel, 1.0])
    origin = c["camera_to_base"][:3, 3]
    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray parallel to object plane")
    distance = (c["visible_plane_z"] - origin[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Object plane behind camera")
    point = origin + distance * ray
    point[:2] += c["grasp_xy_offset"]
    return point


def rotation_step(current, target, max_angle=.02):
    """Shortest SO(3) step; includes the otherwise ambiguous half-turn case."""
    relative = current.T @ target
    angle = float(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
    if angle <= max_angle:
        return target.copy(), angle
    if np.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((relative+relative.T)/2)
        axis = vectors[:, -1]
    else:
        axis = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0],
                         relative[1,0]-relative[0,1]])
        axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    step = np.eye(3)+np.sin(max_angle)*skew+(1-np.cos(max_angle))*(skew@skew)
    return current @ step, angle


class GraspStall(ValueError):
    """A valid observation permits another approach to this target."""


class TargetNotVisible(GraspStall):
    """The current RGB observation does not localize one cube."""


class MissedGrasp(GraspStall):
    """Feedback indicates the cube was left behind."""


class ScriptedLiftCube:
    """Open -> localize -> hover -> descend -> close -> lift -> hold.

    Recoverable grasp and tracking failures trigger a new attempt until reset.
    Holding is not a hardware emergency stop. No object pose or success input.
    """
    action_spec = ActionSpec(frequency_hz=30.0)
    close_dwell_ticks = 15
    aligned_close_height_m = .006

    def __init__(self, calibration, kinematics):
        self.c = validate_calibration(calibration)
        self.adaptive_orientation = calibration.get('adaptive_orientation', False)
        if not isinstance(self.adaptive_orientation, bool):
            raise ValueError('adaptive_orientation must be boolean')
        self.reference_rotations = {key: self.c[key].copy()
                                    for key in ('grasp_rotation', 'lift_rotation')}
        if self.adaptive_orientation and self.c.get("grasp_presets"):
            raise ValueError("adaptive_orientation and grasp_presets select different planners")
        self.kin = kinematics
        self.nominal_open = self.c["open_motor"]
        self.reset()

    def reset(self):
        self.phase = "open"
        self.travel_gain = 1.
        self.failure_reason = None
        self.last = None
        self.target = None
        self.pixel = None
        self.stable = self.ticks = self.settled = 0
        self.telemetry = {}
        self.grasp_motor = self.c["close_motor"]
        self.contact_motor = None
        self.previous_jaw = None
        self.contact_stable = 0
        self.loaded_close_ticks = 0
        self.jaw_blocked = False
        self.retries = 0
        self.retry_reason = None
        self.resume_loaded = False
        self.empty_grasp = False
        self.retry_xy = 0.
        self.face_yaw = self.planned_tilt = None
        self.face_pair = self.next_face_pair = None
        self.motion_anchor = None
        self.miss_count = 0
        self.c["open_motor"] = self.nominal_open

    def _urdf(self, motor):
        return motor_to_degrees(motor)[:5] * self.c["joint_sign"] + self.c["joint_offset_deg"]

    def _pose(self, motor):
        pose = _array(self.kin.forward_kinematics(self._urdf(motor)), (4, 4), "FK")
        return pose

    def _inside(self, xyz):
        if np.any(xyz < self.c["workspace"][0]) or np.any(xyz > self.c["workspace"][1]):
            raise ValueError("TCP target outside calibrated workspace")

    def _solve(self, xyz, measured, grip, rotation=None):
        self._inside(xyz)
        pose = np.eye(4)
        if rotation is None:
            rotation = self.c["lift_rotation"] if xyz[2] >= self.c["lift_z"]-.0001 else self.c["grasp_rotation"]
        pose[:3, :3] = rotation
        pose[:3, 3] = xyz
        q = self._urdf(measured)
        # Repeated calls also work with LeRobot v0.4.2's one-iteration solver.
        for _ in range(20):
            q = _array(self.kin.inverse_kinematics(q, pose, orientation_weight=1.0), (5,), "IK")
            actual = _array(self.kin.forward_kinematics(q), (4, 4), "FK")
            angle = np.arccos(np.clip((np.trace(actual[:3, :3].T @ pose[:3, :3])-1)/2, -1, 1))
            if np.linalg.norm(actual[:3, 3] - xyz) <= .003 and angle <= .15:
                break
        else:
            raise GraspStall("IK residual exceeds 3mm / 0.15rad")
        degrees = motor_to_degrees(measured).copy()
        degrees[:5] = (q - self.c["joint_offset_deg"]) / self.c["joint_sign"]
        result = degrees_to_motor(degrees)
        result[5] = grip
        return validate_action(result, self.action_spec, clip=False)

    def _transition(self, phase):
        self.phase, self.ticks, self.settled = phase, 0, 0
        if phase == "close":
            self.contact_motor = None
        self.motion_anchor = None
        self.miss_count = 0

    def _retry(self, reason, loaded=False):
        self.retry_reason = str(reason)
        self.retries += 1
        self.empty_grasp = isinstance(reason, MissedGrasp)
        if self.empty_grasp:
            self.jaw_blocked = False
            loaded = False
        if self.target is None and self.phase == "localize":
            self.stable, self.pixel = 0, None
            self._transition("localize")
            return
        if self.c.get("hold_contact_on_lift") and self.phase == "close" and self.jaw_blocked:
            loaded = True
            self._begin_lift()
        self.resume_loaded = loaded
        self.open_on_retreat = self.empty_grasp or self.phase in ("descend", "close")
        self._transition("unload")

    def _recover(self, q):
        """Unload the tracking error, retreat vertically, then reopen and relocalize."""
        self.ticks += 1
        if self._stalled(q, 30):
            raise GraspStall("Recovery motion stalled; replanning from measured joints")
        if self.phase == "unload":
            if self.ticks == 1:
                self.unload_motor = np.clip(q[:5], self.action_spec.low[:5], self.action_spec.high[:5])
            goal = self.last.copy()
            goal[:5] = self.unload_motor
            if self.resume_loaded and self.jaw_blocked and self.c.get("hold_contact_on_lift"):
                goal[5] = self.grasp_motor
            if np.max(np.abs(goal[:5]-self.last[:5])) <= 1 and np.max(np.abs(goal[:5]-q[:5])) <= 8:
                if self.resume_loaded:
                    self._transition("lift")
                else:
                    pose = self._pose(q)
                    self._inside(pose[:3, 3])
                    if self.empty_grasp and pose[2, 3] >= self.c["hover_z"]-.003:
                        self._transition("reopen")
                        return goal
                    self.retreat_pose = pose.copy()
                    self.retreat_pose[2, 3] = min(self.c["workspace"][1, 2], pose[2, 3]+.018)
                    self._transition("retreat")
            return goal
        if self.phase == "retreat":
            reference = self._pose(self.last)
            delta = self.retreat_pose[:3, 3]-reference[:3, 3]
            point = reference[:3, 3]+delta*min(1., .003*self.travel_gain/max(np.linalg.norm(delta), 1e-9))
            goal = self._solve(point, self.last, self.last[5], self.retreat_pose[:3, :3])
            if self.open_on_retreat:
                # Open in parallel without letting the full jaw travel slow the arm.
                goal[5] = min(self.last[5]+1., 100., self.nominal_open+8.)
            if self._pose(q)[2, 3] >= self.retreat_pose[2, 3]-.005:
                self._transition("reopen")
            return goal
        goal = self.last.copy()
        goal[5] = min(100., self.nominal_open+8.)
        if abs(q[5]-goal[5]) < 2 or self.c.get("parallel_approach"):
            self.c["open_motor"] = goal[5]
            self.pixel = self.previous_jaw = None
            self.grasp_motor = self.c["close_motor"]
            self.stable = self.contact_stable = 0
            self.loaded_close_ticks = 0
            self.jaw_blocked = False
            self.retry_xy = .003 if self.retries % 2 else -.003
            self._transition("localize")
        return goal

    def _aim(self, point):
        point = point.copy()
        point[0] += self.retry_xy
        return point

    def _jaw_ready(self, q):
        return not self.c.get("parallel_approach") or q[5] >= self.c["open_motor"]-2

    def _travel_gain(self, q):
        speed = self.c.get("arm_travel_speed", 1.)
        if speed == 1.:
            return 1.
        if self.phase not in ("hover", "retreat"):
            if self.phase != "lift" or not self.jaw_blocked or self._pose(q)[2,3] < self.c["grasp_z"]+.012:
                return 1.
        sent = getattr(self, "sent", None)
        previous = self.last if sent is None else sent
        lag = np.max(np.abs(previous[:5]-q[:5]))
        return 1.+(speed-1.)*float(np.clip((7.-lag)/3., 0., 1.))

    def _limit_motion(self, delta):
        if self.travel_gain == 1.:
            return delta*min(1., 1./max(float(np.max(np.abs(delta))), 1e-9))
        arm = delta[:5]
        return np.r_[arm*min(1., self.travel_gain/max(float(np.max(np.abs(arm))), 1e-9)),
                     np.clip(delta[5], -1., 1.)]

    def _begin_lift(self):
        self.target[2] = self.c["lift_z"]
        self._transition("lift")

    def _plan_lift(self, xyz, seed):
        self._solve(np.r_[xyz[:2], self.c["lift_z"]], seed,
                    self.c["open_motor"], rotation=self.c["lift_rotation"])

    def _cube_still_at_pick(self, image):
        if self.pixel is None:
            return False
        try:
            pixel = red_centroid(image, self.c["image_shape"])
        except ValueError:
            return False
        return np.linalg.norm(pixel-self.pixel) <= 2*self.c["image_shape"][0]/64

    def _lost_after_lift(self, image, q):
        return self._cube_still_at_pick(image)

    def _stalled(self, joints, interval):
        if self.ticks % interval:
            return False
        previous = self.motion_anchor
        self.motion_anchor = joints.copy()
        return previous is not None and np.max(np.abs(joints-previous)) < .4

    def _watch_grasp(self, image, q):
        if self.phase in ("hover", "descend", "lift") and self.target is not None:
            if self._stalled(q[:5], 20):
                raise GraspStall("Arm stopped progressing during "+self.phase)
        if self.phase in ("open", "close") and not self.jaw_blocked and self._stalled(q[5:], 30):
            raise GraspStall("Jaw stopped progressing during "+self.phase)
        if self.phase in ("lift", "hold") and self.ticks >= 6 and self.ticks % 3 == 0:
            clearance = .030 if self.retries and self.c.get("follow_retry_close") else .055
            high = self._pose(q)[2, 3] >= self.c["grasp_z"]+clearance
            self.miss_count = self.miss_count+1 if high and self._lost_after_lift(image, q) else 0
            if self.miss_count >= 3:
                raise MissedGrasp("Object visible apart from fingers after lift")
        if self.phase == "hold" and self.target is not None and self._stalled(q, 30):
            raise GraspStall("Lift waypoint settled while the episode is still active")

    def act(self, observation, deterministic=False):
        # Deliberately do not inspect metadata, padded dq/EE, or previous labels.
        measured = _array(observation.proprio[:6], (6,), "measured motor positions")
        # Sensor feedback is not a requested target: contact/solver compliance
        # may put it slightly beyond a stop. Keep raw feedback; validate every
        # outgoing target below and retain measured-state tracking protection.
        if self.last is None:
            self.last = measured.copy()
        requested = self.last.copy()
        try:
            self.travel_gain = self._travel_gain(measured)
            if self.failure_reason is None:
                if self.phase == "open" and self.c.get("parallel_approach"):
                    position = self._pose(measured)[:3, 3]
                    self._inside(position)
                    if position[2] < self.c["hover_z"]-.003:
                        raise ValueError("Reset TCP is below safe hover height")
                    self._transition("localize")
                opening = self.c.get("parallel_approach") and self.phase in ("localize", "hover", "reopen")
                if self.phase in ("unload", "retreat", "reopen"):
                    requested = self._recover(measured)
                else:
                    self._watch_grasp(observation.image, measured)
                    if (self.retries and self.c.get("follow_retry_close") and self.jaw_blocked
                            and self.phase in ("close", "lift", "hold")):
                        # Take up slack as the jaws settle, keeping bounded position lead.
                        if self.contact_motor is None:
                            self.contact_motor = self.grasp_motor
                        floor = max(self.c["close_motor"], self.contact_motor-1.5)
                        self.grasp_motor = min(self.grasp_motor, max(floor, measured[5]-3.))
                    requested = self._step(observation.image, measured)
                    if self.phase == "hold" and self.jaw_blocked and self.retries and self.c.get("follow_retry_close"):
                        requested[5] = self.grasp_motor
                if opening:
                    # Open independently without scaling down all five arm joints.
                    jaw_step = np.clip(self.c["open_motor"]-self.last[5], -1., 1.)
                    requested[5] = self.last[5]+jaw_step
        except GraspStall as exc:
            self.travel_gain = 1.
            self._retry(exc, loaded=self.phase == "lift" and not isinstance(exc, MissedGrasp)
                        and measured[5] > self.c["close_motor"]+2)
            requested = self.last.copy()
        except ValueError as exc:
            self.failure_reason = str(exc)
            self.phase = "failed"
        # Pace loaded motion against the measured joints before applying slew limits.
        if self.phase == "lift" and self.jaw_blocked and self.c.get("pace_loaded_lift"):
            requested[:5] = np.clip(requested[:5], measured[:5]-7.5, measured[:5]+7.5)
        delta = requested - self.last
        # One common fraction preserves the IK joint-space direction.
        sent = self.last + self._limit_motion(delta)
        if self.c.get("parallel_approach") and self.phase in ("hover","localize","reopen"):
            sent[5] = self.last[5]+np.clip(delta[5],-1.,1.)
        if self.failure_reason is None and self.phase != "unload" and np.any(np.abs(sent[:5] - measured[:5]) > 8):
            if self.phase == "retreat" and self.c.get("probe_loaded_close"):
                self._transition("unload")
                self.resume_loaded = False
            elif self.phase in ("hover", "descend", "close", "lift", "reopen", "retreat", "localize", "hold"):
                self._retry("Arm tracking lag exceeds 8 motor units",
                            loaded=self.phase == "lift" and measured[5] > self.c["close_motor"]+2)
            else:
                self.failure_reason, self.phase = "Arm tracking lag exceeds 8 motor units", "failed"
            sent = self.last.copy()
        self.last = validate_action(sent, self.action_spec, clip=False).copy()
        self.telemetry = {"phase": self.phase, "failure_reason": self.failure_reason,
                          "cube_yaw_rad": self.face_yaw, "planned_grasp_tilt_deg": self.planned_tilt,
                          "pixel": None if self.pixel is None else self.pixel.tolist(),
                          "target_xyz": None if self.target is None else self.target.tolist(),
                          "requested_motor": requested.tolist(), "command_motor": self.last.tolist(),
                          "measured_motor": measured.tolist(), "jaw_contact_proxy": self.jaw_blocked,
                          "grasp_motor": self.grasp_motor, "retries": self.retries,
                          "retry_reason": self.retry_reason}
        return self.last.copy()

    def _localize(self, image):
        pixel = red_centroid(image, self.c["image_shape"])
        scale = self.c["image_shape"][0]/64
        self.stable = self.stable + 1 if self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*scale else 1
        self.pixel = pixel
        return pixel_to_base(pixel, self.c) if self.stable >= 3 else None

    def _select_orientations(self, point):
        from scipy.spatial.transform import Rotation

        yaw = Rotation.from_euler('z', np.arctan2(point[0], -point[1])).as_matrix()
        stages = [('grasp_rotation', (self.c['hover_z'], self.c['grasp_z'])),
                  ('lift_rotation', (self.c['lift_z'],))]
        for key, heights in stages:
            reference = self.reference_rotations[key]
            self.c[key] = reference
            radial = yaw @ reference
            candidates = [reference, radial]
            candidates.extend(radial @ Rotation.from_euler('y', angle, degrees=True).as_matrix()
                              for angle in (-12, -20, -30))
            positions = [np.r_[point[:2], z] for z in heights]
            for rotation in candidates:
                try:
                    targets = [self._solve(pos, self.last, self.c['open_motor'], rotation=rotation)
                               for pos in positions]
                except ValueError:
                    continue
                if all(np.max(np.abs(q[:5])) < 98 and np.linalg.norm(self._pose(q)[:3, 3]-pos) < .0025
                       for pos, q in zip(positions, targets)):
                    self.c[key] = rotation
                    break

    def _face_pose(self, point, measured):
        """Prefer vertical fingers; use the nearest reachable cube side normal."""
        if self.face_yaw is None:
            return None
        self.face_pair = None
        q = self._urdf(measured)
        radial = np.arctan2(point[0], -point[1])
        wrist = self._pose(measured)[:2, 0]
        normals = [np.array([np.cos(a), np.sin(a)]) for a in self.face_yaw + np.arange(4)*np.pi/2]
        candidates = [(tilt, side) for tilt in (0, 10, 20, 30) for side in range(4)]
        def preference(item):
            tilt, side = item
            angle = np.deg2rad(tilt)
            slope = np.array([-np.sin(radial), np.cos(radial)]) * np.sin(angle)
            normal = normals[side]
            return (self.next_face_pair is not None and side % 2 != self.next_face_pair,
                    tilt, round(abs(float(normal @ slope)), 8), -float(normal @ wrist))
        for tilt, side in sorted(candidates, key=preference):
            angle = np.deg2rad(tilt)
            z = np.array([-np.sin(angle)*np.sin(radial), np.sin(angle)*np.cos(radial), np.cos(angle)])
            normal = normals[side]
            x = np.r_[normal, -(normal @ z[:2])/z[2]]
            x /= np.linalg.norm(x)
            rotation = np.column_stack((x, np.cross(z,x), z))
            target = np.eye(4)
            target[:3,:3] = rotation
            for h in ("grasp_z", "hover_z"):
                target[:3,3] = [*point[:2], self.c[h]]
                joints = self.kin.inverse_kinematics(q, target, orientation_weight=1.)
                actual = self.kin.forward_kinematics(joints)
                if (np.linalg.norm(actual[:3,3]-target[:3,3]) > .003
                        or rotation_step(actual[:3,:3], rotation)[1] > .12):
                    break
            else:
                try:
                    hover = self._solve(target[:3,3], measured, self.c["open_motor"], rotation=rotation)
                    self._solve(np.r_[point[:2],self.c["grasp_z"]], hover, self.c["open_motor"], rotation=rotation)
                except ValueError:
                    continue
                self.c["grasp_rotation"] = rotation
                self.planned_tilt = tilt
                self.face_pair = side % 2
                return hover
        return None

    def _check_target(self, image):
        pixel = red_centroid(image, self.c["image_shape"])
        if np.linalg.norm(pixel-self.pixel) > 3*self.c["image_shape"][0]/64:
            raise GraspStall("Red target shifted before descent")

    def _step(self, image, q):
        self.ticks += 1
        c = self.c
        goal = self.last.copy()
        if self.phase == "open":
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            if pose[2, 3] < c["hover_z"] - .003:
                raise ValueError("Reset TCP is below safe hover height; no blind recovery motion")
            goal[5] = c["open_motor"]
            self.settled = self.settled + 1 if abs(q[5]-goal[5]) < 2 else 0
            if self.settled >= 3:
                self._transition("localize")
        elif self.phase == "localize":
            xyz = self._localize(image)
            if xyz is not None:
                xyz = self._aim(xyz)
                if self.adaptive_orientation:
                    self._select_orientations(xyz)
                if self.c.get("face_aligned_grasp"):
                    self.planned_tilt = None
                    self._face_pose(xyz, q)
                # Fail before descending if any complete waypoint is unreachable.
                seed = q
                for z, rotation in ((c["hover_z"], c["grasp_rotation"]),
                                    (c["grasp_z"], c["grasp_rotation"])):
                    try:
                        endpoint = self._solve(np.r_[xyz[:2], z], seed, c["open_motor"], rotation=rotation)
                    except GraspStall as exc:
                        raise GraspStall(f"Waypoint at {z:.3f} m: {exc}") from exc
                    if self.face_pair is not None:
                        seed = endpoint
                self._plan_lift(xyz, seed)
                self.target = np.r_[xyz[:2], c["hover_z"]]
                self._transition("hover")
        elif self.phase in ("hover", "descend", "lift"):
            if self.phase == "hover":
                self._check_target(image)
            current_pose = self._pose(q)
            current = current_pose[:3, 3]
            self._inside(current)
            delta = self.target - current
            # Advance the reference from the last sent target, not lagging
            # feedback: otherwise gravity sag repeatedly drags the path down.
            reference = self._pose(self.last)
            reference_delta = self.target - reference[:3, 3]
            intermediate = reference[:3, 3] + reference_delta * min(
                1.0, .004*self.travel_gain / max(np.linalg.norm(reference_delta), 1e-9))
            grip = self.grasp_motor if self.phase == "lift" else c["open_motor"]
            target_rotation = c["lift_rotation"] if self.phase == "lift" else c["grasp_rotation"]
            orientation, _ = rotation_step(reference[:3, :3], target_rotation, .02*self.travel_gain)
            _, remaining_angle = rotation_step(current_pose[:3, :3], target_rotation)
            goal = self._solve(intermediate, self.last, grip, rotation=orientation)
            reached = (np.linalg.norm(delta) <= .003 and remaining_angle <= .15
                       and np.max(np.abs(goal[:5]-q[:5])) <= 2)
            if self.phase == "descend" and c.get("relaxed_close_window", c.get("face_aligned_grasp", False)):
                height = self.aligned_close_height_m if self.planned_tilt is not None else .006
                reached |= (np.linalg.norm(delta[:2]) <= .005 and -height <= delta[2] <= .003
                            and remaining_angle <= .15 and np.max(np.abs(goal[:5]-q[:5])) <= 3)
            if self.phase == "hover":
                reached = reached and self._jaw_ready(q)
            self.settled = self.settled + 1 if reached else 0
            if self.settled >= 3:
                if self.phase == "hover":
                    self.target[2] = c["grasp_z"]
                    self._transition("descend")
                elif self.phase == "descend":
                    self._transition("close")
                else:
                    self._transition("hold")  # Not a declaration of grasp success.
        elif self.phase == "close":
            goal[5] = c["close_motor"]
            if c.get("contact_close", False):
                if not self.jaw_blocked:
                    if c.get("probe_loaded_close"):
                        arm_lag = np.max(np.abs(getattr(self, "sent", self.last)[:5]-q[:5]))
                        resisting = (c["close_motor"]+3 < q[5] < self.nominal_open-8
                                     and q[5]-self.last[5] >= 1.3 and arm_lag >= 6.8)
                        self.loaded_close_ticks = self.loaded_close_ticks+1 if resisting else 0
                        if self.loaded_close_ticks >= 2:
                            self.grasp_motor = max(c["close_motor"], q[5]-2.)
                            self.jaw_blocked = True
                            goal[5] = self.grasp_motor
                            if not (self.retries and c.get("follow_retry_close")):
                                self._begin_lift()
                            return goal
                    blocked = (self.previous_jaw is not None and abs(q[5]-self.previous_jaw) < .1
                               and q[5]-self.last[5] >= 1.5)
                    self.contact_stable = self.contact_stable+1 if blocked else 0
                    self.previous_jaw = float(q[5])
                    # Position-feedback proxy, not a force/contact sensor. Limit
                    # jaw lead instead of driving harder against an obstruction.
                    goal[5] = max(c["close_motor"], float(q[5])-2.)
                    if self.contact_stable >= 3:
                        # Keep a bounded position preload under contact throughout lift.
                        preload = c.get("contact_preload_motor", 0.)
                        self.grasp_motor = max(c["close_motor"], float(self.last[5])-preload)
                        self.jaw_blocked = True
                if self.jaw_blocked:
                    goal[5] = self.grasp_motor
            # Contact can prevent reaching the commanded jaw position. Dwell only
            # after full-close or the optional feedback proxy, not stage entry.
            ready = (self.jaw_blocked or
                     abs(self.last[5]-c["close_motor"]) < 1e-4)
            self.settled = self.settled + 1 if ready else 0
            if self.settled >= self.close_dwell_ticks:
                if (c.get("contact_close") and not self.jaw_blocked
                        and q[5] <= c["close_motor"]+2):
                    raise MissedGrasp("Jaws closed fully without an object")
                self._begin_lift()
        return goal


class SteadyBiasScriptedLiftCube(ScriptedLiftCube):
    """Optional bounded steady-state correction; only RGB/q feedback is used.

    Nominal path and sent target stay separate. Integrate only at a stationary
    reference, never on the jaw; bounds, slew and tracking guards still apply.
    """
    def reset(self):
        super().reset()
        self.bias = np.zeros(5)
        self.previous_nominal = self.sent = None
        self.stationary = 0

    def _retry(self, reason, loaded=False):
        super()._retry(reason, loaded)
        self.bias.fill(0.)
        self.stationary = 0
        self.last = self.sent.copy()

    def act(self, observation, deterministic=False):
        measured = np.asarray(observation.proprio[:6], dtype=float)
        if self.sent is None:
            self.sent = measured.copy()
        nominal = super().act(observation, deterministic=deterministic)
        if self.failure_reason is None:
            stable = (self.previous_nominal is not None
                      and np.max(np.abs(nominal[:5]-self.previous_nominal[:5])) <= .02
                      and self.phase in ('hover', 'descend', 'lift', 'retreat'))
            self.stationary = self.stationary+1 if stable else 0
            if self.stationary >= 3:
                self.bias += .1*(nominal[:5]-measured[:5])
            lower = np.maximum(-3, self.action_spec.low[:5]-nominal[:5])
            upper = np.minimum(3, self.action_spec.high[:5]-nominal[:5])
            self.bias = np.clip(self.bias, lower, upper)
            corrected = nominal.copy()
            corrected[:5] += self.bias
            if self.phase == "lift" and self.jaw_blocked and self.c.get("pace_loaded_lift"):
                corrected[:5] = np.clip(corrected[:5], measured[:5]-7.5, measured[:5]+7.5)
            try:
                validate_action(corrected, self.action_spec, clip=False)
                delta = corrected-self.sent
                proposed = self.sent+self._limit_motion(delta)
                if self.c.get("parallel_approach") and self.phase in ("hover", "localize", "reopen"):
                    proposed[5] = self.sent[5]+np.clip(delta[5], -1., 1.)
                if self.phase != 'unload' and np.any(np.abs(proposed[:5]-measured[:5]) > 8):
                    if self.phase == 'open':
                        raise ValueError('Compensated target tracking lag exceeds 8 motor units')
                    self._retry('Compensated target tracking lag exceeds 8 motor units',
                                loaded=self.phase == 'lift' and measured[5] > self.c["close_motor"]+2)
                else:
                    self.sent = validate_action(proposed, self.action_spec, clip=False).copy()
            except ValueError as exc:
                self.failure_reason, self.phase = str(exc), 'failed'
        self.previous_nominal = nominal.copy()
        self.telemetry.update(phase=self.phase, failure_reason=self.failure_reason,
            retries=self.retries, retry_reason=self.retry_reason,
            nominal_motor=nominal.tolist(), steady_bias_motor=self.bias.tolist(),
            command_motor=self.sent.tolist())
        return self.sent.copy()
