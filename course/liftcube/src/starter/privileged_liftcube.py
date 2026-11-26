"""LiftCube controller with explicit cube pose callbacks in base coordinates.

It shares joint execution with the RGB controller. Object pose is read through
the callbacks and stays outside the public Observation.
"""
import numpy as np

from .scripted_liftcube import GraspStall, MissedGrasp, SteadyBiasScriptedLiftCube, _array, rotation_step


def read_cube_in_base(env):
    from scipy.spatial.transform import Rotation
    raw = env._unwrapped
    robot = raw.scene['robot']
    index = list(robot.body_names).index('base')
    position = robot.data.body_pos_w[0, index].detach().cpu().numpy()
    quat = robot.data.body_quat_w[0, index].detach().cpu().numpy()
    cube = raw.scene['cube'].data.root_pos_w[0].detach().cpu().numpy()
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix().T @ (cube-position)


def read_cube_yaw_in_base(env):
    from scipy.spatial.transform import Rotation
    scene = env._unwrapped.scene
    robot = scene['robot']
    base = robot.data.body_quat_w[0,robot.body_names.index('base')].detach().cpu().numpy()
    cube = scene['cube'].data.root_quat_w[0].detach().cpu().numpy()
    relative = Rotation.from_quat(base[[1,2,3,0]]).inv()*Rotation.from_quat(cube[[1,2,3,0]])
    normal = relative.as_matrix()[:,0]
    return float(np.arctan2(normal[1],normal[0]))


class PrivilegedLiftCube(SteadyBiasScriptedLiftCube):
    # At 20° tilt the fixed fingertip can reach the table before the TCP reaches its target.
    aligned_close_height_m = .008

    def __init__(self, calibration, kinematics, read_cube=None, read_yaw=None):
        if calibration.get('grasp_presets') or calibration.get('plan_while_opening'):
            raise ValueError('Privileged localization requires disabled visual presets and plan_while_opening')
        self.read_cube = read_cube
        self.read_yaw = read_yaw
        super().__init__(calibration, kinematics)


    def reset(self):
        super().reset()
        self.anchor = None
        self.pick_offset = None
        self.slip_ticks = 0
        self.regrasp_offset = None
        self.closing_drift = np.zeros(2)
        self.hover_motor = None
        self.lift_destination = self.transport_rotation = None
        self.clearing_cube = False

    def _plan_lift(self, xyz, seed):
        self.lift_destination = None
        self.clearing_cube = False
        if not self.c.get("face_aligned_grasp"):
            return super()._plan_lift(xyz, seed)
        destination = np.r_[xyz[:2], self.c["lift_z"]]
        rotation = self.c["lift_rotation"].copy()
        try:
            super()._plan_lift(xyz, seed)
            if self.next_face_pair is None:
                return
        except GraspStall:
            from scipy.spatial.transform import Rotation
            inward = -xyz[:2] / max(np.linalg.norm(xyz[:2]), 1e-8)
            radial = Rotation.from_euler("z", np.arctan2(xyz[0], -xyz[1])).as_matrix()
            radial_rotation = radial @ self.reference_rotations["lift_rotation"]
            # A nearby cube can be *too close* for upright transport. Try the
            # calibrated radial wrist first, then search both radial directions.
            options = [(distance, rotation, seed) for distance in (.015, .030, .045, .060)]
            options.extend((distance, radial_rotation, initial)
                           for distance in (0., .015, -.015, .030, -.030, .045, -.045, .060, -.060)
                           for initial in (seed, self.last))
            for distance, orientation, initial in options:
                destination[:2] = xyz[:2] + distance * inward
                if np.any(destination < self.c["workspace"][0]) or np.any(destination > self.c["workspace"][1]):
                    continue
                try:
                    self._solve(destination, initial, self.c["open_motor"], rotation=orientation)
                    rotation = orientation
                    break
                except GraspStall:
                    continue
            else:
                raise GraspStall("No reachable elevated transport waypoint")
        self._solve(np.r_[xyz[:2], self.c["hover_z"]], seed,
                    self.c["open_motor"], rotation=self.c["grasp_rotation"])
        self.lift_destination = destination
        self.transport_rotation = rotation

    def _begin_lift(self):
        super()._begin_lift()
        if self.lift_destination is not None:
            self.target[2] = self.c["hover_z"]
            self.c["lift_rotation"] = self.c["grasp_rotation"].copy()
            self.clearing_cube = True

    def _point(self):
        if not callable(self.read_cube):
            raise ValueError('Explicit privileged cube reader is required')
        point = _array(self.read_cube(), (3,), 'privileged cube in base')
        # The cube center may be below the TCP workspace's safe lower height.
        # Actual TCP waypoint heights still pass the inherited _solve guard.
        if np.any(point[:2] < self.c['workspace'][0][:2]) or np.any(point[:2] > self.c['workspace'][1][:2]):
            raise ValueError('Privileged target outside calibrated XY workspace')
        return point

    def _localize(self, image):
        point = self._point()
        self.stable = self.stable+1 if self.anchor is not None and np.linalg.norm(point-self.anchor) <= .003 else 1
        self.anchor = point.copy()
        if self.stable < 3:
            return None
        if self.c.get('face_aligned_grasp'):
            if self.read_yaw is None:
                raise ValueError('Face-aligned privileged grasp needs the cube yaw in base coordinates')
            self.face_yaw = float(self.read_yaw())
        return point

    def _check_target(self, image):
        point = self._point()
        if self.anchor is None or np.linalg.norm(point-self.anchor) > .005:
            raise GraspStall('Target moved before descent')

    def _step(self, image, q):
        if (self.clearing_cube and self.phase in ("lift", "hold")
                and self._pose(q)[2, 3] >= self.c["grasp_z"]+.025):
            self.clearing_cube = False
            self.target = self.lift_destination.copy()
            self.c["lift_rotation"] = self.transport_rotation.copy()
            self._transition("lift")
        if (self.c.get("retry_dropped_cube") and self.phase == "hover" and self.anchor is not None
                and np.linalg.norm(self._point()-self.anchor) > .005
                and self._pose(q)[2, 3] >= self.c["hover_z"]-.006):
            self.anchor = None
            self.stable = 0
            self._transition("localize")
        if self.c.get("joint_hover_approach") and self.phase == "hover":
            self.ticks += 1
            self._check_target(image)
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            _, angle = rotation_step(pose[:3, :3], self.c["grasp_rotation"])
            arrived = np.linalg.norm(self.target-pose[:3, 3]) < .003 and angle < .15
            self.settled = self.settled+1 if arrived and self._jaw_ready(q) else 0
            if self.settled >= 3:
                self.target[2] = self.c["grasp_z"]
                self._transition("descend")
            delta = self.hover_motor-self.last
            delta = self._limit_motion(delta)
            reference = self._pose(self.last)
            for _ in range(12):
                following = self._pose(self.last+delta)
                _, turn = rotation_step(reference[:3, :3], following[:3, :3])
                travel = np.linalg.norm(following[:3, 3]-reference[:3, 3])
                if travel <= .004*self.travel_gain and turn <= .02*self.travel_gain+.000001:
                    return self.last+delta
                delta *= .99*self.travel_gain*min(.004/max(travel, 1e-9), .02/max(turn, 1e-9))
            raise GraspStall("Cannot advance joint approach to hover")
        was_localizing = self.phase == "localize"
        goal = super()._step(image, q)
        if was_localizing and self.phase == "hover" and self.c.get("joint_hover_approach"):
            self.hover_motor = self._solve(self.target, q, self.c["open_motor"], self.c["grasp_rotation"])
        return goal

    def _watch_grasp(self, image, q):
        if self.c.get("retry_dropped_cube"):
            if self.phase == "close" and self.pick_offset is None:
                self.pick_offset = self._point() - self._pose(q)[:3, 3]
                self.closing_drift.fill(0.)
            if self.pick_offset is not None and self.phase in ("close", "lift", "hold"):
                offset = self._point() - self._pose(q)[:3, 3]
                lost = np.linalg.norm(offset-self.pick_offset) > .022 and self._cube_still_at_pick(image)
                if self.phase == "close" and not lost and self.anchor is not None:
                    self.closing_drift = self._point()[:2]-self.anchor[:2]
                self.slip_ticks = self.slip_ticks+1 if lost else 0
                if self.slip_ticks >= 2:
                    self.jaw_blocked = False
                    self.pick_offset = None
                    self.slip_ticks = 0
                    raise MissedGrasp("Cube left the fingers and returned to the table")
            if self.phase in ("localize", "hover", "descend"):
                self.pick_offset = None
                self.slip_ticks = 0
        super()._watch_grasp(image, q)

    def _cube_still_at_pick(self, image):
        return self.anchor is not None and self._point()[2] <= self.anchor[2]+.008

    def _retry(self, reason, loaded=False):
        previous_phase = self.phase
        super()._retry(reason, loaded)
        if (self.c.get("face_aligned_grasp") and not self.resume_loaded
                and previous_phase in ("localize", "hover", "descend", "close", "lift", "hold")
                and self.face_pair is not None):
            self.next_face_pair = 1 - self.face_pair
            self.open_on_retreat = True
        if self.c.get("retry_dropped_cube") and isinstance(reason, MissedGrasp):
            self.resume_loaded = False
            self.open_on_retreat = True
            drift = self.closing_drift
            self.regrasp_offset = -drift*min(.5, .004/max(np.linalg.norm(drift), 1e-9))

    def _recover(self, q):
        previous = self.phase
        goal = super()._recover(q)
        if self.next_face_pair is not None and previous == "unload" and self.phase == "retreat":
            if self._pose(q)[2, 3] >= self.c["hover_z"]-.003:
                self._transition("reopen")
            else:
                self.retreat_pose[2, 3] = self.c["hover_z"]
        return goal

    def _aim(self, point):
        if self.regrasp_offset is None:
            return super()._aim(point)
        adjusted = point.copy()
        adjusted[:2] += self.regrasp_offset
        return adjusted

    def act(self, observation, deterministic=False):
        action = super().act(observation, deterministic)
        self.telemetry.update(privileged_teacher=True,
            cube_base=None if self.anchor is None else self.anchor.tolist())
        return action
