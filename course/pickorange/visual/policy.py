"""Vision-only object localization with robot-model IK and joint feedback.

The actor has no environment handle. Geometry helpers read only the robot USD.
Gravity bias is a robot-model auxiliary supplied by the runner, not object state.
"""
from pathlib import Path
import sys
import numpy as np
from perception import Detector

EXPERT = Path(__file__).resolve().parents[1] / 'expert'
if str(EXPERT) not in sys.path:
    sys.path.insert(0, str(EXPERT))
from diameter_teacher import DiameterKinematics
from finger_geometry import FingerGeometry


class VisualPolicy:
    home = np.deg2rad([0, -100, 90, 50, 0, -10])
    phases = ('scan', 'approach', 'descend', 'close', 'lift', 'carry', 'lower', 'release', 'retreat', 'home')

    def __init__(self, asset, calibration):
        self.detector = Detector(calibration)
        self.kin = DiameterKinematics(asset)
        self.geometry = FingerGeometry(asset)
        self.kin.fixed, self.kin.moving = self.geometry.pad_pair(-.095)
        self.phase = 'scan'
        self.age = self.stable = self.completed = 0
        self.events = []
        self.integral = np.zeros(5)
        self.previous = None
        self.anchor = self.target = self.plate = None
        self.opening = .75
        self.failure = None
        self.close_history = []
        self.detections = None
        self.attempts = 0

    def tcp(self, q):
        self.kin.set_opening(float(q[5]))
        return self.kin.fk(q[:5])[1]

    def enter(self, phase, q):
        self.events.append(dict(phase=phase, completed=self.completed, from_phase=self.phase,
                                previous_age=self.age, tcp=self.tcp(q).tolist()))
        self.phase, self.age, self.stable = phase, 0, 0
        if phase == 'approach':
            self.target = self.anchor + [0, 0, .080]
        elif phase == 'descend':
            self.waypoint = self.tcp(q).copy()
            self.target = self.anchor.copy()
        elif phase == 'close':
            self.close_history = []
            self.target = self.anchor.copy()
        elif phase == 'lift':
            self.target = self.anchor + [0, 0, .100]
            self.loaded_opening = float(q[5])
        elif phase == 'carry':
            self.kin.desired_axis = None
            self.target = self.drop + [0, 0, .115]
        elif phase == 'lower':
            self.target = self.drop + [0, 0, .065]
        elif phase == 'retreat':
            self.target = self.drop + [0, 0, .125]
        elif phase in ('scan', 'home'):
            self.integral[:] = 0

    def plan(self, q):
        fruit = [d for d in self.detections['oranges']
                 if np.linalg.norm(np.asarray(d['point'])[:2]-self.plate[:2]) > .105]
        if not fruit:
            return False
        fruit.sort(key=lambda d: np.linalg.norm(d['point'][:2]))
        best = None
        for detection in fruit:
            point = np.array(detection['point'])
            yaw = np.arctan2(point[1], point[0])+np.pi/2
            for offset in (60, -60, 30, -30, 0, -90):
                axis = np.array([np.cos(yaw+np.deg2rad(offset)), np.sin(yaw+np.deg2rad(offset)), 0])
                self.kin.desired_axis = axis
                self.kin.set_opening(self.opening)
                grasp = self.kin.solve(point, q[:5])
                if grasp['position_error_m'] > .006 or grasp['tool_axis_error'] > .18:
                    continue
                grasp_frame, _, _ = self.kin.fk(grasp['q'])
                # Upright fingers avoid a nearly singular elbow branch while
                # the unilateral jaw changes the pad midpoint during closure.
                if grasp_frame[2, 2] < .93 or abs(grasp['q'][3]) > 1.58:
                    continue
                approach = self.kin.solve(point+[0, 0, .080], grasp['q'])
                if approach['position_error_m'] > .008:
                    continue
                margin = np.inf
                for f in np.linspace(0, 1, 7):
                    t, _, _ = self.kin.fk((1-f)*approach['q']+f*grasp['q'])
                    clearance = self.geometry.clearance(t, self.opening, point, [.026, .026, .022],
                                                         self.detector.calibration.table_z)
                    margin = min(margin, clearance['table_clearance_m'])
                cost = grasp['position_error_m']+.04*grasp['tool_axis_error']+.002*np.linalg.norm(approach['q']-q[:5])
                cost += .04*(1-grasp_frame[2, 2])
                cost += 10*max(0, .002-margin)
                if best is None or cost < best[0]:
                    best = cost, point, axis, dict(detection=detection, offset_deg=offset,
                        table_clearance_m=float(margin), grasp_error_m=grasp['position_error_m'])
            if best is not None and best[0] < .025:
                break
        if best is None:
            return False
        _, self.anchor, self.kin.desired_axis, detail = best
        self.drop = self.plate.copy()
        # Three interior positions; based on image-estimated plate center only.
        self.drop[:2] += np.array([-.025, (self.completed % 3-1)*.042])
        self.events.append(dict(plan=detail, anchor=self.anchor.tolist(), drop=self.drop.tolist()))
        self.attempts += 1
        self.enter('approach', q)
        return True

    def update(self, rgb, q, gravity_bias):
        q, gravity_bias = np.asarray(q, float), np.asarray(gravity_bias, float)
        if q.shape != (6,) or gravity_bias.shape != (5,) or not np.isfinite(np.r_[q, gravity_bias]).all():
            raise ValueError('Expected finite ordered q6 and robot gravity bias5')
        if self.previous is None:
            self.previous = q.copy()
        self.age += 1
        phase = self.phase
        self.detections = self.detector.detect(rgb) if phase == 'scan' else self.detections
        if phase == 'scan':
            if self.age >= 75:
                if self.detections['plate'] is not None:
                    self.plate = np.asarray(self.detections['plate']['point'])
                    if self.attempts >= 5:
                        self.enter('home', q)
                    elif not self.plan(q) and self.age > 100:
                        self.failure = 'No reachable fruit outside plate detected' if self.completed < 3 else None
                        self.enter('home', q)
                elif self.age > 180:
                    self.failure = 'Plate not detected in RGB'
                    self.enter('home', q)
        elif phase in ('approach', 'descend', 'lift', 'carry', 'lower', 'retreat'):
            distance = float(np.linalg.norm(self.tcp(q)-self.target))
            self.stable = self.stable+1 if distance < .008 else 0
            cap = {'approach':150, 'descend':150, 'lift':120, 'carry':150, 'lower':100, 'retreat':100}[phase]
            if (self.stable >= 4 and self.age >= 12) or self.age >= cap:
                if self.age >= cap:
                    self.events.append(dict(deadline=phase, distance_m=distance))
                next_phase = dict(approach='descend', descend='close', lift='carry', carry='lower',
                                  lower='release', retreat='scan')[phase]
                if phase == 'retreat':
                    self.completed += 1
                self.enter(next_phase, q)
        elif phase == 'close':
            self.close_history.append(float(q[5]))
            stalled = len(self.close_history) > 15 and np.ptp(self.close_history[-12:]) < .008
            if self.age >= 75 or (self.age > 30 and stalled):
                self.enter('lift', q)
        elif phase == 'release' and self.age > 30:
            self.enter('retreat', q)

        phase = self.phase
        if phase in ('scan', 'home'):
            goal = self.home[:5]
            grip = self.home[5] if phase == 'home' else self.opening
        else:
            self.kin.set_opening(float(q[5]))
            target = self.target.copy()
            if phase == 'descend':
                if np.linalg.norm(self.tcp(q)-self.waypoint) < .010:
                    delta = self.target-self.waypoint
                    self.waypoint += delta*min(1, .002/max(np.linalg.norm(delta), 1e-9))
                target = self.waypoint
            solved = self.kin.solve(target, q[:5], local=True)
            goal = solved['q']
            if phase in ('close', 'lift', 'carry', 'lower'):
                # Bound commanded compression against measured jaw position.
                grip = max(.15, float(q[5])-.020)
            else:
                grip = self.opening
        raw = goal+np.clip(gravity_bias, -.35, .35)+self.integral
        if np.max(abs(raw-self.previous[:5])) < .015 and phase != 'close':
            self.integral = np.clip(self.integral+.04*(goal-q[:5]), -.12, .12)
        wanted = np.r_[goal+np.clip(gravity_bias, -.35, .35)+self.integral, grip]
        arm_step = .010 if phase in ('descend', 'close') else .04
        delta = np.clip(wanted-self.previous, -np.r_[np.full(5, arm_step), .012 if phase == 'close' else .03],
                        np.r_[np.full(5, arm_step), .012 if phase == 'close' else .03])
        self.previous = self.previous+delta
        self.previous[:5] = np.clip(self.previous[:5], self.kin.limits[:, 0], self.kin.limits[:, 1])
        self.previous[5] = np.clip(self.previous[5], -np.pi/18, np.deg2rad(100))
        return self.previous.copy()
