"""Opt-in planar TCP development controller, not the course's official IK API."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from starter.interfaces.action import (
    ActionSpec, USD_LOW_DEG, USD_HIGH_DEG, degrees_to_motor, motor_to_degrees, validate_action,
)
from starter.joint_chain import JointChainKinematics, transform
from starter.scripted_liftcube import _array
from .timing import action_hold_steps


def unit_vector(value, name):
    result = _array(value, (3,), name)
    if not np.isclose(np.linalg.norm(result), 1., atol=1e-6):
        raise ValueError(f'{name} must be a unit vector')
    return result


class PlanarEEController:
    """[absolute table x/m, table y/m, absolute jaw/motor_0_100] at 30Hz.

    Height and approach axis are fixed calibration, not policy actions. Rotation
    around that axis is free: this deliberately is NOT a full-orientation lock.
    Only measured joints and fixed kinematics/calibration enter the controller.
    """
    frequency_hz = 30.

    def __init__(self, kinematics, *, table_to_base, xy_bounds, fixed_height_m,
                 tool_axis_local, tool_axis_table, tracking_compensation=False):
        self.kin = kinematics
        self.base_to_table = np.linalg.inv(transform(table_to_base, 'table_to_base'))
        self.bounds = _array(xy_bounds, (2, 2), 'xy_bounds')
        if np.any(self.bounds[0] >= self.bounds[1]):
            raise ValueError('xy_bounds must be [minimum_xy, maximum_xy] in metres')
        self.height = float(fixed_height_m)
        if not np.isfinite(self.height):
            raise ValueError('fixed_height_m must be finite')
        self.local_axis = unit_vector(tool_axis_local, 'tool_axis_local')
        self.axis = unit_vector(tool_axis_table, 'tool_axis_table')
        if not isinstance(tracking_compensation, bool):
            raise ValueError('tracking_compensation must be an explicit boolean')
        self.compensate = tracking_compensation
        self.action_spec = ActionSpec(kind='planar_tcp_position_dev', dim=3,
            names=('tcp_x', 'tcp_y', 'gripper.pos'), units=('m_table', 'm_table', 'motor_0_100'),
            low=np.r_[self.bounds[0], 0.], high=np.r_[self.bounds[1], 100.], frequency_hz=30.,
            notes='Development only. Absolute table XY; fixed height/approach axis; free axial rotation.')
        self.last = None
        self.sent = None
        self.failure_reason = None
        self.telemetry = {}
        self.calibration_sha256 = None

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema') != 'so101-planar-ee-dev-v1' or not data.get('evidence'):
            raise ValueError('Provide a documented so101-planar-ee-dev-v1 calibration')
        chain = path.parent / data['chain_path']
        if hashlib.sha256(chain.read_bytes()).hexdigest() != data['chain_sha256']:
            raise ValueError('Planar controller joint-chain SHA mismatch')
        kin = JointChainKinematics(chain, data['tcp_offset_m'], data['joint_names'])
        controller = cls(kin, **{key:data[key] for key in (
            'table_to_base', 'xy_bounds', 'fixed_height_m', 'tool_axis_local', 'tool_axis_table')},
            tracking_compensation=data.get('tracking_compensation', False))
        controller.calibration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return controller

    def pose(self, motor):
        return self.base_to_table @ self.kin.forward_kinematics(motor_to_degrees(motor)[:5])

    def _on_plane(self, pose):
        axis = pose[:3, :3] @ self.local_axis
        angle = np.arccos(np.clip(axis @ self.axis, -1., 1.))
        return abs(pose[2, 3]-self.height) <= .003 and angle <= .15

    def validate_request(self, action):
        request = _array(action, (3,), 'planar action')
        if np.any(request < self.action_spec.low) or np.any(request > self.action_spec.high):
            raise ValueError('Planar action outside calibrated XY/jaw bounds; no clipping')
        return request

    def reset(self, measured_motor):
        self.last = None  # A failed reset must not retain a command from another episode.
        self.sent = None
        self.failure_reason = None
        self.telemetry = {}
        measured = _array(measured_motor, (6,), 'measured joints')
        validate_action(measured, ActionSpec(), clip=False)
        pose = self.pose(measured)
        if (not self._on_plane(pose) or np.any(pose[:2,3] < self.bounds[0])
                or np.any(pose[:2,3] > self.bounds[1])):
            raise ValueError('Initial TCP must already satisfy calibrated plane, axis and XY bounds')
        self.last = measured.copy()
        self.sent = measured.copy()
        self.bias = np.zeros(5)
        self.stationary = 0

    def command(self, action, measured_motor):
        if self.last is None:
            raise RuntimeError('Reset the planar controller before commanding it')
        request = self.validate_request(action)
        if self.failure_reason is not None:
            self.telemetry.update(requested_ee=request.tolist(), failure=self.failure_reason)
            return self.sent.copy()
        try:
            from scipy.optimize import least_squares
            measured = _array(measured_motor, (6,), 'measured joints')
            target = np.r_[request[:2], self.height]

            def residual(radians):
                pose = self.base_to_table @ self.kin.forward_kinematics(np.rad2deg(radians))
                return np.r_[pose[:3,3]-target, .1*(pose[:3,:3] @ self.local_axis-self.axis)]

            fit = least_squares(residual, np.deg2rad(motor_to_degrees(self.last)[:5]),
                bounds=(np.deg2rad(USD_LOW_DEG[:5].astype(float)),
                        np.deg2rad(USD_HIGH_DEG[:5].astype(float))), max_nfev=100)
            nominal = degrees_to_motor(np.r_[np.rad2deg(fit.x), 0.]).astype(float)
            nominal[5] = request[2]
            validate_action(nominal, ActionSpec(), clip=False)
            target_pose = self.pose(nominal)
            if np.linalg.norm(target_pose[:3,3]-target) > .003 or not self._on_plane(target_pose):
                raise ValueError('Planar target is unreachable within 3mm / 0.15rad axis tolerances')
            delta = nominal-self.last
            fraction = min(1., 1./max(float(np.max(np.abs(delta))), 1e-9))
            previous_pose = self.pose(self.last)
            for _ in range(20):
                reference = self.last+fraction*delta
                pose = self.pose(reference)
                if (np.linalg.norm(pose[:3,3]-previous_pose[:3,3]) <= .004+1e-10
                        and self._on_plane(pose)
                        and np.all(pose[:2,3] >= self.bounds[0])
                        and np.all(pose[:2,3] <= self.bounds[1])):
                    break
                fraction *= .5
            else:
                raise ValueError('Cannot advance while preserving the planar command constraints')
            stable = np.max(np.abs(reference[:5]-self.last[:5])) <= .02
            stationary = self.stationary+1 if stable else 0
            bias = self.bias.copy()
            if self.compensate and stationary >= 3:
                # Same bounded stationary-reference integral as the joint rule
                # policy. Bias is servo correction, never a new EE reference.
                bias = np.clip(bias+.1*(reference[:5]-measured[:5]), -3., 3.)
            corrected = reference.copy()
            corrected[:5] += bias
            validate_action(corrected, ActionSpec(), clip=False)
            sent_delta = corrected-self.sent
            sent = self.sent+sent_delta*min(1., 1./max(float(np.max(np.abs(sent_delta))), 1e-9))
            validate_action(sent, ActionSpec(), clip=False)
            if np.max(np.abs(sent[:5]-measured[:5])) > 8:
                raise ValueError('Planar joint target tracking lag exceeds 8 motor units')
            self.last, self.sent = reference.copy(), sent.copy()
            self.bias, self.stationary = bias, stationary
            measured_pose = self.pose(measured)
            self.telemetry = dict(requested_ee=request.tolist(), nominal_motor=nominal.tolist(),
                reference_motor=reference.tolist(), steady_bias_motor=bias.tolist(),
                sent_motor=sent.tolist(), reference_tcp_table=pose[:3,3].tolist(),
                commanded_tcp_table=self.pose(sent)[:3,3].tolist(),
                reference_tool_axis_table=(pose[:3,:3] @ self.local_axis).tolist(),
                commanded_tool_axis_table=(self.pose(sent)[:3,:3] @ self.local_axis).tolist(),
                measured_tcp_table=measured_pose[:3,3].tolist(),
                measured_height_error_m=float(measured_pose[2,3]-self.height),
                measured_axis_error_rad=float(np.arccos(np.clip(
                    (measured_pose[:3,:3] @ self.local_axis) @ self.axis, -1., 1.))), failure=None)
        except ValueError as error:
            self.failure_reason = str(error)
            self.telemetry = dict(requested_ee=request.tolist(), sent_motor=self.sent.tolist(), failure=str(error))
            raise
        return self.sent.copy()


class PlanarEEEnv:
    """Keep native physics ticks; solve once per 30Hz decision, hold between.

    Joint history in proprio remains native joint history, not a renamed EE
    command. Failure latches a last-command hold; it is NOT a hardware e-stop.
    """
    def __init__(self, env, controller):
        if env.action_spec.kind != 'joint_position' or env.action_spec.dim != 6:
            raise ValueError('Planar development controller requires the native six-joint adapter')
        self.env, self.controller = env, controller
        self.hold = action_hold_steps(env.action_spec.frequency_hz, controller.frequency_hz)
        self.action_spec = replace(controller.action_spec, frequency_hz=env.action_spec.frequency_hz)
        self.observation_spec, self.name = env.observation_spec, env.name
        self.obs = None

    def reset(self, **kwargs):
        self.obs, self.ended = None, True
        obs, info = self.env.reset(**kwargs)
        self.controller.reset(obs.proprio[:6])
        self.obs, self.tick, self.ended, self.request = obs, 0, False, None
        self.motor = self.controller.sent.copy()
        return obs, dict(info, planar_ee_development_contract=dict(
            action_spec=self.action_spec.to_dict(), calibration_sha256=self.controller.calibration_sha256,
            fixed_height_m=self.controller.height, tool_axis_table=self.controller.axis.tolist()))

    def step(self, action):
        if self.obs is None or self.ended:
            raise RuntimeError('Reset before stepping a new planar episode')
        request = self.controller.validate_request(action)
        if self.tick % self.hold == 0:
            self.request = request.copy()
            try:
                self.motor = self.controller.command(request, self.obs.proprio[:6])
            except ValueError:
                self.motor = self.controller.sent.copy()
        elif not np.array_equal(request, self.request):
            raise ValueError('Hold the same EE request between 30Hz decision boundaries')
        obs, reward, terminated, truncated, info = self.env.step(self.motor.copy())
        self.obs, self.ended = obs, bool(terminated or truncated)
        self.tick += 1
        return obs, reward, terminated, truncated, dict(info,
            ee_command=dict(self.controller.telemetry), ee_control_failed=self.controller.failure_reason is not None)

    def close(self):
        self.env.close()
