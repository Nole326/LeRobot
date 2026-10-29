"""Optional fixed-CAD grasp choices and a measured-height clearance subgoal."""
import numpy as np

from .interfaces.action import USD_LOW_DEG, USD_HIGH_DEG, degrees_to_motor
from .scripted_liftcube import (
    SteadyBiasScriptedLiftCube, _array, _rotation, pixel_to_base,
    red_centroid, rotation_step,
)


class GeometricLiftCube(SteadyBiasScriptedLiftCube):
    """RGB/q plus explicit robot/camera calibration; never reads scene objects.

    Presets describe the fixed gripper geometry, not a per-scene object pose.
    The joint-chain backend is required; original motor/lag guards still apply.
    """
    def __init__(self, calibration, kinematics):
        if calibration.get('kinematics_backend') != 'joint_chain' or not calibration.get('gravity_compensation'):
            raise ValueError('Geometric mode requires joint_chain and gravity_compensation')
        if not calibration.get('grasp_asset_sha256') or calibration['grasp_asset_sha256'] != kinematics.asset_sha256:
            raise ValueError('Grasp presets and joint chain must identify the same source asset')
        self.presets = []
        for preset in calibration.get('grasp_presets', []):
            self.presets.append(dict(tilt=float(preset['tilt']),
                rotation=_rotation(preset['rotation'], 'preset rotation'),
                tcp_offset_m=_array(preset['tcp_offset_m'], (3,), 'preset TCP')))
        if not self.presets or not all(np.isfinite(p['tilt']) for p in self.presets):
            raise ValueError('Provide finite, independently calibrated grasp_presets')
        kinematics.tcp = np.array(calibration['tcp_offset_m'])
        super().__init__(calibration, kinematics)

    def reset(self):
        super().reset()
        self.hover_motor = self.selected_tilt = None
        self.clear_lift = False
        self.transport_rotation = None
        self.clearance_stable = 0

    def select_pose(self, point, q):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        initial = np.deg2rad(self._urdf(q))
        vertical = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
        for preset in self.presets:
            self.kin.tcp = preset['tcp_offset_m'].copy()
            R = preset['rotation']
            target = np.r_[point[:2], self.c['grasp_z']]

            def residual(x):
                pose = self.kin._forward_radians(x[:5])
                desired = Rotation.from_euler('z', x[5]).as_matrix() @ R
                return np.r_[pose[:3, 3]-target,
                    .1*Rotation.from_matrix(desired.T @ pose[:3, :3]).as_rotvec()]

            fit = least_squares(residual, np.r_[initial, np.arctan2(point[0], -point[1])],
                bounds=(np.r_[np.deg2rad(USD_LOW_DEG[:5]), -np.pi],
                        np.r_[np.deg2rad(USD_HIGH_DEG[:5]), np.pi]), max_nfev=200)
            yaw = Rotation.from_euler('z', fit.x[5]).as_matrix()
            grasp = yaw @ R
            seed = degrees_to_motor(np.r_[np.rad2deg(fit.x[:5]), 0])
            seed[5] = self.c['open_motor']
            error = residual(fit.x)
            if np.linalg.norm(error[:3]) > .003 or np.linalg.norm(error[3:]) > .015:
                continue
            for lift_tilt in (-40, -60):
                lift = yaw @ vertical @ Rotation.from_euler('y', lift_tilt, degrees=True).as_matrix()
                try:
                    goals = [self._solve(np.r_[point[:2], z], seed, self.c['open_motor'], rotation=r)
                        for z, r in ((self.c['hover_z'], grasp), (self.c['grasp_z'], grasp), (self.c['lift_z'], lift))]
                    if max(float(np.max(np.abs(g[:5]))) for g in goals) > 97:
                        continue
                except ValueError:
                    continue
                self.c['grasp_rotation'], self.c['lift_rotation'] = grasp, lift
                self.c['tcp_offset_m'] = self.kin.tcp.copy()
                self.transport_rotation = lift.copy()
                self.selected_tilt, self.hover_motor = preset['tilt'], goals[0]
                return
        raise ValueError('No CAD grasp has reachable waypoints and 3motor arm margin')

    def _step(self, image, q):
        if self.phase == 'open' and self.c.get('plan_while_opening', False):
            try:
                pixel = red_centroid(image, self.c['image_shape'])
            except ValueError:
                self.pixel, self.stable = None, 0
            else:
                self.stable = self.stable+1 if self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*self.c['image_shape'][0]/64 else 1
                self.pixel = pixel
            # Preserve the complete original jaw motion and measured-state dwell.
            goal = super()._step(image, q)
            if self.phase == 'localize' and self.stable >= 3:
                point = pixel_to_base(self.pixel, self.c)
                self.select_pose(point, q)
                self.target = np.r_[point[:2], self.c['hover_z']]
                self._transition('hover')
            return goal
        if self.phase == 'lift' and self.clear_lift:
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            _, angle = rotation_step(pose[:3, :3], self.c['grasp_rotation'])
            clear = pose[2, 3] >= self.c['hover_z']-.003 and angle <= .15
            self.clearance_stable = self.clearance_stable+1 if clear else 0
            if self.clearance_stable >= 3:
                self.clear_lift = False
                self.c['lift_rotation'] = self.transport_rotation.copy()
                self.target[2] = self.c['lift_z']
                self._transition('lift')
        previous = self.phase
        clearing = self.phase == 'lift' and self.clear_lift
        if self.phase == 'localize':
            self.ticks += 1
            if self.ticks > 150:
                raise ValueError('Stage timeout (5 policy seconds)')
            pixel = red_centroid(image, self.c['image_shape'])
            scale = self.c['image_shape'][0]/64
            self.stable = self.stable+1 if self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*scale else 1
            self.pixel = pixel
            if self.stable >= 3:
                point = pixel_to_base(pixel, self.c)
                self.select_pose(point, q)
                self.target = np.r_[point[:2], self.c['hover_z']]
                self._transition('hover')
            return self.last.copy()
        if self.phase == 'hover':
            self.ticks += 1
            if self.ticks > 150:
                raise ValueError('Stage timeout (5 policy seconds)')
            pixel = red_centroid(image, self.c['image_shape'])
            if np.linalg.norm(pixel-self.pixel) > 3*self.c['image_shape'][0]/64:
                raise ValueError('Red target shifted before descent; reacquire next episode')
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            _, angle = rotation_step(pose[:3, :3], self.c['grasp_rotation'])
            reached = (np.linalg.norm(self.target-pose[:3, 3]) <= .003 and angle <= .15
                       and np.max(np.abs(self.hover_motor[:5]-q[:5])) <= 2)
            self.settled = self.settled+1 if reached else 0
            if self.settled >= 3:
                self.target[2] = self.c['grasp_z']
                self._transition('descend')
            delta = self.hover_motor-self.last
            goal = self.last+delta*min(1., 1./max(float(np.max(np.abs(delta))), 1e-9))
            reference = self._pose(self.last)
            for _ in range(20):
                following = self._pose(goal)
                _, turn = rotation_step(reference[:3, :3], following[:3, :3])
                if np.linalg.norm(following[:3, 3]-reference[:3, 3]) <= .004 and turn <= .02+1e-8:
                    break
                goal = (self.last+goal)/2
            else:
                raise ValueError('Cannot advance bounded joint approach')
        else:
            goal = super()._step(image, q)
        if previous == 'close' and self.phase == 'lift':
            self.clear_lift = True
            self.c['lift_rotation'] = self.c['grasp_rotation'].copy()
            self.target[2] = self.c['hover_z']
        elif previous == 'lift' and self.phase == 'hold' and self.clear_lift:
            self.clear_lift = False
            self.c['lift_rotation'] = self.transport_rotation.copy()
            self.target[2] = self.c['lift_z']
            self._transition('lift')
        if clearing:
            delta = goal-self.last
            goal = self.last+delta*min(1., .25/max(abs(delta).max(), 1e-9))
        return goal

    def act(self, observation, deterministic=False):
        result = super().act(observation, deterministic)
        self.telemetry.update(selected_tilt=self.selected_tilt, tcp_offset_m=self.kin.tcp.tolist(),
            grasp_rotation=self.c['grasp_rotation'].tolist(), clear_lift=self.clear_lift)
        return result
