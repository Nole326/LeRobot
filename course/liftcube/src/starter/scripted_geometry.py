"""Optional fixed-CAD grasp choices and a measured-height clearance subgoal."""
import numpy as np

from .interfaces.action import USD_LOW_DEG, USD_HIGH_DEG, degrees_to_motor
from .scripted_liftcube import (
    GraspStall, MissedGrasp, SteadyBiasScriptedLiftCube, TargetNotVisible, _array, _rotation, pixel_to_base,
    red_centroid, red_mask, rotation_step,
)


def fit_cube_outline(image, pixel, c):
    """Fit a 30 mm tabletop cube to the red silhouette in the calibrated view."""
    from scipy.ndimage import label
    from scipy.optimize import least_squares

    labels,_ = label(red_mask(image))
    ys,xs = np.nonzero(labels)
    component = labels[ys,xs][np.argmin((xs-pixel[0])**2+(ys-pixel[1])**2)]
    ys,xs = np.where(labels==component)
    angles = np.arange(32)*np.pi/16
    directions = np.stack((np.cos(angles),np.sin(angles)))
    observed = (np.column_stack((xs,ys))@directions).max(axis=0)
    camera = c["camera_to_base"]
    origin = pixel_to_base(pixel,c)
    # The projected corners describe the cube. The TCP offset belongs to the grasp.
    origin[:2] -= c["grasp_xy_offset"]
    corners = np.array([[x,y,z] for x in (-.015,.015) for y in (-.015,.015) for z in (-.015,.015)])
    def error(params):
        dx,dy,yaw = params
        co,si = np.cos(yaw),np.sin(yaw)
        xyz = corners.copy()
        xyz[:,:2] = corners[:,:2]@np.array([[co,si],[-si,co]])
        points = (xyz+origin+[dx,dy,0]-camera[:3,3])@camera[:3,:3]@c["K"].T
        return ((points[:,:2]/points[:,2:])@directions).max(axis=0)-observed
    fits = [least_squares(error,[0,0,yaw],bounds=([-.012,-.012,-np.pi/2],[.012,.012,np.pi/2]))
            for yaw in (-.6,0,.6)]
    fit = min(fits,key=lambda f:f.cost)
    origin[:2] += c["grasp_xy_offset"]
    if np.sqrt(np.mean(fit.fun**2)) > 1.5:
        return origin,None
    return origin+np.r_[fit.x[:2],0], float(fit.x[2])


class UnreachableGrasp(GraspStall):
    """No grasp passed the arm limits at all three waypoints."""


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
        low_grasp = calibration.get('low_fingertip_grasp', False)
        if not isinstance(low_grasp, bool):
            raise ValueError('low_fingertip_grasp must be boolean')
        if low_grasp:
            # Keep the same physical TCP for intermediate rotations. Prefer the
            # most level reachable jaw pair before selecting the oblique preset.
            from scipy.spatial.transform import Rotation
            straight = next((p for p in self.presets if p['tilt'] == 0), None)
            if straight is None:
                raise ValueError('low_fingertip_grasp requires a calibrated zero-tilt preset')
            middle = [dict(tilt=t, rotation=straight['rotation'] @
                           Rotation.from_euler('y', t, degrees=True).as_matrix(),
                           tcp_offset_m=straight['tcp_offset_m'].copy()) for t in (-5, -10, -15)]
            self.presets = [straight, *middle, *(p for p in self.presets if p is not straight)]
        kinematics.tcp = np.array(calibration['tcp_offset_m'])
        super().__init__(calibration, kinematics)
        self.nominal_grasp_z = self.c['grasp_z']
        faster = self.c.get('fast_clearance', False)
        self.clearance_motor_step = .5 if faster else .25
        if faster:
            self.close_dwell_ticks = 8

    def reset(self):
        super().reset()
        self.hover_motor = self.selected_tilt = None
        self.clear_lift = False
        self.transport_rotation = None
        self.clearance_stable = 0
        self.pad_active = False
        self.relieved = False
        self.approach_z = self.c['hover_z']

    def _lost_after_lift(self, image, q):
        if not self.c.get('clear_retry_view'):
            return super()._lost_after_lift(image, q)
        try:
            pixel = red_centroid(image, self.c['image_shape'])
        except TargetNotVisible:
            return False
        ray = self.c['camera_to_base'][:3,:3] @ np.linalg.solve(self.c['K'], np.r_[pixel, 1.])
        ray /= np.linalg.norm(ray)
        fingers = self._pose(q)[:3,3] - self.c['camera_to_base'][:3,3]
        # A carried 30 mm cube must overlap the fingers in the image.
        separated = np.linalg.norm(np.cross(fingers, ray)) > .042
        near_pick = self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*self.c['image_shape'][0]/64
        return separated or near_pick

    def _transition(self, phase):
        super()._transition(phase)
        if phase == 'localize' and self.c.get('clear_retry_view'):
            self.pixel = self.target = self.hover_motor = None
            self.stable = 0

    def _pad_grasps(self, point):
        """Align the closing axis with a visible side and keep both pads low."""
        from scipy.spatial.transform import Rotation
        vertical = np.array([[0.,1,0],[-1,0,0],[0,0,1]])
        radial = Rotation.from_euler('z', np.arctan2(point[0],-point[1])).as_matrix()
        shape = np.asarray(self.c['pad_clearance_vertices_m'])
        if shape.ndim != 2 or not len(shape) or shape.shape[1] != 3 or not np.isfinite(shape).all():
            raise ValueError('Expected gripper-frame pad vertices in metres')
        poses = []
        for tilt in (-5, -10, -15, -20, -25, -30):
            nominal = vertical @ Rotation.from_euler('y',tilt,degrees=True).as_matrix()
            world = radial @ nominal
            for k in range(4):
                normal = float(self.face_yaw or 0.) + k*np.pi/2
                side = np.array([np.cos(normal), np.sin(normal), 0.])
                roll = np.arctan2(side @ world[:,1], side @ world[:,0])
                R = nominal @ Rotation.from_euler('z',roll).as_matrix()
                # Center the aperture at the expected enclosing width, accounting
                # for the SO-101's fixed finger and its single pivoting jaw.
                tcp = np.array([.0095, 0., -.094])
                lowest = ((shape-tcp) @ R[2]).min()
                desired = self.c['visible_plane_z']-.015+.0025-self.c['grasp_z']
                tcp[2] += (lowest-desired)/R[2,2]
                poses.append((abs(R[2,0]),abs(roll),dict(tilt=tilt,rotation=R,tcp_offset_m=tcp)))
        return [p for _,__,p in sorted(poses,key=lambda v:(v[0],v[1]))]

    def select_pose(self, point, q):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        initial = np.deg2rad(np.clip(self._urdf(q), USD_LOW_DEG[:5], USD_HIGH_DEG[:5]))
        vertical = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
        full_pad = self.pad_active
        for preset in (self._pad_grasps(point) if full_pad else self.presets):
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
                if full_pad:
                    roll = (vertical @ Rotation.from_euler('y',preset['tilt'],degrees=True).as_matrix()).T @ R
                    lift = lift @ roll
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
        raise UnreachableGrasp('No CAD grasp has reachable waypoints and 3motor arm margin')

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
                point = self._plan_from_image(image, self.pixel, q)
                self.target = np.r_[point[:2], self.approach_z]
                self._transition('hover')
            return goal
        if self.phase == 'lift' and self.clear_lift:
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            _, angle = rotation_step(pose[:3, :3], self.c['grasp_rotation'])
            clear = pose[2, 3] >= self.c['hover_z']-.003 and angle <= .15
            early_clear = self.c.get("fast_loaded_clearance") and self.jaw_blocked and angle <= .15
            if self.c.get('probe_loaded_close') and (self.ticks >= 36 or early_clear):
                clear |= pose[2, 3] >= self.c['grasp_z']+.025 and angle <= .22
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
            try:
                pixel = red_centroid(image, self.c['image_shape'])
            except TargetNotVisible:
                if not (self.retries and self.c.get("clear_retry_view")):
                    raise
                self.pixel, self.stable = None, 0
                pose = self._pose(self.last)
                target = pose[:3, 3].copy()
                target[2] = min(self.c["hover_z"]+.025, self.c["workspace"][1, 2], target[2]+.003)
                return self._solve(target, self.last, self.c["open_motor"], pose[:3, :3])
            scale = self.c['image_shape'][0]/64
            self.stable = self.stable+1 if self.pixel is not None and np.linalg.norm(pixel-self.pixel) <= 2*scale else 1
            self.pixel = pixel
            if self.stable >= 3:
                point = self._plan_from_image(image, pixel, q)
                self.target = np.r_[point[:2], self.approach_z]
                self._transition('hover')
            return self.last.copy()
        if self.phase == 'hover':
            self.ticks += 1
            self._check_target(image)
            pose = self._pose(q)
            self._inside(pose[:3, 3])
            _, angle = rotation_step(pose[:3, :3], self.c['grasp_rotation'])
            reached = (np.linalg.norm(self.target-pose[:3, 3]) <= .003 and angle <= .15
                       and np.max(np.abs(self.hover_motor[:5]-q[:5])) <= 2)
            if self.retries or self.approach_z > self.c['hover_z']:
                # Cartesian alignment establishes clearance even under gravity sag.
                position = pose[:3, 3]
                reached |= (np.linalg.norm(self.target[:2]-position[:2]) <= .0045
                            and (self.c['grasp_z']+.015 if self.retries else self.c['hover_z']) <= position[2] <= self.target[2]+.012
                            and angle <= .15)
            self.settled = self.settled+1 if reached and self._jaw_ready(q) else 0
            if self.settled >= 3:
                self.target[2] = self.c['grasp_z']
                self._transition('descend')
            delta = self.hover_motor-self.last
            if self.c.get("parallel_approach"):
                delta[5] = 0
            goal = self.last+self._limit_motion(delta)
            reference = self._pose(self.last)
            for _ in range(20):
                following = self._pose(goal)
                _, turn = rotation_step(reference[:3, :3], following[:3, :3])
                travel = np.linalg.norm(following[:3, 3]-reference[:3, 3])
                if travel <= .004*self.travel_gain and turn <= .02*self.travel_gain+1e-8:
                    break
                scale = .5
                if self.pad_active or self.c.get("parallel_approach"):
                    scale = .99*self.travel_gain*min(.004/max(travel,1e-9),.02/max(turn,1e-9))
                goal = self.last+(goal-self.last)*scale
            else:
                raise ValueError('Cannot advance bounded joint approach')
        else:
            goal = super()._step(image, q)
        if (previous == 'close' and self.phase == 'lift'
                and self.c.get('follow_retry_close') and self.face_yaw is not None):
            axis = self._pose(q)[:2, 0]
            yaw = np.arctan2(axis[1], axis[0]) - self.face_yaw
            width_mm = 30.*(abs(np.sin(yaw)) + abs(np.cos(yaw)))
            # Distal pad calibration over the grasp range, from the SO-101 collision mesh.
            gap_mm = 3.5 + 1.37*q[5]
            if gap_mm > width_mm + 2.:
                if not self.retries and not self.relieved:
                    # Table contact can stall the moving finger before it reaches the cube.
                    # Reopen at the same XY, with room for the jaw to swing shut.
                    self.relieved = True
                    self.clear_lift = self.jaw_blocked = False
                    self.previous_jaw = None
                    self.contact_stable = self.loaded_close_ticks = 0
                    self.grasp_motor = self.c['close_motor']
                    self.target[2] = self.c['grasp_z']+.0035
                    self._transition('descend')
                    return self.last.copy()
                raise MissedGrasp('Jaw opening exceeds the visible cube width after closing')
        if previous == 'lift' and self.phase == 'hold' and self.clear_lift:
            self.clear_lift = False
            self.c['lift_rotation'] = self.transport_rotation.copy()
            self.target[2] = self.c['lift_z']
            self._transition('lift')
        if clearing:
            delta = goal-self.last
            step = self.clearance_motor_step
            if self.c.get("fast_loaded_clearance") and self.jaw_blocked and pose[2,3] >= self.c["grasp_z"]+.006:
                step = .9*self.travel_gain
            goal = self.last+delta*min(1., step/max(abs(delta).max(), 1e-9))
        return goal

    def _begin_lift(self):
        super()._begin_lift()
        self.clear_lift = True
        self.c['lift_rotation'] = self.c['grasp_rotation'].copy()
        self.target[2] = self.c['hover_z']

    def _plan_from_image(self, image, pixel, q):
        point = pixel_to_base(pixel, self.c)
        follow = self.retries and self.c.get('follow_retry_close')
        if self.c.get('follow_retry_close'):
            self.c['grasp_z'] = self.nominal_grasp_z + (.0035 if follow else 0.)
        self.pad_active = False
        self.relieved = False
        if self.c.get('face_aligned_grasp'):
            point,self.face_yaw = fit_cube_outline(image,pixel,self.c)
        elif follow:
            point, self.face_yaw = fit_cube_outline(image, pixel, self.c)
        elif self.c.get('follow_retry_close'):
            fitted, self.face_yaw = fit_cube_outline(image, pixel, self.c)
            if self.face_yaw is not None and np.linalg.norm(point[:2]) > .35:
                point = fitted
        # Use the fitted center directly; a blind sideways offset discards that estimate.
        fitted_retry = follow and self.face_yaw is not None
        if not fitted_retry:
            point = self._aim(point)
        self.select_pose(point, q)
        if self.c.get('pad_clearance_vertices_m') is not None and abs(self.c['grasp_rotation'][2,0]) > .17:
            # A 30 mm aperture tilted by about 10 degrees already separates
            # the two tips vertically by 5 mm. Prefer the existing flat grasp.
            fitted, self.face_yaw = fit_cube_outline(image, pixel, self.c)
            if self.face_yaw is not None:
                self.pad_active = True
                candidate = fitted if follow else self._aim(fitted)
                try:
                    self.select_pose(candidate, q)
                except UnreachableGrasp:
                    self.pad_active = False
                    self.select_pose(point, q)
                else:
                    point = candidate
        if self.c.get('face_aligned_grasp'):
            self.planned_tilt = None
            hover = self._face_pose(point,q)
            if hover is not None:
                self.hover_motor = hover
        self.approach_z = self.c['hover_z']
        if self.pad_active and self.c.get('clear_retry_view'):
            # Keep the lower tip clear while moving sideways with an inclined wrist.
            extra = min(.018, abs(self.c['grasp_rotation'][2, 0])*.045+.006)
            try:
                hover = self._solve(np.r_[point[:2], self.approach_z+extra], self.hover_motor,
                                    self.c['open_motor'], self.c['grasp_rotation'])
            except GraspStall:
                pass
            else:
                self.approach_z += extra
                self.hover_motor = hover
        return point

    def act(self, observation, deterministic=False):
        result = super().act(observation, deterministic)
        self.telemetry.update(selected_tilt=self.selected_tilt, tcp_offset_m=self.kin.tcp.tolist(),
            grasp_rotation=self.c['grasp_rotation'].tolist(), clear_lift=self.clear_lift)
        return result
