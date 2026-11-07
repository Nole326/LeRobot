"""SO-101 inverse kinematics. Native joint servos only, no simulation writes.

Five arm joints control a TCP position and the tool's upward Z axis. Rotation
around that axis is intentionally free, rather than requesting six pose DOFs.
This is a development teacher, not the unpublished course controller.
"""
import numpy as np
from joint_contract import JOINTS, USD_LIMITS_DEG


def rotation(quat):
    w = float(quat.GetReal())
    x, y, z = np.asarray(quat.GetImaginary(), dtype=float)
    w, x, y, z = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    v = np.array([x, y, z])
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (w*w-v@v)*np.eye(3) + 2*np.outer(v, v) + 2*w*skew


def transform(p, r):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


class Kinematics:
    def __init__(self, asset, tcp=(.030, .010, -.100)):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(asset))
        self.links = []
        self.tcp = np.asarray(tcp, dtype=float)
        self.limits = np.deg2rad(USD_LIMITS_DEG[:5])
        for name in JOINTS[:5]:
            j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/so101_new_calib/joints/'+name))
            assert j and j.GetAxisAttr().Get() == 'Z'
            self.links.append((transform(j.GetLocalPos0Attr().Get(), rotation(j.GetLocalRot0Attr().Get())),
                               np.linalg.inv(transform(j.GetLocalPos1Attr().Get(), rotation(j.GetLocalRot1Attr().Get())))))

    def fk(self, q):
        t = np.eye(4)
        origins, axes = [], []
        self.last_frames = []
        for angle, (a, b) in zip(q, self.links):
            t = t @ a
            origins.append(t[:3, 3].copy())
            axes.append(t[:3, 2].copy())
            c, s = np.cos(angle), np.sin(angle)
            t = t @ transform(np.zeros(3), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])) @ b
            self.last_frames.append(t.copy())
        point = t[:3, 3] + t[:3, :3] @ self.tcp
        z = t[:3, 2]
        jac = np.stack([np.r_[np.cross(axis, point-origin), .10*np.cross(axis, z)]
                        for axis, origin in zip(axes, origins)], axis=1)
        return t, point, jac

    def solve(self, target, reference, up=(0., 0., 1.), axis_weight=.02):
        target, reference, up = map(lambda x: np.asarray(x, dtype=float), (target, reference, up))
        assert target.shape == reference[:3].shape == up.shape == (3,)
        best = None
        candidates = []
        # Deterministic starts; no sampling RNG is consumed by this teacher.
        starts = [reference[:5], np.deg2rad([0, 30, -70, 40, 0]),
                  np.deg2rad([0, -30, 70, 40, 0]), np.deg2rad([0, 70, -90, 20, 0])]
        for start in starts:
            q = np.clip(start.copy(), self.limits[:, 0]+1e-4, self.limits[:, 1]-1e-4)
            from scipy.optimize import least_squares
            def residual(v):
                t, point, _ = self.fk(v)
                return np.r_[point-target, axis_weight*(t[:3, 2]-up), .002*(v[4]-reference[4])]
            def derivative(v):
                _, _, j = self.fk(v)
                j[3:] *= axis_weight/.10
                return np.vstack([j, [0,0,0,0,.002]])
            q = least_squares(residual, q, jac=derivative,
                bounds=(self.limits[:, 0]+1e-4, self.limits[:, 1]-1e-4),
                max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10).x
            t, point, _ = self.fk(q)
            position_error = float(np.linalg.norm(target-point))
            axis_error = float(np.linalg.norm(up-t[:3, 2]))
            cost = position_error + axis_weight*axis_error
            candidates.append(dict(q_deg=np.rad2deg(q).tolist(), cost=cost,
                link_origins_base=[v[:3,3].tolist() for v in self.last_frames]))
            if best is None or cost < best['cost']:
                best = dict(q=q, cost=cost, position_error_m=position_error,
                            tool_axis_error=axis_error, target=target.tolist(), actual=point.tolist())
        best['candidates'] = candidates
        return best


class GeometricTeacher:
    """Measured-stage grasp/place teacher; bounded by the native 1500 steps."""
    stages = ('approach', 'descend', 'close', 'lift', 'carry', 'lower', 'release', 'retreat', 'home')
    codes = (120, 330, 380, 450, 520, 570, 620, 660, 680)
    caps = (160, 100, 80, 100, 180, 70, 40, 70, 250)

    def __init__(self, asset, close_rad=.2):
        self.kin = Kinematics(asset)
        self.close_rad = close_rad

    @property
    def is_episode_done(self):
        return self._episode_done

    def reset(self):
        self._orange_now, self.stage, self.elapsed, self.stable = 1, 0, 0, 0
        self.order_index = 0
        self._episode_done = False
        self.failure_reason = None
        self.events, self.closure_reports, self.grasp_reports = [], {}, {}
        self.angles, self.heights, self.relatives = [], [], []
        self.joint_history = []
        self.integral = np.zeros(5)
        self.gravity_offset = np.zeros(5)
        self.goal = None

    @property
    def _step_count(self):
        return self.codes[self.stage]

    def setup(self, env):
        from scipy.spatial.transform import Rotation
        self.env = env
        self.robot = env.scene['robot']
        self.body_id = self.robot.find_bodies('gripper')[0][0]
        self.ids = self.robot.find_joints(JOINTS, preserve_order=True)[0]
        base = self.robot.data.root_pos_w[0].cpu().numpy()
        quat = self.robot.data.root_quat_w[0].cpu().numpy()
        self.base, self.rotation = base, Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        self._rest_ee_pos_world = self.robot.data.body_pos_w[:,self.body_id].clone()
        self._rest_quat_world = self.robot.data.body_quat_w[:,self.body_id].clone()
        self.initial_q = self.measured()
        self.pick_order = sorted(range(1,4), key=lambda i: float(np.linalg.norm(
            self.env.scene[f'Orange00{i}'].data.root_pos_w[0,:2].cpu().numpy()-base[:2])))
        self._orange_now = self.pick_order[0]
        self.events.append({'pick_order':self.pick_order,
            'selection':'nearest-to-base first; original scene is not moved or filtered'})
        t, _, _ = self.kin.fk(self.initial_q[:5])
        actual = self.robot.data.body_pos_w[0,self.body_id].cpu().numpy()
        assert np.linalg.norm(self.rotation@t[:3,3]+base-actual) < 1e-4, 'FK does not match native articulation'
        self.enter(0)

    def measured(self):
        return self.robot.data.joint_pos[0,self.ids].cpu().numpy().copy()

    def object(self):
        return self.env.scene[f'Orange00{self._orange_now}'].data.root_pos_w[0].cpu().numpy().copy()

    def tcp(self):
        # Measured simulator body transform, not a desired command/FK estimate.
        from scipy.spatial.transform import Rotation
        q = self.robot.data.body_quat_w[0,self.body_id].cpu().numpy()
        r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        return self.robot.data.body_pos_w[0,self.body_id].cpu().numpy()+r@self.kin.tcp

    def fail(self, reason):
        self.failure_reason = f'Orange00{self._orange_now}/{self.stages[self.stage]}: {reason}'
        self._episode_done = True
        self.events.append({'failure_snapshot': {'joint_actual_rad':self.measured().tolist(),
            'last_transmitted_command_rad':getattr(self,'applied_command',self.measured()).tolist(),
            'object':self.object().tolist(),'tcp':self.tcp().tolist(),
            'body_quat_w':self.robot.data.body_quat_w[0,self.body_id].cpu().numpy().tolist()}})
        try:
            self.events[-1]['failure_snapshot']['joint_reaction_wrench_b'] = \
                self.robot.data.body_incoming_joint_wrench_b[0].cpu().numpy().tolist()
        except Exception as exc:
            self.events[-1]['failure_snapshot']['wrench_diagnostic_unavailable'] = type(exc).__name__

    def enter(self, stage):
        self.stage, self.elapsed, self.stable = stage, 0, 0
        self.angles, self.heights, self.relatives = [], [], []
        self.joint_history = []
        if stage == 0:
            self.pick_anchor = self.object()
            plate = self.env.scene['Plate'].data.root_pos_w[0].cpu().numpy().copy()
            direction = (self.base-plate)[:2]
            direction /= np.linalg.norm(direction)
            # Stay well inside the ORIGINAL success box, not on its edge.
            lateral = np.array([-direction[1], direction[0]])
            self.drop_anchor = plate.copy()
            self.drop_anchor[:2] += .03*direction + (self._orange_now-2)*.04*lateral
        if stage in (0,1,3,4,5,7):
            target = self.pick_anchor.copy() if stage in (0,1,3) else self.drop_anchor.copy()
            target[2] += {0:.080,1:.0,3:.110,4:.110,5:.070,7:.110}[stage]
            # Re-snapshot only at descent, before closing. Lift never chases a moving object.
            if stage == 1:
                self.pick_anchor = self.object()
                target = self.pick_anchor.copy()
            self.target = target
            local_target = self.rotation.T@(target-self.base)
            radial = local_target[:2]/np.linalg.norm(local_target[:2])
            tilt = np.deg2rad(-25.)
            up = np.r_[radial*np.sin(tilt), np.cos(tilt)]
            solved = self.kin.solve(local_target, self.measured()[:5], up=up)
            self.goal = solved.pop('q')
            self.events.append({'orange':self._orange_now, 'stage':self.stages[stage],
                                'ik':solved, 'goal_deg':np.rad2deg(self.goal).tolist()})
            assert np.isfinite(self.goal).all()
        elif stage == 8:
            self.goal = np.deg2rad([0,-100,90,50,0])
        elif stage == 2:
            # A loaded articulation can differ from its sent target. Replacing
            # that target with measured q would retract while closing.
            self.close_hold_command = self.applied_command[:5].copy()
        # Close/release keep the previous arm goal. All stages use actual servos.

    def get_action(self, env):
        import torch
        return torch.tensor([[0,0,0,1,0,0,0,1]], device=env.device, dtype=torch.float32)

    def command(self, previous, measured):
        if self.stage == 2 and hasattr(self, 'close_hold_command'):
            wanted = previous.copy()
            wanted[:5] = self.close_hold_command
            wanted[5] = self.close_rad
            return wanted
        # Small bounded outer-loop servo correction compensates gravity without
        # changing actuator gains or directly writing physical joint states.
        # Native servos already track qpos. Integrating the unsent IK goal
        # during slew limiting created +/-0.12 rad windup and TCP drift.
        if hasattr(self, 'robot'):
            # Installed IsaacLab's own OSC action uses this PhysX accessor.
            # Bias the native position command, not the actuator or simulation.
            force = self.robot.root_physx_view.get_gravity_compensation_forces()[0,self.ids[:5]].cpu().numpy()
            assert np.isfinite(force).all()
            self.gravity_offset = np.clip(force/17.8, -.35, .35)
        else:
            self.gravity_offset = np.zeros(5)
        raw=self.goal+self.gravity_offset+self.integral
        # Integrate only after the slew-limited command reaches the requested
        # goal. This removes steady payload error without integrating an
        # unreachable transient waypoint while the arm is still in transit.
        if np.max(np.abs(raw-previous[:5])) < .012:
            self.integral=np.clip(self.integral+.04*(self.goal-measured[:5]),-.12,.12)
        wanted = previous.copy()
        wanted[:5] = self.goal+self.gravity_offset+self.integral
        wanted[5] = (-10*np.pi/180 if self.stage == 8 else
                     self.close_rad if self.stage in (2,3,4,5) else .65)
        return wanted

    def observe(self):
        from grasp_quality import check_closure, check_grasp
        self.elapsed += 1
        obj, tcp, q = self.object(), self.tcp(), self.measured()
        distance = np.linalg.norm(tcp-self.target) if self.stage != 8 else np.inf
        ready = False
        if self.stage in (0,1,4,5,7):
            self.stable = self.stable+1 if distance < .008 else 0
            ready = self.stable >= 6 and self.elapsed >= 16
            if self.stage == 1 and not ready:
                self.joint_history.append(q[:5].copy())
                self.joint_history = self.joint_history[-12:]
                if len(self.joint_history) == 12 and self.elapsed >= 30:
                    span = float(np.ptp(np.array(self.joint_history),axis=0).max())
                    xy = float(np.linalg.norm((obj-tcp)[:2]))
                    clearance = float(tcp[2]-obj[2])
                    try:
                        wrench = self.robot.data.body_incoming_joint_wrench_b[0,self.body_id].cpu().numpy()
                        reaction = float(np.linalg.norm(wrench[:3]))
                    except Exception:
                        reaction = 0.  # no fallback acceptance without evidence
                    # A contact boundary can prevent reaching a free-space pose.
                    # This only authorizes a bounded CLOSURE ATTEMPT: success is
                    # still checked using actual lift, carry and native task_done.
                    blocked = (span <= .02 and xy <= .025 and 0 <= clearance <= .045
                        and reaction >= 8. and np.max(abs(self.applied_command[:5]-q[:5])) >= .12)
                    if blocked:
                        self.events.append({'contact_limited_closure_attempt':{
                            'reaction_force_norm_N':reaction,'joint_span_rad':span,
                            'object_tcp_xy_m':xy,'tcp_above_object_m':clearance,
                            'object_contact_partner_verified':False}})
                        self.goal = q[:5].copy()
                        self.integral[:] = 0.
                        ready = True
        elif self.stage == 2:
            self.angles.append(float(q[5]))
            checked = check_closure(self.angles, float(self.applied_command[5]), self.close_rad)
            ready = checked['verified'] and self.elapsed >= 20
            if ready:
                self.closure_reports[self._orange_now] = checked
                self.lift_initial_z = float(obj[2])
        elif self.stage == 3:
            self.heights.append(float(obj[2]))
            self.relatives.append(obj-tcp)
            self.angles.append(float(q[5]))
            checked = check_grasp(self.relatives, self.heights, self.angles, self.lift_initial_z)
            ready = checked['verified'] and distance < .012
            if ready:
                self.grasp_reports[self._orange_now] = checked
                self.lift_z = float(obj[2])
                self.relative_reference = obj-tcp
        elif self.stage == 6:
            ready = self.elapsed >= 35
        elif self.stage == 8:
            # Native success remains authoritative; don't invent a home predicate.
            ready = False
        if self.stage in (4,5):
            # Lowering the held fruit is intentional. Absolute world height
            # cannot distinguish that motion from losing the grasp.
            lost = np.linalg.norm(obj-tcp-self.relative_reference) > .04
            self.slip = getattr(self, 'slip', 0)+1 if lost else 0
            if self.slip >= 6:
                self.fail('lost physical grasp during transport')
                return
        if ready:
            self.events.append({'orange':self._orange_now,'completed_stage':self.stages[self.stage],
                                'elapsed_steps':self.elapsed,'tcp_error_m':float(distance),
                                'object':obj.tolist(),'tcp':tcp.tolist()})
            if self.stage == 7:
                if self.order_index < 2:
                    self.order_index += 1
                    self._orange_now = self.pick_order[self.order_index]
                    self.enter(0)
                else:
                    self.enter(8)
            else:
                self.enter(self.stage+1)
        elif self.elapsed >= self.caps[self.stage]:
            self.fail(f'stage deadline; measured TCP error {distance:.5f} m')

    def advance(self):
        pass
