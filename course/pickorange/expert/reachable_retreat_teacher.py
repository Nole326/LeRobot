"""Empty third-pick retreat: keep actual pad axis, validate short inward path."""
import numpy as np
from scipy.spatial.transform import Rotation
from verified_release_teacher import VerifiedReleaseTeacher
from geometric_teacher import GeometricTeacher


def plan_retreat(kin, reference):
    """Deterministic local XYZ waypoints; original IK thresholds are unchanged."""
    reference = np.asarray(reference, float).copy()
    transform, start, _ = kin.fk(reference)
    axis = transform[:3, :3] @ kin.pad_axis
    kin.desired_axis = axis.copy()
    radial = start[:2] / np.linalg.norm(start[:2])
    candidates=[]
    for rise, inward in ((.030, .040), (.025, .040), (.020, .030)):
        finish = start + np.r_[-inward*radial, rise]
        points, rows, q = [], [], reference.copy()
        for fraction in np.linspace(0, 1, 13)[1:]:
            target = start + fraction*(finish-start)
            solved = kin.solve(target, q, local=True)
            assert np.isfinite(solved['q']).all()
            q = solved['q'].copy()
            points.append(target.copy())
            rows.append(dict(fraction=float(fraction), position_error_m=solved['position_error_m'],
                             tool_axis_error=solved['tool_axis_error'], q_rad=q.tolist()))
        candidates.append((max(r['position_error_m']+.04*r['tool_axis_error'] for r in rows),
            points,dict(upward_m=rise,inward_m=inward,kept_axis=axis.tolist(),checked_path=rows)))
    _,points,evidence=min(candidates,key=lambda row:row[0])
    return points,evidence


class ReachableRetreatTeacher(VerifiedReleaseTeacher):
    caps = (240, 220, 180, 180, 180, 100, 80, 240, 250)

    def enter(self, stage):
        if stage != 7 or self.order_index != 2:
            return super().enter(stage)
        # Parent stage7 switches to radial-default axis; do not make that jump.
        self.stage, self.elapsed, self.stable = 7, 0, 0
        self.angles, self.heights, self.relatives, self.joint_history = [], [], [], []
        self.kin.set_opening(float(self.measured()[5]))
        try:
            self.retreat_points, evidence = plan_retreat(self.kin, self.measured()[:5])
        except ValueError as exc:
            self.fail(str(exc))
            return
        # Use measured world orientation for feedback; retain the axis selected
        # from the actual FK rather than imposing a new post-release orientation.
        quat = self.robot.data.body_quat_w[0, self.body_id].cpu().numpy()
        self.kin.desired_axis = self.rotation.T @ (Rotation.from_quat(quat[[1,2,3,0]]).as_matrix() @ self.kin.pad_axis)
        self.retreat_index = 0
        self.target = self.base + self.rotation @ self.retreat_points[0]
        self.events.append(dict(orange=self._orange_now, reachable_empty_retreat=True,
                                original_ik_admission=True, **evidence))

    def command(self, previous, measured):
        if self.stage != 7 or self.order_index != 2:
            return super().command(previous, measured)
        self.kin.set_opening(float(measured[5]))
        solved = self.kin.solve(self.rotation.T @ (self.target-self.base), measured[:5], local=True)
        assert np.isfinite(solved['q']).all()
        self.goal = solved['q']
        wanted = GeometricTeacher.command(self, previous, measured)
        # First short lift remains gentle; increase only already-empty free motion.
        step = .008 if self.retreat_index < 4 else .040
        wanted[:5] = previous[:5] + np.clip(wanted[:5]-previous[:5], -step, step)
        wanted[5] = previous[5] + np.clip(.95-previous[5], -.030, .030)
        return wanted

    def observe(self):
        if self.stage != 7 or self.order_index != 2:
            return super().observe()
        self.elapsed += 1
        distance = float(np.linalg.norm(self.tcp()-self.target))
        self.stable = self.stable+1 if distance < .005 else 0
        if self.stable >= 2:
            self.events.append(dict(empty_retreat_waypoint=self.retreat_index,
                                    elapsed_steps=self.elapsed, tcp_error_m=distance))
            self.retreat_index += 1
            self.stable = 0
            if self.retreat_index == len(self.retreat_points):
                self.events.append(dict(orange=self._orange_now, completed_stage='retreat',
                                        elapsed_steps=self.elapsed, tcp_error_m=distance))
                self.enter(8)
                return
            self.target = self.base + self.rotation @ self.retreat_points[self.retreat_index]
        if self.elapsed >= self.caps[7]:
            self.fail('Empty retreat measured waypoint deadline')
