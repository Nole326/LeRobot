"""Third pick: open wider, withdraw fixed finger, verify actual release."""
import numpy as np
from scipy.spatial.transform import Rotation
from release_clearance_teacher import ReleaseClearanceTeacher
from geometric_teacher import GeometricTeacher


def release_admission(root, plate, separation, actual_opening):
    difference = np.asarray(root) - np.asarray(plate)
    return bool(np.isfinite(difference).all() and np.isfinite(separation)
                and np.isfinite(actual_opening) and actual_opening >= .90
                and np.all(np.abs(difference[:2]) < .10)
                and -.07 < difference[2] < .07 and separation >= .025)


class VerifiedReleaseTeacher(ReleaseClearanceTeacher):
    def enter(self, stage):
        if stage == 7 and self.order_index == 2:
            self.kin.desired_axis = None
        super().enter(stage)
        if stage == 6 and self.order_index == 2:
            self.open_rad = .95
            self.release_target = None

    def command(self, previous, measured):
        if self.stage != 6 or self.order_index != 2 or measured[5] < .90:
            return super().command(previous, measured)
        self.kin.set_opening(float(measured[5]))
        if self.release_target is None:
            quat = self.robot.data.body_quat_w[0, self.body_id].cpu().numpy()
            rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            axis = rotation @ self.kin.pad_axis
            self.release_target = self.tcp() - .012 * axis
            self.kin.desired_axis = self.rotation.T @ axis
            self.events.append(dict(third_fixed_finger_withdrawal_m=.012,
                target_world_m=self.release_target.tolist(),
                actual_opening_rad=float(measured[5])))
        solved = self.kin.solve(self.rotation.T @ (self.release_target - self.base),
                               measured[:5], local=True)
        assert np.isfinite(solved['q']).all()
        self.goal = solved['q']
        wanted = GeometricTeacher.command(self, previous, measured)
        wanted[:5] = previous[:5] + np.clip(wanted[:5] - previous[:5], -.004, .004)
        wanted[5] = previous[5] + np.clip(self.open_rad - previous[5], -.030, .030)
        return wanted

    def observe(self):
        if self.stage != 6 or self.order_index != 2:
            return super().observe()
        self.elapsed += 1
        root = self.root_object()
        plate = self.env.scene['Plate'].data.root_pos_w[0].cpu().numpy()
        separation = float(np.linalg.norm(self.object() - self.tcp()))
        admitted = self.release_target is not None and release_admission(
            root, plate, separation, float(self.measured()[5]))
        self.stable = self.stable + 1 if admitted and self.elapsed >= 35 else 0
        if self.stable >= 12:
            self.events.append(dict(orange=self._orange_now, completed_stage='release',
                elapsed_steps=self.elapsed, actual_release_verified=True,
                object_root_plate_delta_m=(root - plate).tolist(),
                object_tcp_separation_m=separation))
            self.enter(7)
        elif self.elapsed >= 180:
            self.fail('Third release not physically separated and in native plate box')
