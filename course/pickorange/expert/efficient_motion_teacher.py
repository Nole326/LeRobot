"""Speed and path planning; contact gates and native physics unchanged.

Privileged scene geometry stays teacher-only. Contact, lift verification, placement
and release delegate to the previously successful ReachableRetreatTeacher.
"""
import numpy as np
from reachable_retreat_teacher import ReachableRetreatTeacher


def free_motion_step(stage, distance):
    """Per 30 Hz command bound, not physics dt or an actuator gain change."""
    if stage not in (0, 4, 7, 8):
        raise ValueError('Contact phase must not use free-motion speed')
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('Invalid measured distance')
    # Smooth braking instead of an abrupt large command near the destination.
    if stage == 8:
        return float(.025 + .015*np.clip((distance-.10)/.25, 0., 1.))
    return float(.020 + .020*np.clip((distance-.008)/.050, 0., 1.))


def path_progress(start, finish, actual):
    start, finish, actual = (np.asarray(v, dtype=float) for v in (start, finish, actual))
    if any(v.shape != (3,) for v in (start, finish, actual)):
        raise ValueError('XYZ vectors required')
    if not np.isfinite(np.r_[start, finish, actual]).all():
        raise ValueError('Nonfinite path geometry')
    direction = finish-start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError('Degenerate retreat path')
    along = float((actual-start) @ direction / length)
    lateral = float(np.linalg.norm(actual-start-along*direction/length))
    return along/length, lateral


class EfficientMotionTeacher(ReachableRetreatTeacher):
    def enter(self, stage):
        super().enter(stage)
        if stage == 7 and self.order_index == 2 and not self.is_episode_done:
            self.continuous_retreat_start = self.tcp().copy()
            self.continuous_retreat_finish = self.base+self.rotation@self.retreat_points[-1]
            self.events.append(dict(continuous_empty_retreat=True,
                preserved_12_point_ik_validation=True, gentle_first_four_points=True,
                intermediate_full_stops=False, final_arrival_m=.005,
                final_stable_physics_steps=2))

    def command(self, previous, measured):
        if self.stage == 7 and self.order_index == 2:
            # One-point lookahead avoids demanding an exact stop at every
            # tiny waypoint. The parent still limits the first four points
            # to .008rad; only MEASURED progress can unlock its .040rad bound.
            target = self.target.copy()
            index = min(self.retreat_index+1, len(self.retreat_points)-1)
            self.target = self.base+self.rotation@self.retreat_points[index]
            try:
                return super().command(previous, measured)
            finally:
                self.target = target
        wanted = super().command(previous, measured)
        if self.is_episode_done or self.stage not in (0, 4, 7, 8):
            return wanted
        distance = (float(np.max(np.abs(self.goal-measured[:5]))) if self.stage == 8
                    else float(np.linalg.norm(self.tcp()-self.target)))
        bound = free_motion_step(self.stage, distance)
        # Parent has already solved/validated IK and updated gravity/integral.
        # Reapply only the independent free-phase bound to the SAME raw goal.
        raw = self.goal+self.gravity_offset+self.integral
        wanted[:5] = previous[:5]+np.clip(raw-previous[:5], -bound, bound)
        # Gripper command is exactly the validated parent's command.
        return wanted

    def observe(self):
        if self.stage != 7 or self.order_index != 2:
            return super().observe()
        self.elapsed += 1
        actual = self.tcp()
        fraction, lateral = path_progress(self.continuous_retreat_start,
                                          self.continuous_retreat_finish, actual)
        # Passing a point requires actual progress, not merely a 5mm radius
        # around tightly spaced points. Thus a motionless arm cannot skip the
        # four gentle points and receive the faster command prematurely.
        while self.retreat_index < len(self.retreat_points)-1:
            point_fraction, _ = path_progress(self.continuous_retreat_start,
                self.continuous_retreat_finish,
                self.base+self.rotation@self.retreat_points[self.retreat_index])
            if fraction < point_fraction:
                break
            self.events.append(dict(empty_retreat_waypoint=self.retreat_index,
                elapsed_steps=self.elapsed, measured_path_fraction=fraction,
                lateral_error_m=lateral, intermediate_stop=False))
            self.retreat_index += 1
            self.target = self.base+self.rotation@self.retreat_points[self.retreat_index]
        distance = float(np.linalg.norm(actual-self.continuous_retreat_finish))
        at_last = self.retreat_index == len(self.retreat_points)-1
        self.stable = self.stable+1 if at_last and distance < .005 else 0
        if self.stable >= 2:
            self.events.append(dict(orange=self._orange_now, completed_stage='retreat',
                elapsed_steps=self.elapsed, tcp_error_m=distance,
                continuous_empty_retreat=True, measured_path_fraction=fraction))
            self.enter(8)
        elif self.elapsed >= self.caps[7]:
            self.fail('Continuous empty retreat measured arrival deadline')
