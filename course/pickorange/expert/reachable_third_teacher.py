"""Third grasp: tiny closure servo, then a reachable inward-and-up lift."""
import numpy as np
from third_closure_teacher import ThirdClosureTeacher


def lift_target(anchor, base):
    anchor, base = np.asarray(anchor, float), np.asarray(base, float)
    radial = anchor[:2]-base[:2]
    assert np.linalg.norm(radial) > .04
    target = anchor.copy()
    target[:2] -= .04*radial/np.linalg.norm(radial)
    target[2] += .08
    return target


def validate_lift_path(kin, micro, full, reference):
    rows = []
    for fraction in np.linspace(0, 1, 9):
        solved = kin.solve(micro+fraction*(full-micro), reference)
        reference = solved['q']
        if solved['position_error_m'] > .004 or solved['tool_axis_error'] > .12:
            raise ValueError('Third inward lift waypoint exceeds original IK admission')
        rows.append(dict(fraction=float(fraction), position_error_m=solved['position_error_m'],
                         tool_axis_error=solved['tool_axis_error']))
    return reference, rows


class ReachableThirdTeacher(ThirdClosureTeacher):
    def enter(self, stage):
        if stage != 3 or self.order_index != 2:
            return super().enter(stage)
        self.stage, self.elapsed, self.stable = stage, 0, 0
        self.angles, self.heights, self.relatives, self.joint_history = [], [], [], []
        self.kin.set_opening(float(self.measured()[5]))
        micro = self.pick_anchor+np.array([0., 0., .015])
        full = lift_target(self.pick_anchor, self.base)
        try:
            self.goal, rows = validate_lift_path(self.kin, self.rotation.T@(micro-self.base),
                                               self.rotation.T@(full-self.base), self.measured()[:5])
        except ValueError as exc:
            self.fail(str(exc))
            return
        self.micro_lift_verified = False
        self.micro_height, self.micro_relative = [], []
        self.full_lift_target, self.target = full, micro
        self.events.append(dict(orange=self._orange_now, stage='lift',
                                reachable_inward_lift=True, upward_m=.08, inward_m=.04,
                                target_world_m=full.tolist(), checked_path=rows,
                                original_micro_and_full_grasp_gates=True))
