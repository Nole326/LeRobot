"""Third-pick release waypoint clears the plate; success remains native."""
from reachable_third_teacher import ReachableThirdTeacher


class ReleaseClearanceTeacher(ReachableThirdTeacher):
    def enter(self, stage):
        super().enter(stage)
        if stage != 5 or self.order_index != 2 or self.is_episode_done:
            return
        self.target = self.target.copy()
        self.target[2] += .015
        solved = self.kin.solve(self.rotation.T@(self.target-self.base), self.measured()[:5])
        self.goal = solved.pop('q')
        self.events.append(dict(orange=self._orange_now, third_release_clearance=True,
                                height_above_plate_m=.085, target_world_m=self.target.tolist(),
                                ik=solved, original_8mm_arrival_gate=True))
