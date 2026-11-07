"""Use continuous motion for final empty retreat and return to rest."""
from efficient_motion_teacher import EfficientMotionTeacher
from reachable_retreat_teacher import ReachableRetreatTeacher


class ConservativeEfficientTeacher(EfficientMotionTeacher):
    def command(self, previous, measured):
        if self.stage == 8 or (self.stage == 7 and self.order_index == 2):
            return super().command(previous, measured)
        return ReachableRetreatTeacher.command(self, previous, measured)
