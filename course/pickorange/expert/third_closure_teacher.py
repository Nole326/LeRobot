"""Third-pick-only small-step midpoint servo; frozen first two picks intact."""
import numpy as np
from adaptive_face_teacher import AdaptiveFaceTeacher
from calibrated_face_teacher import CalibratedFaceTeacher


class ThirdClosureTeacher(AdaptiveFaceTeacher):
    def command(self, previous, measured):
        if self.stage != 2 or self.order_index != 2:
            return super().command(previous, measured)
        # FixedFaceTeacher deliberately holds arm commands. On the third pick
        # the rotating jaw's measured midpoint drifts beyond the diameter line.
        # Use actual jaw angle to compensate with ordinary bounded commands,
        # anchored at pre-close center, never chasing a pushed object.
        wanted = CalibratedFaceTeacher.command(self, previous, measured)
        if self.is_episode_done:
            return previous.copy()
        wanted[:5] = previous[:5] + np.clip(wanted[:5]-previous[:5], -.002, .002)
        if not self.adaptive_latched and self.pad_metrics()['gap_m'] < 1.25*self.object_diameter:
            wanted[5] = previous[5]+np.clip(self.close_rad-previous[5], -.004, .004)
        return wanted
