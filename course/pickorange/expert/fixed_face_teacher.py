"""Controlled closure variant: preserve loaded arm command while closing.

Uses the same calibrated faces/path/actual gates. No physics/threshold changes.
"""
import numpy as np
from calibrated_face_teacher import CalibratedFaceTeacher


class FixedFaceTeacher(CalibratedFaceTeacher):
    def command(self,previous,measured):
        if self.stage!=2:
            return super().command(previous,measured)
        if np.linalg.norm((self.object()-self.close_anchor)[:2])>.012:
            self.fail('object pushed >12mm during fixed-arm closure')
            return previous.copy()
        wanted=previous.copy()
        wanted[:5]=self.close_hold_command
        wanted[5]=previous[5]+np.clip(self.close_rad-previous[5],-.012,.012)
        return wanted
