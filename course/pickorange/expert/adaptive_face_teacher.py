"""Native position-command closure with bounded overclosing after contact proxy.

No actuator stiffness/effort/friction changes. A blocked-angle proxy does not
prove contact; unchanged measured diameter, microlift and full-grasp gates do.
"""
import numpy as np
from fixed_face_teacher import FixedFaceTeacher


def latch_admission(angles,metrics,sent,diameter):
    angles=np.asarray(angles,dtype=float)
    return bool(len(angles)>=8 and np.isfinite(angles).all()
        and np.ptp(angles[-8:])<.015
        and .70*diameter<=metrics['gap_m']<=1.12*diameter
        and angles[-1]-sent>.035 and metrics['object_between_fingers'])


class AdaptiveFaceTeacher(FixedFaceTeacher):
    def enter(self,stage):
        if stage==0:self.close_rad=.2
        super().enter(stage)
        if stage==2:
            self.contact_angles=[];self.adaptive_latched=False

    def command(self,previous,measured):
        wanted=super().command(previous,measured)
        if self.stage==2 and not self.adaptive_latched:
            metrics=self.pad_metrics()
            if metrics['gap_m']<1.25*self.object_diameter:
                wanted[5]=previous[5]+np.clip(self.close_rad-previous[5],-.004,.004)
        return wanted

    def observe(self):
        if self.stage==2 and not self.adaptive_latched:
            q=float(self.measured()[5]);self.contact_angles.append(q)
            metrics=self.pad_metrics()
            if latch_admission(self.contact_angles,metrics,float(self.applied_command[5]),self.object_diameter):
                self.close_rad=max(.2,q-.035)
                self.adaptive_latched=True
                self.events.append(dict(adaptive_closure_latched=True,orange=self._orange_now,
                    held_target_rad=self.close_rad,measured_rad=q,
                    position_overclosing_rad=.035,metrics=metrics,
                    contact_sensor_verified=False))
        super().observe()
