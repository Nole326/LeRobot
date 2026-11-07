"""Independent diameter-aligned, slow-closing privileged teacher.

Finger-pad proxies are frozen-mesh geometry, not claimed contact sensors.
Only ordinary position commands are sent; all native physics is unchanged.
"""
import json
from pathlib import Path
import numpy as np
from geometric_teacher import GeometricTeacher, Kinematics, transform, rotation


class DiameterKinematics(Kinematics):
    def __init__(self, asset):
        super().__init__(asset)
        from pxr import Usd, UsdPhysics
        stage=Usd.Stage.Open(str(asset))
        j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/so101_new_calib/joints/gripper'))
        self.jaw_a=transform(j.GetLocalPos0Attr().Get(),rotation(j.GetLocalRot0Attr().Get()))
        self.jaw_b=np.linalg.inv(transform(j.GetLocalPos1Attr().Get(),rotation(j.GetLocalRot1Attr().Get())))
        geometry=json.loads(Path(__file__).with_name('diameter_geometry.json').read_text())
        self.fixed=np.asarray(geometry['pads']['gripper']['inner_proxy'])
        self.moving=np.asarray(geometry['pads']['jaw']['inner_proxy'])
        self.desired_axis=None
        self.set_opening(.85)

    def set_opening(self, angle):
        c,s=np.cos(angle),np.sin(angle)
        t=self.jaw_a@transform([0,0,0],np.array([[c,-s,0],[s,c,0],[0,0,1]]))@self.jaw_b
        moving=(t@np.r_[self.moving,1])[:3]
        self.tcp=(moving+self.fixed)/2
        self.pad_axis=(moving-self.fixed)/np.linalg.norm(moving-self.fixed)

    def solve(self, target, reference, up=None, axis_weight=None, local=False):
        from scipy.optimize import least_squares
        target=np.asarray(target)
        radial=target[:2]/np.linalg.norm(target[:2])
        desired=self.desired_axis if self.desired_axis is not None else np.r_[-radial[1],radial[0],0.]
        starts=[reference[:5]] if local else [reference[:5],
            np.deg2rad([0,-30,60,40,0]),np.deg2rad([0,30,-60,70,0])]
        best=None
        for start in starts:
            def residual(q):
                t,point,_=self.fk(q)
                # Five primary task DOFs: center XYZ and selected pad-pair direction.
                # Tool-up preference and roll continuity are soft, not 6D pose.
                return np.r_[point-target,.04*(t[:3,:3]@self.pad_axis-desired),
                    .005*(t[2,2]-1),.001*(q[4]-reference[4])]
            q=least_squares(residual,np.clip(start,self.limits[:,0]+1e-4,self.limits[:,1]-1e-4),
                bounds=(self.limits[:,0]+1e-4,self.limits[:,1]-1e-4),
                max_nfev=40 if local else 140,ftol=1e-8,xtol=1e-8,gtol=1e-8).x
            t,point,_=self.fk(q)
            vertical=abs(float((t[:3,:3]@self.pad_axis)[2]))
            axis_error=float(np.linalg.norm(t[:3,:3]@self.pad_axis-desired))
            position=float(np.linalg.norm(point-target))
            cost=position+.04*axis_error+.005*max(0.,.7-t[2,2])
            if best is None or cost<best['cost']:
                best=dict(q=q,cost=cost,position_error_m=position,
                    tool_axis_error=axis_error,target=target.tolist(),actual=point.tolist(),
                    pad_axis_vertical=vertical,tool_up_z=float(t[2,2]))
        return best


class DiameterTeacher(GeometricTeacher):
    caps=(220,180,180,140,180,100,80,90,250)

    def __init__(self, asset, close_rad=.2):
        self.kin=DiameterKinematics(asset)
        self.close_rad=close_rad

    def enter(self, stage):
        self.kin.set_opening(float(self.measured()[5]))
        if stage==0:
            local=self.rotation.T@(self.object()-self.base)
            radial=local[:2]/np.linalg.norm(local[:2])
            self.kin.desired_axis=np.r_[-radial[1],radial[0],0.]
        elif stage==4:
            self.kin.desired_axis=None  # transport may require another reachable azimuth
        super().enter(stage)
        if stage==2:
            self.close_anchor=self.object().copy()

    def tcp(self):
        self.kin.set_opening(float(self.measured()[5]))
        return super().tcp()

    def command(self, previous, measured):
        self.kin.set_opening(float(measured[5]))
        if self.stage in (0,1,2,3,4,5,7):
            # Recenter while the unilateral jaw rotates: midpoint is NOT fixed.
            target=self.object().copy() if self.stage in (1,2) else self.target.copy()
            if self.stage==2 and np.linalg.norm((target-self.close_anchor)[:2])>.012:
                self.fail('object pushed >12mm during closure; reject rather than chase')
                return previous.copy()
            self.target=target
            solved=self.kin.solve(self.rotation.T@(target-self.base),measured[:5],local=True)
            if solved['position_error_m']>.004 or solved['tool_axis_error']>.12:
                self.fail('diameter center / horizontal-axis target unreachable')
                return previous.copy()
            self.goal=solved['q']
        # Parent's loaded-command hold must not disable midpoint compensation.
        hold=getattr(self,'close_hold_command',None)
        if hold is not None: del self.close_hold_command
        wanted=super().command(previous,measured)
        if hold is not None: self.close_hold_command=hold
        wanted[5]=(-np.pi/18 if self.stage==8 else
            self.close_rad if self.stage in (2,3,4,5) else .85)
        if self.stage in (1,2):
            distance=np.linalg.norm(self.tcp()-self.target)
            arm_step=.008 if distance<.030 or self.stage==2 else .030
            wanted[:5]=previous[:5]+np.clip(wanted[:5]-previous[:5],-arm_step,arm_step)
        grip_step=.012 if self.stage==2 else .030
        wanted[5]=previous[5]+np.clip(wanted[5]-previous[5],-grip_step,grip_step)
        return wanted

    def observe(self):
        # No force-threshold shortcut. Actual pad-center alignment is required.
        if self.stage==1:
            self.elapsed+=1
            distance=float(np.linalg.norm(self.tcp()-self.object()))
            self.stable=self.stable+1 if distance<.006 else 0
            if self.stable>=6 and self.elapsed>=16:
                self.events.append(dict(orange=self._orange_now,completed_stage='descend',
                    elapsed_steps=self.elapsed,diameter_center_error_m=distance))
                self.enter(2)
            elif self.elapsed>=self.caps[1]:
                self.fail(f'diameter alignment deadline; center error {distance:.5f} m')
        else:
            super().observe()
