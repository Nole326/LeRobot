"""Check that a slew-limited transient cannot wind up the outer integrator."""
from types import SimpleNamespace
import numpy as np
from geometric_teacher import GeometricTeacher

t=SimpleNamespace(goal=np.full(5,.1),integral=np.zeros(5),stage=0)
previous=np.zeros(6)
measured=np.zeros(6)
for _ in range(100):
    GeometricTeacher.command(t,previous,measured)
assert np.all(t.integral==0), 'Unsent target must not accumulate integral'
previous[:5]=.1
measured[:5]=.08
GeometricTeacher.command(t,previous,measured)
assert np.allclose(t.integral,.0008), 'Steady tracking error should be corrected'
print('Slew anti-windup and steady feedback checks passed')

from calibrated_face_teacher import CalibratedFaceTeacher
def reject(reason):
    raise AssertionError('Approximate IK must not force recovery: '+reason)
kin=SimpleNamespace(set_opening=lambda angle:None,
    solve=lambda *args,**kwargs:dict(q=np.zeros(5),position_error_m=.000738,tool_axis_error=.120086))
t=SimpleNamespace(kin=kin,stage=0,target=np.zeros(3),base=np.zeros(3),rotation=np.eye(3),
    integral=np.zeros(5),goal=np.zeros(5),open_rad=.85,close_rad=.2,fail=reject)
command=CalibratedFaceTeacher.command(t,np.zeros(6),np.zeros(6))
assert np.isfinite(command).all() and command[5]>0
print('Borderline IK residual continues to a finite command')

from privileged_teacher import PrivilegedTeacher
entered=[]
t=SimpleNamespace(stage=3,elapsed=130,root_object=lambda:np.array([0.,0.,.9]),
    object=lambda:np.array([0.,0.,.9]),tcp=lambda:np.array([0.,0.,1.]),
    micro_height=[.9]*11,micro_relative=[np.array([0.,0.,-.1]) for _ in range(11)],
    grasp_root_z=.9,micro_lift_verified=True,events=[],_orange_now=3,enter=entered.append)
PrivilegedTeacher.observe(t)
assert entered==[0], 'A slipped object must replan even after an earlier micro-lift succeeded'
print('Post-micro-lift slip recovery check passed')

from unittest.mock import patch
openings=[]
t=SimpleNamespace(kin=SimpleNamespace(set_opening=openings.append),
    open_rad=.85,measured=lambda:np.array([0,0,0,0,0,.6]),
    tcp=lambda:np.zeros(3))
with patch.object(GeometricTeacher,'enter',lambda self,stage:None):
    CalibratedFaceTeacher.enter(t,1)
assert openings==[.85], 'Descent IK must use the opening actually commanded during descent'
print('Descent planned opening matches commanded opening')

failures=[]
t=SimpleNamespace(stage=5,elapsed=0,stable=0,object=lambda:np.array([0.,0.,.9]),
    tcp=lambda:np.array([0.,0.,.9]),measured=lambda:np.zeros(6),target=np.ones(3),
    lift_initial_z=.9,relative_reference=np.zeros(3),caps=(2701,)*9,fail=failures.append)
for _ in range(8):GeometricTeacher.observe(t)
assert not failures, 'Intentional lowering with fixed relative pose is not slip'
t.object=lambda:np.array([.05,0.,.9])
for _ in range(6):GeometricTeacher.observe(t)
assert failures, 'Actual relative separation must still request recovery'
print('Intentional lowering vs actual grasp separation check passed')
