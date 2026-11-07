"""Compare cached-plane clearance to a freshly computed world-space hull."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from finger_geometry import FingerGeometry

rng=np.random.default_rng(18)
g=FingerGeometry.__new__(FingerGeometry)
g.a=np.eye(4);g.b=np.eye(4)
g.meshes={}
for body in ('gripper','jaw'):
    pts=rng.normal(size=(80,3))*.03
    g.meshes[body]=[dict(path=body,points=pts,canonical=pts,equations=ConvexHull(pts).equations)]
for _ in range(30):
    t=np.eye(4);t[:3,:3]=Rotation.random(random_state=rng).as_matrix();t[:3,3]=rng.normal(size=3)
    angle=rng.uniform(.2,.9);center=rng.normal(size=3);radius=np.array([.02,.025,.03])
    expected=[np.inf,np.inf]
    for _,pts in g.transformed_points(t,angle):
        eq=ConvexHull(pts).equations
        expected[0]=min(expected[0],pts[:,2].min()-.1)
        expected[1]=min(expected[1],np.max(eq[:,:3]@center+eq[:,3]-np.sqrt(np.sum((eq[:,:3]*radius)**2,axis=1))))
    actual=g.clearance(t,angle,center,radius,.1)
    np.testing.assert_allclose(list(actual.values()),expected,atol=1e-8,rtol=1e-8)
print('30 rigid-pose cached clearance comparisons passed')
