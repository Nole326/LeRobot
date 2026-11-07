"""Read-only frozen collision-mesh geometry, NOT a contact sensor.

Convex envelopes are conservative planning proxies; PhysX cooking can differ.
No asset writes, state injection, physics overrides, or policy features.
"""
import numpy as np
from geometric_teacher import transform, rotation


class FingerGeometry:
    def __init__(self, asset):
        from pxr import Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull
        self.stage = Usd.Stage.Open(str(asset))
        j = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath('/so101_new_calib/joints/gripper'))
        self.a = transform(j.GetLocalPos0Attr().Get(), rotation(j.GetLocalRot0Attr().Get()))
        self.b = np.linalg.inv(transform(j.GetLocalPos1Attr().Get(), rotation(j.GetLocalRot1Attr().Get())))
        self.meshes, self.metadata = {}, []
        for body in ('gripper', 'jaw'):
            root = self.stage.GetPrimAtPath('/so101_new_calib/'+body)
            rows = []
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if '/collisions' not in str(prim.GetPath()):
                    continue
                attrs = {v.GetName(): str(v.Get()) for v in prim.GetAttributes()
                         if 'physics' in v.GetName().lower()}
                if prim.GetAppliedSchemas() or attrs:
                    self.metadata.append(dict(path=str(prim.GetPath()),
                        schemas=list(prim.GetAppliedSchemas()), attributes=attrs))
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                xf = UsdGeom.XformCache().ComputeRelativeTransform(prim, root)[0]
                pts = np.asarray([xf.Transform(tuple(p)) for p in
                    UsdGeom.Mesh(prim).GetPointsAttr().Get()], dtype=float)
                if body == 'jaw':
                    canonical = (self.a@self.b@np.c_[pts, np.ones(len(pts))].T).T[:, :3]
                else:
                    canonical = pts
                hull = ConvexHull(canonical)
                rows.append(dict(path=str(prim.GetPath()), points=pts,
                    canonical=canonical, equations=hull.equations))
            assert rows, 'No collision meshes found'
            self.meshes[body] = rows

    def jaw_transform(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        return self.a@transform([0, 0, 0], np.array([[c,-s,0],[s,c,0],[0,0,1]]))@self.b

    def pad_pair(self, z):
        """Inner x faces on canonical y=0,z plane; reject missing slices."""
        pair = []
        for body in ('gripper', 'jaw'):
            intervals = []
            for mesh in self.meshes[body]:
                eq = mesh['equations']
                a, rhs = eq[:,0], -(eq[:,2]*z+eq[:,3])
                if np.any(rhs[np.abs(a)<1e-9] < -1e-8):
                    continue
                lo = np.max(rhs[a < -1e-9]/a[a < -1e-9], initial=-np.inf)
                hi = np.min(rhs[a > 1e-9]/a[a > 1e-9], initial=np.inf)
                if np.isfinite([lo,hi]).all() and lo <= hi:
                    intervals.append((lo,hi))
            if not intervals:
                raise ValueError(f'No {body} surface at z={z}')
            x = max(b for a,b in intervals) if body == 'gripper' else min(a for a,b in intervals)
            p = np.array([x,0.,z])
            if body == 'jaw':
                p = (np.linalg.inv(self.a@self.b)@np.r_[p,1])[:3]
            pair.append(p)
        return pair

    def transformed_points(self, arm_transform, opening):
        rows = []
        for body in ('gripper','jaw'):
            xf = arm_transform if body == 'gripper' else arm_transform@self.jaw_transform(opening)
            for mesh in self.meshes[body]:
                rows.append((mesh['path'], (xf@np.c_[mesh['points'],np.ones(len(mesh['points']))].T).T[:,:3]))
        return rows

    def clearance(self, arm_transform, opening, center, radius, table_z):
        """Conservative envelope table clearance and sphere exclusion.

        The sphere is only an geometric estimate. An overlap is not
        contact confirmation, and clearance does not certify PhysX collision-free.
        """
        minimum = np.inf
        sphere_margin = np.inf
        for body in ('gripper','jaw'):
            xf=arm_transform if body=='gripper' else arm_transform@self.jaw_transform(opening)@np.linalg.inv(self.a@self.b)
            for mesh in self.meshes[body]:
                pts=mesh['canonical']@xf[:3,:3].T+xf[:3,3]
                minimum=min(minimum,float(pts[:,2].min()-table_z))
                equations=mesh['equations']
                normals=equations[:,:3]@xf[:3,:3].T
                offsets=equations[:,3]-normals@xf[:3,3]
                support=np.sqrt(np.sum((normals*np.asarray(radius))**2,axis=1))
                signed=float(np.max(normals@center+offsets-support))
                sphere_margin=min(sphere_margin,signed)
        return dict(table_clearance_m=float(minimum), sphere_plane_margin_m=float(sphere_margin))
