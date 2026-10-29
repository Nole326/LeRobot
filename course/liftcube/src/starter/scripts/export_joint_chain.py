"""Export fixed robot geometry without SimulationApp; requires the USD Python API."""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(args.asset))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Expected a metre-scale robot USD')

    def frame(position, quaternion):
        matrix = Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Quatd(quaternion))
        matrix.SetTranslateOnly(Gf.Vec3d(position))
        return [[float(matrix[j][i]) for j in range(4)] for i in range(4)]

    joints = []
    previous_child = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parent) != 1 or len(child) != 1 or (previous_child is not None and parent[0] != previous_child):
            raise ValueError('Expected an ordered, unbranched SO101 chain')
        previous_child = child[0]
        joints.append(dict(name=prim.GetName(),axis=prim.GetAttribute('physics:axis').Get(),
            lower=prim.GetAttribute('physics:lowerLimit').Get(),upper=prim.GetAttribute('physics:upperLimit').Get(),
            parent_frame=frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()),
            child_frame=frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())))
        if len(joints) == 5:
            break  # The separate jaw is not an arm IK degree of freedom.
    if len(joints) != 5:
        raise ValueError('Expected five SO101 arm joints')
    result = dict(schema='so101-joint-chain-v1',meters_per_unit=1,
        asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),joints=joints)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)


if __name__ == '__main__':
    main()
