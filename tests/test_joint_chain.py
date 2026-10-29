"""Synthetic software checks; real asset calibration is a separate acceptance."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('numpy','scipy'))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'course/liftcube/src'))
    from starter.joint_chain import JointChainKinematics
    from starter.interfaces.action import USD_LOW_DEG,USD_HIGH_DEG


@unittest.skipUnless(AVAILABLE,'Requires NumPy/SciPy, no simulator')
class JointChainTests(unittest.TestCase):
    def data(self):
        return dict(schema='so101-joint-chain-v1',meters_per_unit=1,asset_sha256='a'*64,
            joints=[dict(name=str(i),axis='Z',lower=float(USD_LOW_DEG[i]),upper=float(USD_HIGH_DEG[i]),
                parent_frame=np.eye(4).tolist(),child_frame=np.eye(4).tolist()) for i in range(5)])

    def test_fk_ik_and_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'chain.json'; data=self.data(); path.write_text(json.dumps(data))
            kin=JointChainKinematics(path,[0,0,-.1],list(map(str,range(5))))
            q=np.array([1,2,3,4,5.]); target=kin.forward_kinematics(q)
            np.testing.assert_allclose(target[:3,3],[0,0,-.1])
            recovered=kin.inverse_kinematics(q,target)
            np.testing.assert_allclose(kin.forward_kinematics(recovered),target,atol=1e-8)
            for field,value in (('upper',111.),('axis','X')):
                data=self.data(); data['joints'][0][field]=value; path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    JointChainKinematics(path,[0,0,-.1],list(map(str,range(5))))


if __name__=='__main__':
    unittest.main()
