"""Task-space labels in the calibrated TCP frame and a joint-limited IK adapter."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..expert.geometric_teacher import Kinematics


class EEAction:
    def __init__(self, asset):
        self.kin = Kinematics(
            asset, tcp=(0.04239292443776763, -0.00119974788459086, -0.09303105684230396)
        )
        self.low, self.high = self.kin.limits.T

    def pose(self, arm):
        t, point, _ = self.kin.fk(arm)
        return point, t[:3, :3]

    def command(self, label, measured):
        p, r = label[:3], Rotation.from_rotvec(label[3:6]).as_matrix()
        q0 = np.clip(measured[:5], self.low + 1e-8, self.high - 1e-8)

        def residual(q):
            pq, rq = self.pose(q)
            angle = Rotation.from_matrix(r @ rq.T).as_rotvec()
            return np.r_[p - pq, 0.06 * angle]

        result = least_squares(
            residual,
            q0,
            bounds=(self.low, self.high),
            max_nfev=24,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        error = residual(result.x)
        return np.r_[result.x, label[6]], (
            np.linalg.norm(error[:3]),
            np.linalg.norm(error[3:]) / 0.06,
        )
