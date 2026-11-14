"""RGB + measured q6 + known cameras only; scene geometry is inferred visually."""
import numpy as np
from policy import VisualPolicy
from stereo import CameraModel, VisualGeometryDetector, reconstruct_support


class StrictVisualPolicy(VisualPolicy):
    def __init__(self, asset, front_camera, wrist_intrinsic, wrist_to_tool):
        super().__init__(asset, front_camera)
        self.detector = VisualGeometryDetector(front_camera)
        self.wrist_intrinsic = np.asarray(wrist_intrinsic)
        self.wrist_to_tool = np.asarray(wrist_to_tool)
        self.previous_q = []
        self.bootstrap_attempts = 0
        # Fixed robot scan pose provides overlapping RGB views with useful
        # baseline. This is a motion preset, not a scene geometry observation.
        self.home = np.deg2rad([0, -50, -20, 90, -90, 43])

    def plan(self, q):
        plate_detection = self.detections['plate']
        plate_radius = plate_detection['radius_m']
        fruit = [d for d in self.detections['oranges']
                 if np.linalg.norm(np.asarray(d['point'])[:2]-self.plate[:2]) > plate_radius]
        if not fruit:
            return False
        fruit.sort(key=lambda d: np.linalg.norm(d['point'][:2]))
        best = None
        normal, offset_plane = self.detector.plane[:3], self.detector.plane[3]
        for detection in fruit:
            point = np.asarray(detection['point'])
            yaw = np.arctan2(point[1], point[0])+np.pi/2
            for offset in (60, -60, 30, -30, 0, -90):
                axis = np.array([np.cos(yaw+np.deg2rad(offset)), np.sin(yaw+np.deg2rad(offset)), 0])
                self.kin.desired_axis = axis
                self.kin.set_opening(self.opening)
                grasp = self.kin.solve(point, q[:5])
                if grasp['position_error_m'] > .006 or grasp['tool_axis_error'] > .18:
                    continue
                frame, _, _ = self.kin.fk(grasp['q'])
                if frame[2, 2] < .93 or abs(grasp['q'][3]) > 1.58:
                    continue
                approach = self.kin.solve(point+[0, 0, .080], grasp['q'])
                if approach['position_error_m'] > .008:
                    continue
                margin = np.inf
                for fraction in np.linspace(0, 1, 7):
                    t, _, _ = self.kin.fk((1-fraction)*approach['q']+fraction*grasp['q'])
                    for _, points in self.geometry.transformed_points(t, self.opening):
                        margin = min(margin, float(np.min(points@normal+offset_plane)))
                cost = grasp['position_error_m']+.04*grasp['tool_axis_error']
                cost += .002*np.linalg.norm(approach['q']-q[:5])+.04*(1-frame[2, 2])
                cost += 10*max(0, .002-margin)
                if best is None or cost < best[0]:
                    best = cost, point, axis, dict(detection=detection, offset_deg=offset,
                        visual_plane_clearance_m=float(margin), grasp_error_m=grasp['position_error_m'])
            if best is not None and best[0] < .025:
                break
        if best is None:
            return False
        _, self.anchor, self.kin.desired_axis, detail = best
        self.drop = self.plate.copy()
        self.drop[:2] += plate_radius*np.array([-.25, (self.completed % 3-1)*.42])
        self.events.append(dict(plan=detail, anchor=self.anchor.tolist(), drop=self.drop.tolist(),
                                geometry_source='RGB_only', visual_plate_radius_m=plate_radius))
        self.attempts += 1
        self.enter('approach', q)
        return True

    def update(self, front_rgb, wrist_rgb, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Expected finite ordered measured q6')
        self.previous_q.append(q.copy())
        self.previous_q = self.previous_q[-8:]
        stable = len(self.previous_q) == 8 and np.ptp(self.previous_q, axis=0).max() < .012
        if self.phase == 'scan' and self.detector.plane is None and stable and self.age > 50 and self.age % 15 == 0:
            tool, _, _ = self.kin.fk(q[:5])
            wrist = CameraModel(self.wrist_intrinsic, tool@self.wrist_to_tool)
            self.bootstrap_attempts += 1
            try:
                plane, detail, points = reconstruct_support(front_rgb, wrist_rgb, self.detector.camera, wrist)
                self.detector.plane, self.detector.detail = plane, detail
                self.events.append(dict(visual_support=detail))
                self.home = VisualPolicy.home.copy()
            except ValueError as exc:
                self.events.append(dict(visual_support_unavailable=str(exc), age=self.age))
        # Internal zero is not a supplied model quantity. Only q error drives PI.
        return super().update(front_rgb, q, np.zeros(5))
