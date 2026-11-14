"""RGB detections and calibrated plane intersections; no simulator imports."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    camera_to_base: np.ndarray  # ROS optical frame: x right, y down, z forward.
    table_z: float = .030

    def __post_init__(self):
        k, t = np.asarray(self.intrinsic), np.asarray(self.camera_to_base)
        if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(np.r_[k.ravel(), t.ravel(), self.table_z]).all():
            raise ValueError("Expected finite K(3,3), camera_to_base(4,4), table_z")
        if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError("Invalid camera calibration")
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(t[:3, :3]) < .99:
            raise ValueError("Camera rotation must be proper orthonormal")

    def intersect(self, pixel, height):
        ray = self.camera_to_base[:3, :3] @ np.linalg.solve(self.intrinsic, np.r_[pixel, 1.])
        origin = self.camera_to_base[:3, 3]
        if abs(ray[2]) < 1e-8:
            raise ValueError("Ray parallel to support plane")
        distance = (height-origin[2])/ray[2]
        if distance <= 0:
            raise ValueError("Support plane behind camera")
        return origin + distance*ray


class Detector:
    """HSV + contour ellipses; known support plane and fruit-size priors."""
    def __init__(self, calibration):
        self.calibration = calibration

    def detect(self, rgb):
        import cv2
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 HWC RGB")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        oranges = cv2.inRange(hsv, (2, 155, 65), (24, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 125), (179, 65, 255))
        kernel = np.ones((3, 3), np.uint8)
        oranges = cv2.morphologyEx(oranges, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        found = []
        for kind, mask, area_min, area_max, height in (
            ('orange', oranges, 180, 7000, self.calibration.table_z+.026),
            ('plate', white, 4500, 70000, self.calibration.table_z+.012)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not area_min < area < area_max or len(contour) < 12:
                    continue
                center, axes, angle = cv2.fitEllipse(contour)
                if min(axes) < 10 or max(axes)/min(axes) > 2.2:
                    continue
                fill = area / (np.pi*axes[0]*axes[1]/4)
                if not .55 < fill < 1.25:
                    continue
                point = self.calibration.intersect(center, height)
                if np.linalg.norm(point[:2]) > .55:
                    continue
                found.append(dict(kind=kind, pixel=list(center), axes=list(axes), angle=angle,
                                  area=float(area), point=point.tolist(), fill=float(fill)))
        plate = max((x for x in found if x['kind'] == 'plate'), key=lambda x:x['area'], default=None)
        fruit = [x for x in found if x['kind'] == 'orange']
        return dict(oranges=fruit, plate=plate)

    def annotate(self, rgb, detections):
        import cv2
        result = rgb.copy()
        for d in detections['oranges'] + ([detections['plate']] if detections['plate'] else []):
            color = (0, 255, 0) if d['kind'] == 'orange' else (0, 160, 255)
            cv2.ellipse(result, (tuple(d['pixel']), tuple(d['axes']), d['angle']), color, 2)
            text = f"{d['kind']} {d['point'][0]:.3f},{d['point'][1]:.3f}"
            cv2.putText(result, text, tuple(int(v) for v in d['pixel']), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        return result
