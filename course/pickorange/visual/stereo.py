"""Metric visual geometry from calibrated RGB views, without scene-size priors."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    intrinsic: np.ndarray
    camera_to_base: np.ndarray

    def __post_init__(self):
        k, t = np.asarray(self.intrinsic), np.asarray(self.camera_to_base)
        if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(np.r_[k.ravel(), t.ravel()]).all():
            raise ValueError('Expected finite camera K(3,3), optical-to-base(4,4)')
        if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError('Invalid camera calibration')
        if not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(t[:3, :3]) < .99:
            raise ValueError('Invalid camera rotation')

    @property
    def projection(self):
        return self.intrinsic@np.linalg.inv(self.camera_to_base)[:3]

    def rays(self, pixels):
        pixels = np.asarray(pixels)
        rays = np.linalg.solve(self.intrinsic, np.c_[pixels, np.ones(len(pixels))].T).T
        return rays@self.camera_to_base[:3, :3].T

    def intersect(self, pixels, plane):
        rays = self.rays(np.atleast_2d(pixels))
        origin = self.camera_to_base[:3, 3]
        normal, offset = np.asarray(plane[:3]), float(plane[3])
        denom = rays@normal
        if np.any(abs(denom) < 1e-8):
            raise ValueError('Visual support plane parallel to optical ray')
        scales = -(origin@normal+offset)/denom
        if np.any(scales <= 0):
            raise ValueError('Visual support plane behind camera')
        points = origin+scales[:, None]*rays
        return points if np.asarray(pixels).ndim == 2 else points[0]


def triangulate(first, second, pixels_first, pixels_second):
    """Known optical projection matrices set metric scale; reject weak geometry."""
    import cv2
    a, b = np.asarray(pixels_first, float), np.asarray(pixels_second, float)
    if not len(a):
        return np.empty((0, 3)), np.zeros(0, bool)
    homogeneous = cv2.triangulatePoints(first.projection, second.projection, a.T, b.T).T
    finite = abs(homogeneous[:, 3]) > 1e-9
    points = homogeneous[:, :3]/np.where(finite, homogeneous[:, 3], 1)[:, None]
    rays_a, rays_b = first.rays(a), second.rays(b)
    cosine = np.sum(rays_a*rays_b, axis=1)/(np.linalg.norm(rays_a, axis=1)*np.linalg.norm(rays_b, axis=1))
    valid = finite & (cosine < np.cos(np.deg2rad(3))) & np.isfinite(points).all(axis=1)
    for camera, observed in ((first, a), (second, b)):
        projected = np.c_[points, np.ones(len(points))]@camera.projection.T
        valid &= projected[:, 2] > .01
        pixels = projected[:, :2]/np.maximum(projected[:, 2:], 1e-8)
        valid &= np.linalg.norm(pixels-observed, axis=1) < 2.
    return points, valid


def fit_plane(points):
    """Dominant visually reconstructed support plane; no supplied height/normal."""
    points = np.asarray(points)
    if len(points) < 18:
        raise ValueError('Need at least 18 triangulated visual features')
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(350):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        if np.linalg.norm(normal) < 1e-8:
            continue
        normal /= np.linalg.norm(normal)
        inliers = abs((points-a)@normal) < .0025
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 15:
        raise ValueError('Insufficient visual plane consensus')
    subset = points[best]
    center = subset.mean(0)
    _, singular, vt = np.linalg.svd(subset-center, full_matrices=False)
    if singular[1] < .025:
        raise ValueError('Visual plane points lack spatial extent')
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    plane = np.r_[normal, -center@normal]
    return plane, dict(inliers=int(best.sum()), points=len(points), singular_values=singular.tolist(),
                       residual_rms_m=float(np.sqrt(np.mean(((subset-center)@normal)**2))))


def reconstruct_support(front_rgb, wrist_rgb, front, wrist):
    import cv2
    # Fine wood-grain features have weak contrast in this renderer; retaining
    # those features is necessary for cross-camera metric reconstruction.
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.008)
    features = []
    for rgb in (front_rgb, wrist_rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Robot/fruit colors are excluded so the plane comes from scene texture.
        excluded = cv2.inRange(hsv, (0, 155, 120), (45, 255, 255))
        features.append(sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 255-excluded))
    (ka, da), (kb, db) = features
    if da is None or db is None:
        raise ValueError('Insufficient RGB texture')
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .68*b.distance]
    a = np.array([ka[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    b = np.array([kb[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    points, valid = triangulate(front, wrist, a, b)
    # Robot reach envelope, not a scene/object position prior.
    valid &= np.linalg.norm(points, axis=1) < .8
    plane, detail = fit_plane(points[valid])
    detail.update(matches=len(matches), triangulated=int(valid.sum()), plane=plane.tolist())
    return plane, detail, points[valid]


class VisualGeometryDetector:
    """RGB silhouettes projected using a visually estimated support plane.

    The spherical contact approximation estimates fruit radius from pixels;
    no metric fruit diameter or table height is supplied.
    """
    def __init__(self, camera):
        self.camera = camera
        self.plane = None
        self.detail = None

    def detect(self, rgb):
        import cv2
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 HWC RGB')
        if self.plane is None:
            return dict(oranges=[], plate=None)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        fruit = cv2.inRange(hsv, (2, 155, 65), (24, 255, 255))
        fruit = cv2.morphologyEx(fruit, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        plate_mask = cv2.inRange(hsv, (0, 0, 125), (179, 65, 255))
        plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        detections = []
        for kind, mask, lo, hi in (('orange', fruit, 180, 7000), ('plate', plate_mask, 4500, 70000)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not lo < area < hi or len(contour) < 12:
                    continue
                center, axes, angle = cv2.fitEllipse(contour)
                fill = area/(np.pi*axes[0]*axes[1]/4)
                if min(axes) < 10 or max(axes)/min(axes) > 2.2 or not .55 < fill < 1.25:
                    continue
                on_plane = self.camera.intersect(center, self.plane)
                # Project silhouette contour to infer metric size from vision.
                cloud = self.camera.intersect(contour[::max(1, len(contour)//80), 0], self.plane)
                distances = np.linalg.norm(cloud-on_plane, axis=1)
                apparent_radius = float(np.median(distances))
                point = on_plane
                radius = apparent_radius
                if kind == 'orange':
                    camera_height = abs(self.camera.camera_to_base[:3, 3]@self.plane[:3]+self.plane[3])
                    radius = apparent_radius/(1+apparent_radius/camera_height)
                    elevated = self.plane.copy(); elevated[3] -= radius
                    point = self.camera.intersect(center, elevated)
                if np.linalg.norm(point) > .8:
                    continue
                detections.append(dict(kind=kind, pixel=list(center), axes=list(axes), angle=angle,
                    area=float(area), point=point.tolist(), radius_m=radius, fill=float(fill)))
        plate = max((x for x in detections if x['kind'] == 'plate'), key=lambda x:x['area'], default=None)
        return dict(oranges=[x for x in detections if x['kind'] == 'orange'], plate=plate)

    def annotate(self, rgb, detections):
        from perception import Detector
        return Detector.annotate(self, rgb, detections)
