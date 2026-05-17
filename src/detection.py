import cv2
import numpy as np


def detect_edges_canny(gray, threshold1=50, threshold2=150):
    """Detecte les contours avec Canny."""
    if gray.ndim != 2:
        raise ValueError("Canny attend une image en niveaux de gris.")
    return cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)


def detect_good_features(
    gray,
    mask,
    maxCorners=80,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
):
    """Detecte des points caracteristiques avec Shi-Tomasi."""
    if gray.ndim != 2:
        raise ValueError("goodFeaturesToTrack attend une image en niveaux de gris.")
    if mask is not None and mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        mask=mask,
        blockSize=blockSize,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def filter_points_inside_mask(points, mask):
    """Garde uniquement les points situes dans une zone blanche du masque."""
    if points is None or len(points) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    height, width = mask.shape[:2]
    keep = []
    for x, y in pts:
        xi = int(round(x))
        yi = int(round(y))
        inside = 0 <= xi < width and 0 <= yi < height and mask[yi, xi] > 0
        keep.append(inside)
    keep = np.array(keep, dtype=bool)
    return pts[keep].reshape(-1, 1, 2).astype(np.float32)

