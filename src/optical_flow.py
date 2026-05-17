import cv2
import numpy as np


DEFAULT_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def compute_lucas_kanade(prev_gray, curr_gray, points, lk_params=None):
    """Calcule le flot optique sparse avec Lucas-Kanade pyramidal."""
    if points is None or len(points) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, np.empty((0, 1), dtype=np.uint8), np.empty((0, 1), dtype=np.float32)

    params = DEFAULT_LK_PARAMS.copy()
    if lk_params:
        params.update(lk_params)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    points_new, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, **params)
    if points_new is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, np.empty((0, 1), dtype=np.uint8), np.empty((0, 1), dtype=np.float32)
    return points_new.astype(np.float32), status, error


def filter_valid_flow(points_old, points_new, status):
    """Filtre les points correctement suivis par Lucas-Kanade."""
    if points_old is None or points_new is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    old = np.asarray(points_old, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(points_new, dtype=np.float32).reshape(-1, 2)
    valid = status.reshape(-1).astype(bool)
    valid &= np.isfinite(old).all(axis=1)
    valid &= np.isfinite(new).all(axis=1)
    return old[valid].reshape(-1, 1, 2), new[valid].reshape(-1, 1, 2)


def compute_displacements(points_old, points_new):
    """Retourne dx, dy et distance pour chaque point suivi."""
    old = np.asarray(points_old, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(points_new, dtype=np.float32).reshape(-1, 2)
    displacement = new - old
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    distance = np.sqrt(dx**2 + dy**2)
    return dx, dy, distance


def draw_motion_vectors(frame, points_old, points_new):
    """Dessine les vecteurs de mouvement sur une copie de la frame."""
    output = frame.copy()
    old = np.asarray(points_old, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(points_new, dtype=np.float32).reshape(-1, 2)

    for (x0, y0), (x1, y1) in zip(old, new):
        start = (int(round(x0)), int(round(y0)))
        end = (int(round(x1)), int(round(y1)))
        cv2.arrowedLine(output, start, end, (0, 0, 255), 2, tipLength=0.35)
        cv2.circle(output, start, 3, (0, 255, 0), -1)
        cv2.circle(output, end, 3, (255, 0, 0), -1)
    return output

