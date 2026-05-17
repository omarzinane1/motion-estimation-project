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


def filter_tracked_points(
    points_old,
    points_new,
    status,
    previous_center=None,
    max_point_displacement=25,
    max_distance_to_center=60
):
    """
    Filtre les points suivis par Lucas-Kanade.

    Nous gardons seulement :
    - les points validés par Lucas-Kanade ;
    - les points avec un déplacement raisonnable ;
    - les points proches du centre précédent de la voiture.
    """

    if points_old is None or points_new is None or status is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    points_old = np.asarray(points_old, dtype=np.float32).reshape(-1, 2)
    points_new = np.asarray(points_new, dtype=np.float32).reshape(-1, 2)
    status = np.asarray(status).reshape(-1)

    valid_mask = status == 1

    good_old = points_old[valid_mask]
    good_new = points_new[valid_mask]

    if len(good_new) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    displacements = np.linalg.norm(good_new - good_old, axis=1)
    displacement_mask = displacements <= max_point_displacement

    good_old = good_old[displacement_mask]
    good_new = good_new[displacement_mask]

    if len(good_new) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    if previous_center is not None:
        previous_center = np.asarray(previous_center, dtype=np.float32).reshape(1, 2)
        distances_to_center = np.linalg.norm(good_new - previous_center, axis=1)
        center_mask = distances_to_center <= max_distance_to_center

        good_old = good_old[center_mask]
        good_new = good_new[center_mask]

    return good_old, good_new


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
