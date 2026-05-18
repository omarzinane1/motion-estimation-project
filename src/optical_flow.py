"""Lucas-Kanade optical-flow helpers."""

import cv2
import numpy as np


def filter_good_points(old_points, new_points, status):
    """Keep points successfully tracked by Lucas-Kanade."""
    if old_points is None or new_points is None or status is None:
        return None, None

    good_mask = status.ravel() == 1

    if not np.any(good_mask):
        return None, None

    good_old = old_points[good_mask].reshape(-1, 2)
    good_new = new_points[good_mask].reshape(-1, 2)

    return good_old, good_new


def compute_lucas_kanade(prev_gray, next_gray, points):
    """Apply Lucas-Kanade and return successfully tracked points."""
    if points is None or len(points) == 0:
        return None, None

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        points.astype(np.float32),
        None,
        **lk_params,
    )

    good_old, good_new = filter_good_points(points, new_points, status)
    return good_old, good_new


def compute_mean_displacement(good_old, good_new):
    """Compute mean displacement dx, dy."""
    if good_old is None or good_new is None or len(good_old) == 0:
        return None, None

    displacements = good_new - good_old
    mean_dx = float(np.mean(displacements[:, 0]))
    mean_dy = float(np.mean(displacements[:, 1]))

    return mean_dx, mean_dy


def compute_motion_vectors(good_old, good_new):
    """Return local motion vectors as dx, dy rows."""
    if good_old is None or good_new is None or len(good_old) == 0:
        return np.empty((0, 2), dtype=np.float32)

    good_old = np.asarray(good_old, dtype=np.float32).reshape(-1, 2)
    good_new = np.asarray(good_new, dtype=np.float32).reshape(-1, 2)
    return good_new - good_old


def compute_global_motion_from_vectors(vectors):
    """Return the mean global motion from local vectors."""
    vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 2)
    if vectors.size == 0:
        return 0.0, 0.0

    dx_global = float(np.mean(vectors[:, 0]))
    dy_global = float(np.mean(vectors[:, 1]))
    return dx_global, dy_global


def draw_optical_flow_vectors(frame, good_old, good_new):
    """Draw Lucas-Kanade vectors as arrows on a BGR frame."""
    if frame is None:
        return None

    output = frame.copy()
    if good_old is None or good_new is None:
        return output

    old_points = np.asarray(good_old).reshape(-1, 2)
    new_points = np.asarray(good_new).reshape(-1, 2)

    for old_point, new_point in zip(old_points, new_points):
        old_x, old_y = old_point.astype(int)
        new_x, new_y = new_point.astype(int)
        cv2.arrowedLine(
            output,
            (old_x, old_y),
            (new_x, new_y),
            (0, 255, 255),
            2,
            tipLength=0.35,
        )
        cv2.circle(output, (old_x, old_y), 3, (0, 255, 0), -1)
        cv2.circle(output, (new_x, new_y), 3, (0, 0, 255), -1)

    return output


def estimate_motion_field(prev_gray, next_gray, points):
    """Estimate local and global motion with Lucas-Kanade."""
    good_old, good_new = compute_lucas_kanade(prev_gray, next_gray, points)
    vectors = compute_motion_vectors(good_old, good_new)
    dx_global, dy_global = compute_global_motion_from_vectors(vectors)

    return {
        "good_old": good_old,
        "good_new": good_new,
        "vectors": vectors,
        "dx_global": dx_global,
        "dy_global": dy_global,
        "num_points": int(len(vectors)),
    }
