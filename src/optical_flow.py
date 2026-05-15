"""Fonctions simples pour le flot optique Lucas-Kanade.

Ce module permet de suivre des points caractéristiques entre deux images
successives et de mesurer leur déplacement moyen.
"""

import cv2
import numpy as np


def filter_good_points(old_points, new_points, status):
    """Conserver uniquement les points suivis avec succès."""
    if old_points is None or new_points is None or status is None:
        return None, None

    good_mask = status.ravel() == 1

    if not np.any(good_mask):
        return None, None

    good_old = old_points[good_mask].reshape(-1, 2)
    good_new = new_points[good_mask].reshape(-1, 2)

    return good_old, good_new


def compute_lucas_kanade(prev_gray, next_gray, points):
    """Appliquer Lucas-Kanade et retourner les bons points."""
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
        points,
        None,
        **lk_params,
    )

    good_old, good_new = filter_good_points(points, new_points, status)
    return good_old, good_new


def compute_mean_displacement(good_old, good_new):
    """Calculer le déplacement moyen dx, dy."""
    if good_old is None or good_new is None or len(good_old) == 0:
        return None, None

    displacements = good_new - good_old
    mean_dx = float(np.mean(displacements[:, 0]))
    mean_dy = float(np.mean(displacements[:, 1]))

    return mean_dx, mean_dy
