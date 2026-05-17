from pathlib import Path

import numpy as np
import pandas as pd


def _as_xy(points):
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def robust_center(points):
    """Calcule le centre robuste par mediane des points suivis."""
    pts = _as_xy(points)
    if len(pts) == 0:
        return None
    center = np.median(pts, axis=0)
    return float(center[0]), float(center[1])


def filter_points_by_displacement(points_old, points_new, max_displacement=30):
    """Supprime les points dont le deplacement entre deux frames est trop grand."""
    old = _as_xy(points_old)
    new = _as_xy(points_new)
    if len(old) == 0 or len(new) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    distances = np.linalg.norm(new - old, axis=1)
    keep = distances <= max_displacement
    return (
        old[keep].reshape(-1, 1, 2).astype(np.float32),
        new[keep].reshape(-1, 1, 2).astype(np.float32),
    )


def filter_points_by_distance_to_center(points, center, max_distance=70):
    """Garde les points proches du centre courant de l'objet."""
    pts = _as_xy(points)
    if len(pts) == 0 or center is None:
        return np.empty((0, 1, 2), dtype=np.float32)

    center = np.asarray(center, dtype=np.float32)
    distances = np.linalg.norm(pts - center, axis=1)
    keep = distances <= max_distance
    return pts[keep].reshape(-1, 1, 2).astype(np.float32)


def filter_center_jump(previous_center, current_center, max_jump=80):
    """Rejette un centre courant si le saut avec le centre precedent est trop grand."""
    if current_center is None:
        return previous_center
    if previous_center is None:
        return current_center

    previous = np.asarray(previous_center, dtype=np.float32)
    current = np.asarray(current_center, dtype=np.float32)
    if np.linalg.norm(current - previous) > max_jump:
        return previous_center
    return current_center


def save_trajectory_csv(trajectory, output_path):
    """Sauvegarde la trajectoire estimee."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trajectory)
    expected = ["frame_id", "center_x", "center_y", "nb_points"]
    df = df[expected]
    df.to_csv(output_path, index=False)
    return df


def load_trajectory_csv(path):
    """Charge une trajectoire sauvegardee."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier trajectoire introuvable: {csv_path}")
    return pd.read_csv(csv_path)


def compute_groundtruth_centers(groundtruth_path):
    """Calcule les centres des boites groundtruth pour l'evaluation finale."""
    path = Path(groundtruth_path)
    if not path.exists():
        raise FileNotFoundError(f"Groundtruth introuvable: {path}")

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError("groundtruth.txt doit contenir au moins 4 colonnes: x,y,w,h.")

    df = df.iloc[:, :4].copy()
    df.columns = ["x", "y", "w", "h"]
    df["frame_id"] = np.arange(len(df))
    df["center_x"] = df["x"] + df["w"] / 2.0
    df["center_y"] = df["y"] + df["h"] / 2.0
    return df[["frame_id", "center_x", "center_y", "x", "y", "w", "h"]]


def compare_trajectory_with_groundtruth(estimated, groundtruth):
    """Compare la trajectoire estimee avec les centres groundtruth."""
    estimated_df = pd.DataFrame(estimated).copy()
    gt_df = pd.DataFrame(groundtruth).copy()

    comparison = estimated_df.merge(gt_df, on="frame_id", suffixes=("_estimated", "_gt"))
    comparison["error_px"] = np.sqrt(
        (comparison["center_x_estimated"] - comparison["center_x_gt"]) ** 2
        + (comparison["center_y_estimated"] - comparison["center_y_gt"]) ** 2
    )
    return comparison

