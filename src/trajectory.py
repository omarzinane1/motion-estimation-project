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


def filter_points_by_displacement(points_old, points_new, max_displacement=25):
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


def filter_points_by_distance_to_center(points, center, max_distance=60):
    """Garde les points proches du centre courant de l'objet."""
    pts = _as_xy(points)
    if len(pts) == 0 or center is None:
        return np.empty((0, 1, 2), dtype=np.float32)

    center = np.asarray(center, dtype=np.float32)
    distances = np.linalg.norm(pts - center, axis=1)
    keep = distances <= max_distance
    return pts[keep].reshape(-1, 1, 2).astype(np.float32)


def center_distance(previous_center, current_center):
    """Calcule la distance entre deux centres."""
    if previous_center is None or current_center is None:
        return np.inf
    previous = np.asarray(previous_center, dtype=np.float32)
    current = np.asarray(current_center, dtype=np.float32)
    return float(np.linalg.norm(current - previous))


def is_center_jump_valid(previous_center, current_center, max_jump=40):
    """Indique si le saut du centre reste plausible."""
    if current_center is None:
        return False
    if previous_center is None:
        return True
    return center_distance(previous_center, current_center) <= max_jump


def filter_center_jump(previous_center, current_center, max_jump=40):
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


def create_centered_roi(center, image_shape, roi_size=(120, 120)):
    """Cree une ROI rectangulaire centree sur le dernier centre valide."""
    if center is None:
        raise ValueError("Un centre valide est necessaire pour creer une ROI dynamique.")

    roi_w, roi_h = [int(v) for v in roi_size]
    cx, cy = center
    height, width = image_shape[:2]

    x = int(round(cx - roi_w / 2))
    y = int(round(cy - roi_h / 2))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(roi_w, width - x))
    h = max(1, min(roi_h, height - y))
    return x, y, w, h


def smooth_trajectory(trajectory_df, window=5):
    """Ajoute center_x_smooth et center_y_smooth avec une mediane glissante."""
    df = trajectory_df.copy()
    df["center_x_smooth"] = (
        df["center_x"].rolling(window=window, center=True, min_periods=1).median()
    )
    df["center_y_smooth"] = (
        df["center_y"].rolling(window=window, center=True, min_periods=1).median()
    )
    return df


def save_trajectory_csv(trajectory, output_path):
    """Sauvegarde la trajectoire estimee."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trajectory)
    expected = ["frame_id", "center_x", "center_y", "nb_points"]
    optional = ["center_x_smooth", "center_y_smooth"]
    columns = expected + [column for column in optional if column in df.columns]
    df = df[columns]
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

    x_col = "center_x_smooth" if "center_x_smooth" in estimated_df.columns else "center_x"
    y_col = "center_y_smooth" if "center_y_smooth" in estimated_df.columns else "center_y"
    estimated_eval = estimated_df[["frame_id", x_col, y_col]].rename(
        columns={x_col: "center_x", y_col: "center_y"}
    )

    comparison = estimated_eval.merge(gt_df, on="frame_id", suffixes=("_estimated", "_gt"))
    comparison["error_px"] = np.sqrt(
        (comparison["center_x_estimated"] - comparison["center_x_gt"]) ** 2
        + (comparison["center_y_estimated"] - comparison["center_y_gt"]) ** 2
    )
    return comparison
