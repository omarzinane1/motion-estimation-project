"""Pipeline simple de tracking avec Lucas-Kanade.

Ce fichier ne dépend pas des notebooks. Il permet de lancer le tracking sur un
intervalle de frames et de récupérer la trajectoire, l'analyse et un résumé.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "car" / "car-11"
IMG_PATH = DATASET_PATH / "img"
GROUNDTRUTH_PATH = DATASET_PATH / "groundtruth.txt"
RESULTS_PATH = PROJECT_ROOT / "results"


def _get_image_files(img_path):
    """Récupérer les images triées."""
    image_extensions = [".jpg", ".jpeg", ".png"]

    if not img_path.exists():
        return []

    return sorted([
        file for file in img_path.iterdir()
        if file.suffix.lower() in image_extensions
    ])


def _read_groundtruth(groundtruth_path):
    """Lire le fichier groundtruth."""
    if not groundtruth_path.exists() or groundtruth_path.stat().st_size == 0:
        raise ValueError("Le fichier groundtruth.txt est introuvable ou vide.")

    groundtruth_df = pd.read_csv(
        groundtruth_path,
        header=None,
        sep=r"[,\s]+",
        engine="python",
    )
    groundtruth_df = groundtruth_df.iloc[:, :4]
    groundtruth_df.columns = ["x", "y", "w", "h"]
    groundtruth_df = groundtruth_df.astype(int)

    return groundtruth_df


def _preprocess_image(image_bgr):
    """Prétraiter une image pour Lucas-Kanade."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred


def _detect_points_in_bbox(gray_image, bbox):
    """Détecter des points caractéristiques dans une bounding box."""
    x, y, w, h = [int(value) for value in bbox]
    roi_gray = gray_image[y:y + h, x:x + w]

    if roi_gray.size == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    if points is None:
        return None

    points = points.astype(np.float32)
    points[:, 0, 0] += x
    points[:, 0, 1] += y

    return points


def _compute_center(points):
    """Calculer le centre moyen des points."""
    if points is None or len(points) == 0:
        return None

    points_array = np.asarray(points).reshape(-1, 2)
    center_x = float(np.mean(points_array[:, 0]))
    center_y = float(np.mean(points_array[:, 1]))

    return center_x, center_y


def _bbox_center(bbox):
    """Calculer le centre d'une bounding box."""
    x, y, w, h = [int(value) for value in bbox]
    return x + w / 2, y + h / 2


def _validate_frames(start_frame, end_frame, total_images):
    """Vérifier que l'intervalle de frames est valide."""
    if start_frame < 0:
        raise ValueError("La frame de départ doit être supérieure ou égale à 0.")

    if end_frame <= start_frame:
        raise ValueError("La frame de fin doit être supérieure à la frame de départ.")

    if end_frame > total_images:
        raise ValueError("La frame de fin dépasse le nombre total d'images.")


def run_tracking(start_frame=0, end_frame=100, fps=30):
    """
    Lance le pipeline complet de tracking sur un intervalle de frames.

    Entrées :
    - start_frame : indice de la frame de départ
    - end_frame : indice de la frame de fin
    - fps : nombre de frames par seconde utilisé pour calculer la vitesse

    Sorties :
    - trajectory_df : DataFrame contenant frame, x, y, tracked_points
    - analysis_df : DataFrame contenant frame, x, y, dx, dy, distance,
      speed_px_per_frame, speed_px_per_second, direction_deg
    - summary : dictionnaire contenant un résumé numérique du mouvement
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    fps = float(fps)

    image_files = _get_image_files(IMG_PATH)

    if len(image_files) == 0:
        raise ValueError("Aucune image n'a été trouvée dans data/car/car-11/img.")

    _validate_frames(start_frame, end_frame, len(image_files))

    groundtruth_df = _read_groundtruth(GROUNDTRUTH_PATH)

    if end_frame > len(groundtruth_df):
        raise ValueError("Le fichier groundtruth ne contient pas assez de lignes.")

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    first_image = cv2.imread(str(image_files[start_frame]))

    if first_image is None:
        raise ValueError("La frame de départ n'a pas pu être chargée.")

    initial_bbox = tuple(groundtruth_df.iloc[start_frame].astype(int))
    prev_gray = _preprocess_image(first_image)
    points = _detect_points_in_bbox(prev_gray, initial_bbox)

    if points is None or len(points) == 0:
        raise ValueError("Aucun point caractéristique n'a été détecté dans l'objet.")

    trajectory = []
    trajectory_frames = []
    tracked_points_count = []

    initial_center = _compute_center(points)

    if initial_center is None:
        initial_center = _bbox_center(initial_bbox)

    trajectory.append(initial_center)
    trajectory_frames.append(start_frame)
    tracked_points_count.append(len(points))

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    for frame_index in range(start_frame + 1, end_frame):
        current_image = cv2.imread(str(image_files[frame_index]))

        if current_image is None:
            break

        current_gray = _preprocess_image(current_image)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            current_gray,
            points,
            None,
            **lk_params,
        )

        if new_points is not None and status is not None:
            good_points = new_points[status.ravel() == 1].reshape(-1, 2)
        else:
            good_points = np.empty((0, 2), dtype=np.float32)

        if len(good_points) >= 5:
            center = _compute_center(good_points)
            points = good_points.reshape(-1, 1, 2).astype(np.float32)
            tracked_count = len(points)
        else:
            # Cette réinitialisation évite la perte complète du suivi.
            current_bbox = tuple(groundtruth_df.iloc[frame_index].astype(int))
            points = _detect_points_in_bbox(current_gray, current_bbox)
            center = _bbox_center(current_bbox)
            tracked_count = 0 if points is None else len(points)

            if points is None or len(points) < 5:
                trajectory.append(center)
                trajectory_frames.append(frame_index)
                tracked_points_count.append(tracked_count)
                break

        trajectory.append(center)
        trajectory_frames.append(frame_index)
        tracked_points_count.append(tracked_count)
        prev_gray = current_gray

    trajectory_array = np.asarray(trajectory)

    trajectory_df = pd.DataFrame({
        "frame": trajectory_frames,
        "x": trajectory_array[:, 0],
        "y": trajectory_array[:, 1],
        "tracked_points": tracked_points_count,
    })

    analysis_df = trajectory_df.copy()
    analysis_df["dx"] = analysis_df["x"].diff().fillna(0)
    analysis_df["dy"] = analysis_df["y"].diff().fillna(0)
    analysis_df["distance"] = np.sqrt(analysis_df["dx"] ** 2 + analysis_df["dy"] ** 2)
    analysis_df["speed_px_per_frame"] = analysis_df["distance"]
    analysis_df["speed_px_per_second"] = analysis_df["distance"] * fps
    analysis_df["direction_deg"] = np.degrees(
        np.arctan2(analysis_df["dy"], analysis_df["dx"])
    )

    trajectory_df.to_csv(RESULTS_PATH / "trajectory.csv", index=False)
    analysis_df.to_csv(RESULTS_PATH / "motion_analysis.csv", index=False)

    total_dx = float(analysis_df["dx"].sum())
    total_dy = float(analysis_df["dy"].sum())

    if total_dx > 0:
        horizontal_text = "L'objet se déplace globalement vers la droite."
    elif total_dx < 0:
        horizontal_text = "L'objet se déplace globalement vers la gauche."
    else:
        horizontal_text = "Le déplacement horizontal est faible."

    if total_dy > 0:
        vertical_text = "L'objet descend dans l'image."
    elif total_dy < 0:
        vertical_text = "L'objet monte dans l'image."
    else:
        vertical_text = "Le déplacement vertical est faible."

    direction_values = analysis_df["direction_deg"].iloc[1:]

    summary = {
        "number_of_frames": int(len(analysis_df)),
        "total_distance": float(analysis_df["distance"].sum()),
        "mean_speed": float(analysis_df["speed_px_per_frame"].mean()),
        "max_speed": float(analysis_df["speed_px_per_frame"].max()),
        "mean_direction": float(direction_values.mean()) if len(direction_values) > 0 else 0.0,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "movement_text": horizontal_text + " " + vertical_text,
    }

    return trajectory_df, analysis_df, summary
