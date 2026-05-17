"""Pipeline simple de tracking par segmentation et Lucas-Kanade.

Le groundtruth n'est pas utilise pour detecter ou suivre la voiture. Il peut
seulement servir a comparer la trajectoire estimee a la fin.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detection import (
    clean_mask,
    compare_masks,
    detect_canny,
    detect_features_in_mask,
    preprocess_roi,
    read_groundtruth,
    segment_adaptive,
    segment_otsu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "car" / "car-11"
IMG_PATH = DATASET_PATH / "img"
GROUNDTRUTH_PATH = DATASET_PATH / "groundtruth.txt"
RESULTS_PATH = PROJECT_ROOT / "results"
MANUAL_BBOX = (535, 295, 220, 110)


def _get_image_files(img_path):
    """Recuperer les images triees."""
    image_extensions = [".jpg", ".jpeg", ".png"]

    if not img_path.exists():
        return []

    return sorted([
        file for file in img_path.iterdir()
        if file.suffix.lower() in image_extensions
    ])


def _preprocess_image(image_bgr):
    """Pretraiter une image pour Lucas-Kanade."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred


def _compute_center(points):
    """Calculer le centre moyen des points."""
    if points is None or len(points) == 0:
        return None

    points_array = np.asarray(points).reshape(-1, 2)
    center_x = float(np.mean(points_array[:, 0]))
    center_y = float(np.mean(points_array[:, 1]))

    return center_x, center_y


def _clip_bbox(bbox, image_shape):
    """Limiter une bbox aux dimensions de l'image."""
    x, y, w, h = [int(value) for value in bbox]
    height, width = image_shape[:2]

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))

    return x, y, w, h


def _bbox_around_center(center, image_shape, base_size):
    """Creer une petite ROI autour de la derniere position estimee."""
    width, height = base_size
    cx, cy = center
    search_w = int(width * 1.25)
    search_h = int(height * 1.25)
    x = int(cx - search_w / 2)
    y = int(cy - search_h / 2)

    return _clip_bbox((x, y, search_w, search_h), image_shape)


def _detect_points_by_segmentation(image_bgr, bbox):
    """Segmenter une ROI et detecter les points dans le meilleur masque."""
    bbox = _clip_bbox(bbox, image_bgr.shape)
    x, y, w, h = bbox
    roi_bgr = image_bgr[y:y + h, x:x + w]
    gray_roi = preprocess_roi(roi_bgr)

    if gray_roi is None:
        return None, None, None, None, None

    mask_otsu = segment_otsu(gray_roi)
    mask_adaptive = segment_adaptive(gray_roi, block_size=31, C=5)
    mask_otsu_clean = clean_mask(mask_otsu)
    mask_adaptive_clean = clean_mask(mask_adaptive)

    points_otsu = detect_features_in_mask(gray_roi, mask_otsu_clean)
    points_adaptive = detect_features_in_mask(gray_roi, mask_adaptive_clean)
    best_method = compare_masks(
        mask_otsu_clean,
        mask_adaptive_clean,
        points_otsu,
        points_adaptive,
    )

    if best_method == "Adaptive":
        best_mask = mask_adaptive_clean
        best_points = points_adaptive
    else:
        best_mask = mask_otsu_clean
        best_points = points_otsu

    if best_points is None or len(best_points) == 0:
        return None, best_mask, best_method, gray_roi, bbox

    global_points = best_points.copy().astype(np.float32)
    global_points[:, 0, 0] += x
    global_points[:, 0, 1] += y

    return global_points, best_mask, best_method, gray_roi, bbox


def _save_initialization_outputs(image_bgr, bbox, mask, points, method_name):
    """Sauvegarder les resultats principaux de l'initialisation."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        cv2.imwrite(str(RESULTS_PATH / "initial_best_mask.png"), mask)
        edges = detect_canny(mask, 50, 150)
        cv2.imwrite(str(RESULTS_PATH / "initial_canny_edges.png"), edges)

    if image_bgr is not None and points is not None:
        output = image_bgr.copy()
        for point in np.asarray(points).reshape(-1, 2):
            px, py = point.astype(int)
            cv2.circle(output, (px, py), 3, (0, 0, 255), -1)
        cv2.imwrite(str(RESULTS_PATH / "initial_points.png"), output)

    if mask is not None and points is not None:
        np.savez(
            RESULTS_PATH / "initialization_data.npz",
            manual_bbox=np.asarray(bbox, dtype=np.int32),
            best_mask=mask,
            best_points=points,
            best_method_name=np.asarray(method_name),
        )


def _validate_frames(start_frame, end_frame, total_images):
    """Verifier que l'intervalle de frames est valide."""
    if start_frame < 0:
        raise ValueError("La frame de depart doit etre superieure ou egale a 0.")

    if end_frame <= start_frame:
        raise ValueError("La frame de fin doit etre superieure a la frame de depart.")

    if end_frame > total_images:
        raise ValueError("La frame de fin depasse le nombre total d'images.")


def _compare_with_groundtruth(trajectory_df):
    """Comparer la trajectoire estimee avec le groundtruth, a la fin seulement."""
    groundtruth_df = read_groundtruth(GROUNDTRUTH_PATH)

    if groundtruth_df.empty or trajectory_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in trajectory_df.iterrows():
        frame_index = int(row["frame"])
        if frame_index >= len(groundtruth_df):
            continue

        gt_x, gt_y, gt_w, gt_h = groundtruth_df.iloc[frame_index]
        gt_center_x = float(gt_x + gt_w / 2)
        gt_center_y = float(gt_y + gt_h / 2)
        error = float(np.sqrt((row["x"] - gt_center_x) ** 2 + (row["y"] - gt_center_y) ** 2))

        rows.append({
            "frame": frame_index,
            "estimated_x": float(row["x"]),
            "estimated_y": float(row["y"]),
            "groundtruth_x": gt_center_x,
            "groundtruth_y": gt_center_y,
            "error": error,
        })

    comparison_df = pd.DataFrame(rows)

    if not comparison_df.empty:
        comparison_df.to_csv(RESULTS_PATH / "trajectory_comparison.csv", index=False)

    return comparison_df


def run_tracking(start_frame=0, end_frame=100, fps=30, manual_bbox=MANUAL_BBOX):
    """
    Lancer le tracking par segmentation + Lucas-Kanade.

    Le groundtruth n'est pas utilise pour initialiser les points ni pour les
    reinitialiser. Il est seulement lu a la fin pour calculer une erreur de
    comparaison si le fichier existe.
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    fps = float(fps)

    image_files = _get_image_files(IMG_PATH)

    if len(image_files) == 0:
        raise ValueError("Aucune image n'a ete trouvee dans data/car/car-11/img.")

    _validate_frames(start_frame, end_frame, len(image_files))
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    first_image = cv2.imread(str(image_files[start_frame]))

    if first_image is None:
        raise ValueError("La frame de depart n'a pas pu etre chargee.")

    points, best_mask, best_method, _, used_bbox = _detect_points_by_segmentation(
        first_image,
        manual_bbox,
    )

    if points is None or len(points) == 0:
        raise ValueError("Aucun point caracteristique n'a ete detecte par segmentation.")

    _save_initialization_outputs(first_image, used_bbox, best_mask, points, best_method)

    trajectory = []
    trajectory_frames = []
    tracked_points_count = []
    prev_gray = _preprocess_image(first_image)
    center = _compute_center(points)

    trajectory.append(center)
    trajectory_frames.append(start_frame)
    tracked_points_count.append(len(points))

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    base_size = (used_bbox[2], used_bbox[3])

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
            search_bbox = _bbox_around_center(center, current_image.shape, base_size)
            points, _, _, _, _ = _detect_points_by_segmentation(current_image, search_bbox)
            tracked_count = 0 if points is None else len(points)

            if points is None or len(points) < 5:
                trajectory.append(center)
                trajectory_frames.append(frame_index)
                tracked_points_count.append(tracked_count)
                break

            center = _compute_center(points)

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

    trajectory_df.to_csv(RESULTS_PATH / "trajectory_estimated.csv", index=False)
    analysis_df.to_csv(RESULTS_PATH / "motion_analysis_estimated.csv", index=False)
    comparison_df = _compare_with_groundtruth(trajectory_df)

    total_dx = float(analysis_df["dx"].sum())
    total_dy = float(analysis_df["dy"].sum())

    if total_dx > 0:
        horizontal_text = "L'objet se deplace globalement vers la droite."
    elif total_dx < 0:
        horizontal_text = "L'objet se deplace globalement vers la gauche."
    else:
        horizontal_text = "Le deplacement horizontal est faible."

    if total_dy > 0:
        vertical_text = "L'objet descend dans l'image."
    elif total_dy < 0:
        vertical_text = "L'objet monte dans l'image."
    else:
        vertical_text = "Le deplacement vertical est faible."

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
        "tracking_method": "segmentation + Lucas-Kanade",
        "initial_segmentation_method": best_method,
        "groundtruth_usage": "comparaison seulement",
        "mean_tracking_error": (
            float(comparison_df["error"].mean()) if not comparison_df.empty else None
        ),
    }

    return trajectory_df, analysis_df, summary
