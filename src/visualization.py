"""Fonctions simples de visualisation des résultats.

Les images manipulées par OpenCV sont en BGR. Pour l'affichage avec Matplotlib,
il faut convertir les images en RGB dans les notebooks.
"""

from pathlib import Path

import cv2
import numpy as np


def draw_bbox(image, bbox):
    """Dessiner une bounding box sur une image."""
    if image is None:
        return None

    output = image.copy()

    if bbox is None:
        return output

    x, y, w, h = [int(value) for value in bbox]
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output


def draw_points(image, points):
    """Dessiner des points sur une image."""
    if image is None:
        return None

    output = image.copy()

    if points is None or len(points) == 0:
        return output

    points_array = np.asarray(points).reshape(-1, 2)

    for point in points_array:
        x, y = point.astype(int)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

    return output


def draw_motion_vectors(image, old_points, new_points):
    """Dessiner les vecteurs de mouvement entre deux ensembles de points."""
    if image is None:
        return None

    output = image.copy()

    if old_points is None or new_points is None:
        return output

    old_points = np.asarray(old_points).reshape(-1, 2)
    new_points = np.asarray(new_points).reshape(-1, 2)

    for old_point, new_point in zip(old_points, new_points):
        old_x, old_y = old_point.astype(int)
        new_x, new_y = new_point.astype(int)
        cv2.line(output, (old_x, old_y), (new_x, new_y), (255, 0, 0), 2)
        cv2.circle(output, (old_x, old_y), 3, (0, 255, 0), -1)
        cv2.circle(output, (new_x, new_y), 3, (0, 0, 255), -1)

    return output


def draw_trajectory(image, trajectory):
    """Dessiner la trajectoire sur une image."""
    if image is None:
        return None

    output = image.copy()

    if trajectory is None or len(trajectory) == 0:
        return output

    trajectory_points = np.asarray(trajectory).reshape(-1, 2).astype(int)

    for index, point in enumerate(trajectory_points):
        x, y = point
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

        if index > 0:
            prev_x, prev_y = trajectory_points[index - 1]
            cv2.line(output, (prev_x, prev_y), (x, y), (255, 0, 0), 2)

    return output


def draw_current_center(image, center):
    """Dessiner le centre actuel de l'objet."""
    if image is None:
        return None

    output = image.copy()

    if center is None:
        return output

    x, y = [int(value) for value in center]
    cv2.circle(output, (x, y), 6, (0, 255, 255), -1)

    return output


def save_frame(image, output_path):
    """Sauvegarder une frame."""
    if image is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return cv2.imwrite(str(output_path), image)


def create_tracking_video(image_files, trajectory_df, output_path, fps=20):
    """Créer une vidéo simple avec la trajectoire progressive."""
    if image_files is None or len(image_files) == 0:
        return False

    if trajectory_df is None or len(trajectory_df) == 0:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = None
    for image_file in image_files:
        first_frame = cv2.imread(str(image_file))
        if first_frame is not None:
            break

    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        return False

    trajectory = []
    written_frames = 0

    for _, row in trajectory_df.iterrows():
        frame_index = int(row["frame"]) if "frame" in trajectory_df.columns else written_frames

        if frame_index < 0 or frame_index >= len(image_files):
            continue

        image = cv2.imread(str(image_files[frame_index]))

        if image is None:
            continue

        center = (float(row["x"]), float(row["y"]))
        trajectory.append(center)

        output = draw_trajectory(image, trajectory)
        output = draw_current_center(output, center)
        writer.write(output)
        written_frames += 1

    writer.release()

    return written_frames > 0
