"""Fonctions simples pour extraire et gérer la trajectoire.

Ce module calcule une position représentative de l'objet à partir des points
suivis et permet de sauvegarder ou dessiner la trajectoire obtenue.
"""

from pathlib import Path

import cv2
import numpy as np


def compute_object_center(points):
    """Calculer le centre moyen des points suivis."""
    if points is None or len(points) == 0:
        return None

    points_array = np.asarray(points).reshape(-1, 2)
    center_x = float(np.mean(points_array[:, 0]))
    center_y = float(np.mean(points_array[:, 1]))

    return center_x, center_y


def update_trajectory(trajectory, center):
    """Ajouter une position à la trajectoire."""
    if center is not None:
        trajectory.append(center)

    return trajectory


def save_trajectory(trajectory_df, output_path):
    """Sauvegarder un DataFrame de trajectoire dans un fichier CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(output_path, index=False)


def draw_trajectory_on_image(image, trajectory):
    """Dessiner la trajectoire sur une image."""
    image_with_trajectory = image.copy()

    if trajectory is None or len(trajectory) == 0:
        return image_with_trajectory

    trajectory_points = np.asarray(trajectory, dtype=np.int32)

    for index, point in enumerate(trajectory_points):
        x, y = point
        cv2.circle(image_with_trajectory, (x, y), 3, (0, 0, 255), -1)

        if index > 0:
            previous_x, previous_y = trajectory_points[index - 1]
            cv2.line(
                image_with_trajectory,
                (previous_x, previous_y),
                (x, y),
                (255, 0, 0),
                2,
            )

    return image_with_trajectory
