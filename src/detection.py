"""Fonctions simples pour initialiser l'objet à suivre.

Ce module lit les annotations, extrait la région d'intérêt de l'objet et détecte
des points caractéristiques dans cette région.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_groundtruth(groundtruth_path):
    """Lire le fichier groundtruth et retourner x, y, w, h."""
    groundtruth_path = Path(groundtruth_path)

    if not groundtruth_path.exists() or groundtruth_path.stat().st_size == 0:
        return pd.DataFrame(columns=["x", "y", "w", "h"])

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


def get_initial_bbox(groundtruth_df):
    """Retourner la première bounding box sous forme (x, y, w, h)."""
    if groundtruth_df.empty:
        return None

    x, y, w, h = groundtruth_df.iloc[0]
    return int(x), int(y), int(w), int(h)


def extract_roi(image, bbox):
    """Extraire la région d'intérêt correspondant à la bounding box."""
    if bbox is None:
        return None

    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    return roi


def detect_features_in_roi(
    gray_image,
    bbox,
    max_corners=80,
    quality_level=0.01,
    min_distance=7,
    block_size=7,
):
    """Détecter des points dans la ROI et retourner les coordonnées globales."""
    if bbox is None:
        return None

    x, y, w, h = bbox
    roi_gray = gray_image[y:y + h, x:x + w]

    if roi_gray.size == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )

    if points is None:
        return None

    points = points.astype(np.float32)
    points[:, 0, 0] += x
    points[:, 0, 1] += y

    return points
