"""Data loading utilities for motion estimation sequences."""

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _numeric_sort_key(filename):
    stem = Path(filename).stem
    if stem.isdigit():
        return int(stem)
    return stem


def load_frames(sequence_path):
    """Load sequence frames from the ``img`` folder in numeric order."""
    image_dir = Path(sequence_path) / "img"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    image_files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files = sorted(image_files, key=lambda path: _numeric_sort_key(path.name))

    frames = []
    for image_path in image_files:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError("Impossible de lire l'image : " + str(image_path))
        frames.append(image_bgr)

    return frames


def load_groundtruth(sequence_path):
    """Load bounding boxes from ``groundtruth.txt`` as a list of rows."""
    groundtruth_path = Path(sequence_path) / "groundtruth.txt"
    if not groundtruth_path.is_file():
        raise FileNotFoundError(f"Groundtruth file not found: {groundtruth_path}")

    groundtruth = np.loadtxt(groundtruth_path, delimiter=",")
    if groundtruth.ndim == 1:
        groundtruth = groundtruth.reshape(1, -1)

    return groundtruth.tolist()
