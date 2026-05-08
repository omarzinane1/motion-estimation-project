"""Image preprocessing utilities for motion estimation pipelines."""

import cv2
import numpy as np


def preprocess_image(
    image_bgr,
    target_size=(640, 360),
    clip_limit=2.0,
    tile_grid_size=(8, 8),
):
    """Resize, convert to grayscale, enhance with CLAHE, and normalize."""
    resized = cv2.resize(image_bgr, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe_operator = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced = clahe_operator.apply(gray)
    normalized = enhanced.astype(np.float32) / 255.0
    return normalized


def preprocess_image_from_path(
    image_path,
    target_size=(640, 360),
    clip_limit=2.0,
    tile_grid_size=(8, 8),
):
    """Read an image from disk and preprocess it."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError("Impossible de lire l'image : " + str(image_path))

    return preprocess_image(
        image_bgr,
        target_size=target_size,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def preprocess_sequence(
    frames,
    target_size=(640, 360),
    clip_limit=2.0,
    tile_grid_size=(8, 8),
):
    """Apply ``preprocess_image`` to each frame in a sequence."""
    return [
        preprocess_image(
            frame,
            target_size=target_size,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
        for frame in frames
    ]
