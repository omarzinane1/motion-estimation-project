"""Outils simples pour le calcul et la visualisation de l'optical flow."""

from __future__ import annotations

import cv2
import numpy as np


def compute_optical_flow(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    method: str = "farneback",
) -> np.ndarray:
    """Calculer le flot optique dense entre deux frames consecutives."""
    if method.lower() != "farneback":
        raise ValueError("Seule la methode 'farneback' est supportee.")

    previous = np.asarray(prev_frame)
    current = np.asarray(next_frame)

    # Conversion en niveaux de gris si les images sont en BGR.
    if previous.ndim == 3:
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    if current.ndim == 3:
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("Les frames doivent etre des images grayscale ou BGR.")
    if previous.shape != current.shape:
        raise ValueError("Les deux frames doivent avoir la meme taille.")

    # Farneback fonctionne de maniere stable avec des images grayscale uint8.
    if previous.dtype != np.uint8:
        previous = previous.astype(np.float32)
        if previous.size > 0 and previous.max() <= 1.0:
            previous = previous * 255.0
        previous = np.clip(previous, 0, 255).astype(np.uint8)

    if current.dtype != np.uint8:
        current = current.astype(np.float32)
        if current.size > 0 and current.max() <= 1.0:
            current = current * 255.0
        current = np.clip(current, 0, 255).astype(np.uint8)

    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def compute_motion_mask(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    threshold: int = 25,
    min_area: int = 500,
    kernel_size: int = 5,
    padding: int = 8,
    max_boxes: int = 1,
    roi_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Segmenter le mouvement par difference d'images et retourner masque + boxes."""
    previous = np.asarray(prev_frame)
    current = np.asarray(next_frame)

    if previous.ndim == 3:
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    if current.ndim == 3:
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("Les frames doivent etre des images grayscale ou BGR.")
    if previous.shape != current.shape:
        raise ValueError("Les deux frames doivent avoir la meme taille.")

    if previous.dtype != np.uint8:
        previous = previous.astype(np.float32)
        if previous.size > 0 and previous.max() <= 1.0:
            previous = previous * 255.0
        previous = np.clip(previous, 0, 255).astype(np.uint8)

    if current.dtype != np.uint8:
        current = current.astype(np.float32)
        if current.size > 0 and current.max() <= 1.0:
            current = current * 255.0
        current = np.clip(current, 0, 255).astype(np.uint8)

    difference = cv2.absdiff(previous, current)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    roi_mask = np.ones_like(mask, dtype=np.uint8) * 255
    if roi_box is not None:
        roi_mask = np.zeros_like(mask, dtype=np.uint8)
        x, y, box_width, box_height = map(int, roi_box)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(mask.shape[1], x + box_width)
        y1 = min(mask.shape[0], y + box_height)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    clean_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + box_width + padding)
        y1 = min(height, y + box_height + padding)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)[:max_boxes]
    if not boxes and roi_box is not None:
        x, y, box_width, box_height = map(int, roi_box)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        boxes = [(x0, y0, x1 - x0, y1 - y0)]

    for x, y, box_width, box_height in boxes:
        clean_mask[y : y + box_height, x : x + box_width] = mask[
            y : y + box_height,
            x : x + box_width,
        ]

    return clean_mask, boxes


def visualize_optical_flow(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 30,
    scale: float = 8.0,
    min_magnitude: float = 1.0,
    percentile: float = 90.0,
    motion_mask: np.ndarray | None = None,
    bounding_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Dessiner des vecteurs de mouvement utiles et retourner une image RGB."""
    if step < 1:
        raise ValueError("step doit etre superieur ou egal a 1.")
    if scale <= 0:
        raise ValueError("scale doit etre strictement positif.")
    if min_magnitude < 0:
        raise ValueError("min_magnitude doit etre positif ou nul.")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile doit etre compris entre 0 et 100.")

    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow doit avoir la forme (hauteur, largeur, 2).")

    image = np.asarray(frame)

    # Preparation d'une image RGB pour l'affichage avec Matplotlib.
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("frame doit etre une image grayscale ou BGR.")

    if canvas.shape[:2] != flow.shape[:2]:
        raise ValueError("frame et flow doivent avoir la meme taille.")

    height, width = flow.shape[:2]
    valid_region = np.ones((height, width), dtype=bool)

    if motion_mask is not None:
        mask = np.asarray(motion_mask)
        if mask.shape[:2] != flow.shape[:2]:
            raise ValueError("motion_mask et flow doivent avoir la meme taille.")
        valid_region &= mask > 0

    if bounding_box is not None:
        x, y, box_width, box_height = map(int, bounding_box)
        box_region = np.zeros((height, width), dtype=bool)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        box_region[y0:y1, x0:x1] = True
        valid_region &= box_region

        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

    y_coords, x_coords = np.mgrid[step // 2 : height : step, step // 2 : width : step]
    region_values = valid_region[y_coords, x_coords]

    sampled_flow = flow[y_coords, x_coords]
    sampled_magnitude = np.hypot(sampled_flow[..., 0], sampled_flow[..., 1])
    valid_magnitudes = sampled_magnitude[region_values]
    if valid_magnitudes.size == 0:
        return canvas

    magnitude_threshold = max(
        min_magnitude,
        float(np.percentile(valid_magnitudes, percentile)),
    )

    for x, y, (dx, dy), magnitude, is_valid in zip(
        x_coords.ravel(),
        y_coords.ravel(),
        sampled_flow.reshape(-1, 2),
        sampled_magnitude.ravel(),
        region_values.ravel(),
    ):
        if not is_valid or magnitude < magnitude_threshold:
            continue

        start_point = (int(x), int(y))
        end_point = (
            int(np.clip(round(x + dx * scale), 0, width - 1)),
            int(np.clip(round(y + dy * scale), 0, height - 1)),
        )
        if start_point == end_point:
            continue

        cv2.arrowedLine(
            canvas,
            start_point,
            end_point,
            color=(0, 255, 0),
            thickness=2,
            tipLength=0.35,
        )

    return canvas


def compute_lucas_kanade_flow(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: float = 7.0,
    block_size: int = 7,
    win_size: tuple[int, int] = (21, 21),
    max_level: int = 3,
    bounding_box: tuple[int, int, int, int] | None = None,
    motion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Suivre des points caracteristiques avec Lucas-Kanade pyramidal."""
    previous = np.asarray(prev_frame)
    current = np.asarray(next_frame)

    if previous.ndim == 3:
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    if current.ndim == 3:
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("Les frames doivent etre des images grayscale ou BGR.")
    if previous.shape != current.shape:
        raise ValueError("Les deux frames doivent avoir la meme taille.")

    if previous.dtype != np.uint8:
        previous = previous.astype(np.float32)
        if previous.size > 0 and previous.max() <= 1.0:
            previous = previous * 255.0
        previous = np.clip(previous, 0, 255).astype(np.uint8)

    if current.dtype != np.uint8:
        current = current.astype(np.float32)
        if current.size > 0 and current.max() <= 1.0:
            current = current * 255.0
        current = np.clip(current, 0, 255).astype(np.uint8)

    height, width = previous.shape[:2]
    feature_mask = np.ones((height, width), dtype=np.uint8) * 255

    if motion_mask is not None:
        mask = np.asarray(motion_mask)
        if mask.shape[:2] != previous.shape[:2]:
            raise ValueError("motion_mask et les frames doivent avoir la meme taille.")
        feature_mask = cv2.bitwise_and(feature_mask, (mask > 0).astype(np.uint8) * 255)

    if bounding_box is not None:
        x, y, box_width, box_height = map(int, bounding_box)
        box_mask = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        box_mask[y0:y1, x0:x1] = 255
        feature_mask = cv2.bitwise_and(feature_mask, box_mask)

    prev_points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=feature_mask,
        blockSize=block_size,
    )
    if prev_points is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        prev_points,
        None,
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    valid = status.ravel() == 1
    return (
        prev_points.reshape(-1, 2)[valid].astype(np.float32),
        next_points.reshape(-1, 2)[valid].astype(np.float32),
    )


def visualize_lucas_kanade_flow(
    frame: np.ndarray,
    prev_points: np.ndarray,
    next_points: np.ndarray,
    scale: float = 1.0,
    min_magnitude: float = 0.2,
    bounding_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Dessiner les trajectoires Lucas-Kanade et retourner une image RGB."""
    if scale <= 0:
        raise ValueError("scale doit etre strictement positif.")
    if min_magnitude < 0:
        raise ValueError("min_magnitude doit etre positif ou nul.")

    image = np.asarray(frame)
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("frame doit etre une image grayscale ou BGR.")

    height, width = canvas.shape[:2]

    if bounding_box is not None:
        x, y, box_width, box_height = map(int, bounding_box)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

    prev_points = np.asarray(prev_points, dtype=np.float32).reshape(-1, 2)
    next_points = np.asarray(next_points, dtype=np.float32).reshape(-1, 2)
    if len(prev_points) != len(next_points):
        raise ValueError("prev_points et next_points doivent avoir la meme longueur.")

    for (x0, y0), (x1, y1) in zip(prev_points, next_points):
        dx = x1 - x0
        dy = y1 - y0
        if np.hypot(dx, dy) < min_magnitude:
            continue

        start_point = (int(round(x0)), int(round(y0)))
        end_point = (
            int(np.clip(round(x0 + dx * scale), 0, width - 1)),
            int(np.clip(round(y0 + dy * scale), 0, height - 1)),
        )
        cv2.line(canvas, start_point, end_point, color=(0, 255, 0), thickness=2)
        cv2.circle(canvas, end_point, radius=3, color=(255, 0, 0), thickness=-1)

    return canvas
