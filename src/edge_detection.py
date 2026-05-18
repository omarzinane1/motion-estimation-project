"""Canny edge-detection helpers used for visual validation."""

from pathlib import Path

import cv2
import numpy as np

from src.detection import extract_roi, validate_bbox


def canny_edges(gray_image, low_threshold=50, high_threshold=150):
    """Apply the Canny detector to a grayscale image."""
    return cv2.Canny(gray_image.astype(np.uint8), low_threshold, high_threshold)


def canny_on_roi(gray_image, bbox, low_threshold=50, high_threshold=150):
    """Apply Canny inside a bounding box."""
    bbox = validate_bbox(bbox, gray_image.shape)
    if bbox is None:
        return None

    roi = extract_roi(gray_image, bbox)
    if roi is None or roi.size == 0:
        return None

    return canny_edges(roi, low_threshold=low_threshold, high_threshold=high_threshold)


def overlay_edges_on_frame(frame_bgr, edges, bbox=None):
    """Overlay Canny edges on a BGR frame in red."""
    if frame_bgr is None:
        return None

    output = frame_bgr.copy()
    if edges is None:
        return output

    if bbox is None:
        edge_mask = edges > 0
        if edge_mask.shape[:2] != output.shape[:2]:
            edge_mask = cv2.resize(
                edge_mask.astype(np.uint8),
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        output[edge_mask] = (0, 0, 255)
        return output

    bbox = validate_bbox(bbox, frame_bgr.shape)
    if bbox is None:
        return output

    x, y, w, h = bbox
    roi_edges = edges
    if roi_edges.shape[:2] != (h, w):
        roi_edges = cv2.resize(roi_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    edge_mask = roi_edges > 0
    roi_output = output[y:y + h, x:x + w]
    roi_output[edge_mask] = (0, 0, 255)
    output[y:y + h, x:x + w] = roi_output
    return output


def save_edge_detection_results(frame, edges, output_path, bbox=None):
    """Save an edge overlay image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_edges_on_frame(frame, edges, bbox=bbox)
    if overlay is None:
        return False
    return cv2.imwrite(str(output_path), overlay)
