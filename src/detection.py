"""Classical object initialization, segmentation and morphology helpers.

The historical groundtruth helpers are kept only for backward-compatible
imports. The final workflow initializes the car with a manual ROI or a user
provided bounding box, then estimates masks from image processing only.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_groundtruth(groundtruth_path):
    """Read a groundtruth file.

    Kept for old notebooks only. The final pipeline never calls this function.
    """
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
    """Return the first bbox from a DataFrame.

    Kept for compatibility only. The final workflow does not use groundtruth.
    """
    if groundtruth_df.empty:
        return None

    x, y, w, h = groundtruth_df.iloc[0]
    return int(x), int(y), int(w), int(h)


def extract_roi(image, bbox):
    """Extract the region of interest described by ``bbox``."""
    if bbox is None:
        return None

    x, y, w, h = [int(value) for value in bbox]
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
    """Detect Shi-Tomasi points inside a ROI and return global coordinates."""
    bbox = validate_bbox(bbox, gray_image.shape)
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


def select_initial_roi(frame_bgr):
    """Select the initial car ROI manually with OpenCV."""
    if frame_bgr is None:
        return None

    try:
        bbox = cv2.selectROI(
            "Selection ROI voiture",
            frame_bgr,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Selection ROI voiture")
    except cv2.error:
        return None

    x, y, w, h = [int(value) for value in bbox]
    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def validate_bbox(bbox, image_shape):
    """Validate and clip a bbox to the image boundaries."""
    if bbox is None:
        return None

    if len(bbox) != 4:
        return None

    height, width = image_shape[:2]
    x, y, w, h = [int(round(float(value))) for value in bbox]

    if w <= 0 or h <= 0:
        return None

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x2 = max(x + 1, min(x + w, width))
    y2 = max(y + 1, min(y + h, height))
    w = x2 - x
    h = y2 - y

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def otsu_threshold(gray_image, invert=False):
    """Apply Otsu thresholding."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray_image.astype(np.uint8), 0, 255, flag | cv2.THRESH_OTSU)
    return mask


def adaptive_threshold(gray_image, block_size=31, C=5, invert=False):
    """Apply adaptive Gaussian thresholding."""
    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray_image.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        flag,
        block_size,
        C,
    )


def segment_roi(gray_image, bbox, method="otsu", invert=False):
    """Segment the car inside a ROI with Otsu or adaptive thresholding."""
    bbox = validate_bbox(bbox, gray_image.shape)
    if bbox is None:
        return {"roi": None, "mask": None}

    roi = extract_roi(gray_image, bbox)
    if roi is None or roi.size == 0:
        return {"roi": roi, "mask": None}

    method = (method or "otsu").lower()
    if method == "otsu":
        mask = otsu_threshold(roi, invert=invert)
    elif method == "adaptive":
        mask = adaptive_threshold(roi, invert=invert)
    else:
        raise ValueError("Methode de segmentation inconnue. Choisir otsu ou adaptive.")

    return {
        "roi": roi,
        "mask": mask,
    }


def create_structuring_element(kernel_size=(3, 3), shape="ellipse"):
    """Create a morphology structuring element."""
    shape = (shape or "ellipse").lower()
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "rectangle": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shape_map.get(shape, cv2.MORPH_ELLIPSE), kernel_size)


def erode_mask(mask, kernel_size=(3, 3)):
    """Erode a binary mask."""
    if mask is None:
        return None
    kernel = create_structuring_element(kernel_size)
    return cv2.erode(mask, kernel, iterations=1)


def dilate_mask(mask, kernel_size=(3, 3)):
    """Dilate a binary mask."""
    if mask is None:
        return None
    kernel = create_structuring_element(kernel_size)
    return cv2.dilate(mask, kernel, iterations=1)


def open_mask(mask, kernel_size=(3, 3)):
    """Apply morphological opening: erosion then dilation."""
    if mask is None:
        return None
    kernel = create_structuring_element(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def close_mask(mask, kernel_size=(3, 3)):
    """Apply morphological closing: dilation then erosion."""
    if mask is None:
        return None
    kernel = create_structuring_element(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def clean_segmentation_mask(mask, kernel_size=(3, 3)):
    """Clean a segmentation mask with opening followed by closing."""
    if mask is None:
        return None

    opened = open_mask(mask, kernel_size=kernel_size)
    closed = close_mask(opened, kernel_size=kernel_size)
    return closed


def largest_connected_component(mask):
    """Keep the largest foreground component in a binary mask."""
    if mask is None:
        return None

    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))
    largest = np.zeros_like(mask, dtype=np.uint8)
    largest[labels == largest_label] = 255
    return largest


def bbox_from_mask(mask, parent_bbox=None):
    """Compute a bounding box from the largest foreground component."""
    component = largest_connected_component(mask)
    if component is None or np.count_nonzero(component) == 0:
        return None

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    if parent_bbox is not None:
        parent_x, parent_y, _, _ = [int(value) for value in parent_bbox]
        x += parent_x
        y += parent_y

    return int(x), int(y), int(w), int(h)


def segment_and_clean_car(gray_image, bbox, method="otsu", invert=False):
    """Full ROI segmentation pipeline for the car."""
    bbox = validate_bbox(bbox, gray_image.shape)
    if bbox is None:
        return {
            "roi": None,
            "raw_mask": None,
            "clean_mask": None,
            "largest_component": None,
            "segmented_bbox": None,
        }

    segmented = segment_roi(gray_image, bbox, method=method, invert=invert)
    roi = segmented["roi"]
    raw_mask = segmented["mask"]
    clean_mask = clean_segmentation_mask(raw_mask, kernel_size=(5, 5))
    largest_component = largest_connected_component(clean_mask)
    segmented_bbox = bbox_from_mask(largest_component, parent_bbox=bbox)

    if segmented_bbox is not None:
        segmented_bbox = validate_bbox(segmented_bbox, gray_image.shape)

    return {
        "roi": roi,
        "raw_mask": raw_mask,
        "clean_mask": clean_mask,
        "largest_component": largest_component,
        "segmented_bbox": segmented_bbox,
    }


def filter_features_by_mask(points, bbox, mask):
    """Keep only global feature points that fall inside the ROI mask."""
    if points is None or bbox is None or mask is None:
        return points

    x, y, w, h = [int(value) for value in bbox]
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    kept_points = []

    for px, py in points_array:
        local_x = int(round(px - x))
        local_y = int(round(py - y))
        if 0 <= local_x < w and 0 <= local_y < h and mask[local_y, local_x] > 0:
            kept_points.append([px, py])

    if not kept_points:
        return None

    return np.asarray(kept_points, dtype=np.float32).reshape(-1, 1, 2)
