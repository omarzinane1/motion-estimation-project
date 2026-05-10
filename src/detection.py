"""Classical motion detection utilities.

The functions in this module deliberately use traditional image processing
steps only: filtering, frame differencing, thresholding, morphology, contours,
and bounding boxes. They are designed to be reused by later tracking, optical
flow, and analysis notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    """Container returned by the motion detection pipeline."""

    previous_filtered: np.ndarray
    current_filtered: np.ndarray
    difference: np.ndarray
    mask: np.ndarray
    contours: list[np.ndarray]
    bounding_boxes: list[BoundingBox]


def _validate_kernel_size(kernel_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return a valid odd OpenCV kernel size."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if len(kernel_size) != 2:
        raise ValueError("kernel_size must contain two values.")

    height, width = int(kernel_size[0]), int(kernel_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("kernel_size values must be positive.")
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError("OpenCV kernels must have odd dimensions.")

    return height, width


def _to_float_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale float32 in the [0, 1] range."""
    array = np.asarray(image)

    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim != 2:
        raise ValueError("image must be a grayscale or BGR image.")

    array = array.astype(np.float32)
    if array.size > 0 and array.max() > 1.0:
        array = array / 255.0

    return np.clip(array, 0.0, 1.0)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or BGR image to uint8 for drawing/display."""
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array.copy()

    array = array.astype(np.float32)
    if array.size > 0 and array.max() <= 1.0:
        array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)


def _to_binary_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert any non-zero mask representation to binary uint8."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int | tuple[int, int] = (5, 5),
    sigma: float = 0.0,
) -> np.ndarray:
    """Reduce local noise with a Gaussian filter."""
    kernel_size = _validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, kernel_size, sigma)


def compute_frame_difference(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
) -> np.ndarray:
    """Compute the absolute normalized difference between two frames."""
    previous = _to_float_gray(previous_frame)
    current = _to_float_gray(current_frame)

    if previous.shape != current.shape:
        raise ValueError("Frames must have the same shape.")

    return cv2.absdiff(previous, current)


def frame_difference(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
) -> np.ndarray:
    """Alias for ``compute_frame_difference``."""
    return compute_frame_difference(previous_frame, current_frame)


def threshold_difference(
    difference: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Create a binary motion mask from a normalized difference image."""
    difference = np.asarray(difference, dtype=np.float32)
    if difference.size > 0 and difference.max() > 1.0:
        difference = difference / 255.0

    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def apply_threshold(
    difference: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Alias for ``threshold_difference``."""
    return threshold_difference(difference, threshold=threshold)


def apply_morphology(
    mask: np.ndarray,
    kernel_size: int | tuple[int, int] = (5, 5),
    open_iterations: int = 1,
    close_iterations: int = 2,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """Clean a binary mask with opening, closing, and optional dilation."""
    kernel_size = _validate_kernel_size(kernel_size)
    cleaned = _to_binary_uint8(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    if open_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )
    if close_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )
    if dilate_iterations > 0:
        cleaned = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)

    return cleaned


def apply_morphological_operations(
    mask: np.ndarray,
    kernel_size: int | tuple[int, int] = (5, 5),
    open_iterations: int = 1,
    close_iterations: int = 2,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """Alias for ``apply_morphology``."""
    return apply_morphology(
        mask,
        kernel_size=kernel_size,
        open_iterations=open_iterations,
        close_iterations=close_iterations,
        dilate_iterations=dilate_iterations,
    )


def find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Extract external contours from a binary motion mask."""
    binary_mask = _to_binary_uint8(mask)
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return list(contours)


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Alias for ``find_contours``."""
    return find_contours(mask)


def filter_contours(
    contours: Iterable[np.ndarray],
    min_area: float = 100.0,
    max_area: float | None = None,
    min_width: int = 0,
    min_height: int = 0,
    aspect_ratio_range: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Keep contours that satisfy geometric constraints."""
    filtered: list[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        _, _, width, height = cv2.boundingRect(contour)
        if width < min_width or height < min_height:
            continue

        if aspect_ratio_range is not None:
            min_ratio, max_ratio = aspect_ratio_range
            aspect_ratio = width / height if height else 0.0
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                continue

        filtered.append(contour)

    return filtered


def contours_to_bounding_boxes(
    contours: Iterable[np.ndarray],
) -> list[BoundingBox]:
    """Convert contours to bounding boxes sorted by decreasing area."""
    boxes = [tuple(map(int, cv2.boundingRect(contour))) for contour in contours]
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def get_bounding_boxes(
    contours: Iterable[np.ndarray],
) -> list[BoundingBox]:
    """Alias for ``contours_to_bounding_boxes``."""
    return contours_to_bounding_boxes(contours)


def select_largest_bounding_box(
    bounding_boxes: Sequence[BoundingBox],
) -> BoundingBox | None:
    """Select the largest bounding box by area."""
    if not bounding_boxes:
        return None
    return max(bounding_boxes, key=lambda box: box[2] * box[3])


def merge_bounding_boxes(
    bounding_boxes: Sequence[BoundingBox],
    padding: int = 0,
    image_shape: tuple[int, ...] | None = None,
) -> BoundingBox | None:
    """Merge several boxes into a single enclosing box."""
    if not bounding_boxes:
        return None

    x_min = min(box[0] for box in bounding_boxes) - padding
    y_min = min(box[1] for box in bounding_boxes) - padding
    x_max = max(box[0] + box[2] for box in bounding_boxes) + padding
    y_max = max(box[1] + box[3] for box in bounding_boxes) + padding

    if image_shape is not None:
        image_height, image_width = image_shape[:2]
        x_min = max(0, min(x_min, image_width - 1))
        y_min = max(0, min(y_min, image_height - 1))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))

    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: Sequence[BoundingBox],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes on a copy of an image."""
    canvas = _to_uint8_image(image)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for index, box in enumerate(bounding_boxes):
        x, y, width, height = map(int, box)
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width, y + height),
            color,
            thickness,
        )
        if labels is not None and index < len(labels):
            cv2.putText(
                canvas,
                labels[index],
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def scale_bounding_box(
    bounding_box: Sequence[float],
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> BoundingBox:
    """Scale a bounding box from one image geometry to another."""
    from_height, from_width = from_shape[:2]
    to_height, to_width = to_shape[:2]

    scale_x = to_width / from_width
    scale_y = to_height / from_height

    x, y, width, height = bounding_box
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(width * scale_x)),
        int(round(height * scale_y)),
    )


def scale_bounding_boxes(
    bounding_boxes: Iterable[Sequence[float]],
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
) -> list[BoundingBox]:
    """Scale several bounding boxes between image geometries."""
    return [
        scale_bounding_box(box, from_shape=from_shape, to_shape=to_shape)
        for box in bounding_boxes
    ]


def detect_motion_between_frames(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    gaussian_kernel_size: int | tuple[int, int] = (5, 5),
    gaussian_sigma: float = 0.0,
    threshold: float = 0.05,
    morphology_kernel_size: int | tuple[int, int] = (5, 5),
    open_iterations: int = 1,
    close_iterations: int = 2,
    dilate_iterations: int = 1,
    min_area: float = 100.0,
    max_area: float | None = None,
    min_width: int = 0,
    min_height: int = 0,
    aspect_ratio_range: tuple[float, float] | None = None,
    merge_boxes: bool = False,
) -> DetectionResult:
    """Detect moving regions between two frames with classical CV steps."""
    previous = _to_float_gray(previous_frame)
    current = _to_float_gray(current_frame)

    previous_filtered = apply_gaussian_filter(
        previous,
        kernel_size=gaussian_kernel_size,
        sigma=gaussian_sigma,
    )
    current_filtered = apply_gaussian_filter(
        current,
        kernel_size=gaussian_kernel_size,
        sigma=gaussian_sigma,
    )

    difference = compute_frame_difference(previous_filtered, current_filtered)
    mask = threshold_difference(difference, threshold=threshold)
    mask = apply_morphology(
        mask,
        kernel_size=morphology_kernel_size,
        open_iterations=open_iterations,
        close_iterations=close_iterations,
        dilate_iterations=dilate_iterations,
    )

    contours = find_contours(mask)
    filtered_contours = filter_contours(
        contours,
        min_area=min_area,
        max_area=max_area,
        min_width=min_width,
        min_height=min_height,
        aspect_ratio_range=aspect_ratio_range,
    )
    bounding_boxes = contours_to_bounding_boxes(filtered_contours)

    if merge_boxes and bounding_boxes:
        merged = merge_bounding_boxes(
            bounding_boxes,
            padding=0,
            image_shape=mask.shape,
        )
        bounding_boxes = [merged] if merged is not None else []

    return DetectionResult(
        previous_filtered=previous_filtered,
        current_filtered=current_filtered,
        difference=difference,
        mask=mask,
        contours=filtered_contours,
        bounding_boxes=bounding_boxes,
    )


def detect_moving_objects(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    **kwargs,
) -> DetectionResult:
    """Alias for ``detect_motion_between_frames``."""
    return detect_motion_between_frames(previous_frame, current_frame, **kwargs)


def detect_motion_sequence(
    frames: Sequence[np.ndarray],
    frame_step: int = 1,
    **kwargs,
) -> list[tuple[int, DetectionResult]]:
    """Run motion detection on a sequence of already preprocessed frames."""
    if frame_step < 1:
        raise ValueError("frame_step must be greater than or equal to 1.")
    if len(frames) <= frame_step:
        return []

    detections: list[tuple[int, DetectionResult]] = []
    for current_index in range(frame_step, len(frames)):
        previous_index = current_index - frame_step
        result = detect_motion_between_frames(
            frames[previous_index],
            frames[current_index],
            **kwargs,
        )
        detections.append((current_index, result))

    return detections
