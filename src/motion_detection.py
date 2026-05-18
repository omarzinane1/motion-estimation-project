"""Frame-difference motion detection helpers."""

from pathlib import Path

import cv2
import numpy as np

from src.detection import extract_roi, validate_bbox


def frame_difference(gray1, gray2):
    """Return the absolute difference between two grayscale frames."""
    return cv2.absdiff(gray1.astype(np.uint8), gray2.astype(np.uint8))


def threshold_motion(diff, threshold=25):
    """Threshold a frame difference image into a binary motion mask."""
    _, motion_mask = cv2.threshold(diff.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)
    return motion_mask


def detect_motion_between_frames(gray1, gray2, threshold=25):
    """Compute frame difference and motion mask between two frames."""
    diff = frame_difference(gray1, gray2)
    motion_mask = threshold_motion(diff, threshold=threshold)
    return {
        "difference": diff,
        "motion_mask": motion_mask,
    }


def motion_detection_on_roi(gray1, gray2, bbox, threshold=25):
    """Apply frame differencing inside a ROI."""
    bbox = validate_bbox(bbox, gray1.shape)
    if bbox is None:
        return {
            "difference": None,
            "motion_mask": None,
        }

    roi1 = extract_roi(gray1, bbox)
    roi2 = extract_roi(gray2, bbox)
    if roi1 is None or roi2 is None or roi1.size == 0 or roi2.size == 0:
        return {
            "difference": None,
            "motion_mask": None,
        }

    return detect_motion_between_frames(roi1, roi2, threshold=threshold)


def overlay_motion_mask(frame_bgr, motion_mask, bbox=None):
    """Overlay a binary motion mask on a BGR frame in cyan."""
    if frame_bgr is None:
        return None

    output = frame_bgr.copy()
    if motion_mask is None:
        return output

    if bbox is None:
        mask = motion_mask > 0
        if mask.shape[:2] != output.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        output[mask] = (0, 255, 255)
        return output

    bbox = validate_bbox(bbox, frame_bgr.shape)
    if bbox is None:
        return output

    x, y, w, h = bbox
    roi_mask = motion_mask
    if roi_mask.shape[:2] != (h, w):
        roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = roi_mask > 0
    roi_output = output[y:y + h, x:x + w]
    roi_output[mask] = (0, 255, 255)
    output[y:y + h, x:x + w] = roi_output
    return output


def save_motion_detection_results(diff, mask, output_dir):
    """Save difference and binary motion-mask images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = True
    if diff is not None:
        saved = cv2.imwrite(str(output_dir / "difference.png"), diff) and saved
    if mask is not None:
        saved = cv2.imwrite(str(output_dir / "motion_mask.png"), mask) and saved
    return saved
