"""Preprocessing utilities for the motion-estimation project.

The module keeps the original helper functions and adds histogram-based
comparison tools used by the notebooks, the pipeline and the Tkinter interface.
"""

import cv2
import numpy as np


def load_image(image_path):
    """Load a BGR image with OpenCV."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Image introuvable ou illisible : {image_path}")

    return image


def convert_to_gray(image_bgr):
    """Convert a BGR image to grayscale."""
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray_image


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Improve local contrast with CLAHE."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced_image = clahe.apply(gray_image.astype(np.uint8))
    return enhanced_image


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Reduce local noise with a Gaussian filter."""
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def preprocess_image(image_bgr):
    """Original preprocessing helper kept for notebook compatibility."""
    gray_image = convert_to_gray(image_bgr)
    enhanced_image = apply_clahe(gray_image)
    preprocessed_image = apply_gaussian_blur(enhanced_image)
    return preprocessed_image


def contrast_stretching(gray_image):
    """Apply linear histogram stretching and return an uint8 image."""
    gray_float = gray_image.astype(np.float32)
    image_min = float(np.min(gray_float))
    image_max = float(np.max(gray_float))

    if image_max == image_min:
        return gray_image.astype(np.uint8).copy()

    stretched = 255.0 * (gray_float - image_min) / (image_max - image_min)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(gray_image):
    """Apply global histogram equalization."""
    return cv2.equalizeHist(gray_image.astype(np.uint8))


def compute_histogram(gray_image):
    """Return the 256-bin grayscale histogram."""
    histogram = cv2.calcHist([gray_image.astype(np.uint8)], [0], None, [256], [0, 256])
    return histogram.ravel()


def preprocess_image_with_method(image_bgr, method="stretching"):
    """Preprocess a frame with a selectable contrast method.

    The order is grayscale conversion, Gaussian blur, then contrast
    enhancement. This keeps the Lucas-Kanade input smoother and less noisy.
    """
    gray = convert_to_gray(image_bgr)
    blur = apply_gaussian_blur(gray)
    method = (method or "none").lower()

    if method == "none":
        enhanced = blur.copy()
    elif method == "clahe":
        enhanced = apply_clahe(blur)
    elif method == "stretching":
        enhanced = contrast_stretching(blur)
    elif method == "equalization":
        enhanced = histogram_equalization(blur)
    else:
        raise ValueError(
            "Methode de pretraitement inconnue. "
            "Choisir parmi: none, clahe, stretching, equalization."
        )

    return {
        "gray": gray,
        "blur": blur,
        "enhanced": enhanced,
    }


def compare_preprocessing_methods(image_bgr):
    """Return the main preprocessing variants for visual comparison."""
    gray = convert_to_gray(image_bgr)
    gaussian = apply_gaussian_blur(gray)

    return {
        "gray": gray,
        "gaussian": gaussian,
        "clahe": apply_clahe(gaussian),
        "stretching": contrast_stretching(gaussian),
        "equalization": histogram_equalization(gaussian),
    }
