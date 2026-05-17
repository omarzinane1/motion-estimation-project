import cv2
import matplotlib.pyplot as plt
import numpy as np


def otsu_segmentation(gray_roi):
    """Segmentation binaire de la ROI avec Otsu."""
    if gray_roi.ndim != 2:
        raise ValueError("Otsu attend une ROI en niveaux de gris.")
    _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def adaptive_segmentation(gray_roi):
    """Segmentation locale de la ROI avec un seuil adaptatif gaussien."""
    if gray_roi.ndim != 2:
        raise ValueError("Le seuillage adaptatif attend une ROI en niveaux de gris.")
    return cv2.adaptiveThreshold(
        gray_roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )


def clean_mask(mask, kernel_size=(5, 5), closing_iter=2, opening_iter=1):
    """Nettoie un masque par closing puis opening morphologique."""
    kernel = np.ones(kernel_size, dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=opening_iter)
    return cleaned


def keep_largest_component(mask):
    """Garde la plus grande composante connexe blanche d'un masque."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return ((labels == largest_label).astype(np.uint8) * 255)


def _clip_roi(image_shape, roi):
    x, y, w, h = [int(v) for v in roi]
    height, width = image_shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def create_roi_mask(image_shape, roi):
    """Cree un masque plein limite a la ROI manuelle."""
    x, y, w, h = _clip_roi(image_shape, roi)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def paste_roi_mask_to_full_image(mask_roi, image_shape, roi):
    """Replace un masque calcule dans la ROI dans une image pleine."""
    x, y, w, h = _clip_roi(image_shape, roi)
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y : y + h, x : x + w] = cv2.resize(mask_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    return full_mask


def compare_masks_visualization(gray_roi, otsu_mask, adaptive_mask, save_path=None):
    """Affiche la ROI et les deux masques pour choisir une segmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray_roi, cmap="gray")
    axes[0].set_title("ROI pretraitee")
    axes[1].imshow(otsu_mask, cmap="gray")
    axes[1].set_title("Otsu")
    axes[2].imshow(adaptive_mask, cmap="gray")
    axes[2].set_title("Adaptive Threshold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
