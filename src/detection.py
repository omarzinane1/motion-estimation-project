"""Fonctions simples pour segmenter et initialiser l'objet a suivre.

Le suivi principal ne depend pas du groundtruth. Nous utilisons une ROI
manuelle, puis nous segmentons cette ROI pour detecter des points
caracteristiques sur la voiture.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_groundtruth(groundtruth_path):
    """Lire le groundtruth.

    Utilise seulement pour comparaison / evaluation, jamais pour initialiser ou
    suivre l'objet.
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
    """Retourner la premiere bbox du groundtruth pour comparaison seulement."""
    if groundtruth_df.empty:
        return None

    x, y, w, h = groundtruth_df.iloc[0]
    return int(x), int(y), int(w), int(h)


def extract_roi(image, bbox):
    """Extraire la region d'interet correspondant a une bounding box."""
    if image is None or bbox is None:
        return None

    x, y, w, h = [int(value) for value in bbox]
    return image[y:y + h, x:x + w]


def preprocess_roi(roi_bgr):
    """Pretraite une ROI : grayscale, CLAHE, GaussianBlur."""
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blurred


def segment_otsu(gray_roi):
    """Segmente la ROI avec Otsu."""
    if gray_roi is None or gray_roi.size == 0:
        return None

    _, mask_otsu = cv2.threshold(
        gray_roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return mask_otsu


def segment_adaptive(gray_roi, block_size=31, C=5):
    """Segmente la ROI avec un seuillage adaptatif."""
    if gray_roi is None or gray_roi.size == 0:
        return None

    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray_roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )


def clean_mask(mask, kernel_size=(5, 5), close_iter=2, open_iter=1):
    """Nettoie un masque binaire avec closing puis opening."""
    if mask is None or mask.size == 0:
        return None

    kernel = np.ones(kernel_size, np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    return cleaned


def detect_canny(mask, low_threshold=50, high_threshold=150):
    """Detecte les contours avec Canny."""
    if mask is None or mask.size == 0:
        return None

    return cv2.Canny(mask, low_threshold, high_threshold)


def detect_features_in_mask(
    gray_roi,
    mask,
    max_corners=80,
    quality_level=0.01,
    min_distance=7,
    block_size=7,
):
    """Detecte les points caracteristiques dans un masque."""
    if gray_roi is None or gray_roi.size == 0:
        return None

    if mask is None or mask.size == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        gray_roi,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=mask,
    )

    if points is None:
        return None

    return points.astype(np.float32)


def _points_count(points):
    """Compter les points detectes."""
    return 0 if points is None else len(points)


def _mask_score(mask, points):
    """Calculer un score simple pour comparer deux masques."""
    if mask is None or mask.size == 0:
        return -1

    area_ratio = cv2.countNonZero(mask) / float(mask.size)
    points_count = _points_count(points)

    if area_ratio < 0.03 or area_ratio > 0.85:
        return -1

    area_penalty = abs(area_ratio - 0.35) * 20
    return points_count - area_penalty


def compare_masks(mask_otsu, mask_adaptive, points_otsu, points_adaptive):
    """Compare simplement les deux masques et retourne la meilleure methode.

    Criteres simples :
    - nombre de points detectes ;
    - aire du masque raisonnable ;
    - eviter un masque vide ou trop grand.
    """
    score_otsu = _mask_score(mask_otsu, points_otsu)
    score_adaptive = _mask_score(mask_adaptive, points_adaptive)

    if score_adaptive > score_otsu:
        return "Adaptive"

    return "Otsu"


def detect_features_in_roi(
    gray_image,
    bbox,
    max_corners=80,
    quality_level=0.01,
    min_distance=7,
    block_size=7,
):
    """Detecter des points dans une ROI et retourner les coordonnees globales."""
    if gray_image is None or bbox is None:
        return None

    x, y, w, h = [int(value) for value in bbox]
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
