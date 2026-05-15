"""Fonctions simples de prétraitement des images.

Ce module prépare les frames avant la détection des points et le calcul du flot
optique Lucas-Kanade dans les prochaines étapes du projet.
"""

import cv2


def load_image(image_path):
    """Charger une image couleur avec OpenCV."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Image introuvable ou illisible : {image_path}")

    return image


def convert_to_gray(image_bgr):
    """Convertir une image BGR en niveaux de gris."""
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray_image


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Améliorer localement le contraste d'une image."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Réduire légèrement le bruit avec un filtre gaussien."""
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def preprocess_image(image_bgr):
    """Prétraiter une image pour l'estimation de mouvement."""
    gray_image = convert_to_gray(image_bgr)
    enhanced_image = apply_clahe(gray_image)
    preprocessed_image = apply_gaussian_blur(enhanced_image)
    return preprocessed_image
