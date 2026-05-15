"""Fonctions de prétraitement des images.

Ce module préparera les frames avant la détection des points et le calcul du
flot optique Lucas-Kanade.
"""


def load_image(image_path):
    """Charger une image depuis son chemin."""
    # Cette fonction lira une image de la séquence avec OpenCV.
    pass


def convert_to_gray(image):
    """Convertir une image couleur en niveaux de gris."""
    # Cette fonction préparera l'image pour les traitements basés sur l'intensité.
    pass


def apply_clahe(gray_image):
    """Améliorer localement le contraste d'une image en niveaux de gris."""
    # Cette fonction pourra renforcer les détails utiles au suivi des points.
    pass


def preprocess_image(image_path):
    """Appliquer la chaîne minimale de prétraitement à une image."""
    # Cette fonction combinera le chargement, la conversion et l'amélioration.
    pass
