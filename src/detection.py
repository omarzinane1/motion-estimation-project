"""Fonctions d'initialisation de la détection de l'objet.

Ce module préparera la lecture des annotations et la sélection des points
caractéristiques à suivre dans la région de l'objet.
"""


def read_groundtruth(path):
    """Lire les annotations de boîtes englobantes depuis un fichier."""
    # Cette fonction analysera le fichier groundtruth.txt.
    pass


def get_initial_bbox(groundtruth):
    """Extraire la boîte englobante initiale de l'objet."""
    # Cette fonction récupérera la première position connue de l'objet.
    pass


def extract_roi(image, bbox):
    """Extraire la région d'intérêt correspondant à une boîte englobante."""
    # Cette fonction isolera la zone de l'image où se trouve l'objet.
    pass


def detect_features_in_roi(gray_image, bbox):
    """Détecter des points caractéristiques dans la région de l'objet."""
    # Cette fonction préparera les points à suivre par Lucas-Kanade.
    pass
