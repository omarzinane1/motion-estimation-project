"""Fonctions de visualisation des résultats.

Ce module regroupera les outils pour afficher les boîtes, les points suivis, les
vecteurs de mouvement et la trajectoire finale.
"""


def draw_bbox(image, bbox):
    """Dessiner une boîte englobante sur une image."""
    # Cette fonction affichera la position estimée de l'objet.
    pass


def draw_points(image, points):
    """Dessiner des points caractéristiques sur une image."""
    # Cette fonction affichera les points suivis par le flot optique.
    pass


def draw_motion_vectors(image, old_points, new_points):
    """Dessiner les vecteurs de mouvement entre anciens et nouveaux points."""
    # Cette fonction visualisera le déplacement local des points.
    pass


def draw_trajectory(image, trajectory):
    """Dessiner la trajectoire globale de l'objet sur une image."""
    # Cette fonction affichera le chemin suivi par l'objet.
    pass


def save_frame(image, output_path):
    """Sauvegarder une frame annotée dans un fichier."""
    # Cette fonction enregistrera les visualisations dans results/frames_output.
    pass
