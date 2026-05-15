"""Fonctions liées au flot optique Lucas-Kanade.

Ce module contiendra les opérations de suivi des points entre deux frames
consécutives à l'aide de la méthode Lucas-Kanade.
"""


def compute_lucas_kanade(prev_gray, next_gray, points):
    """Calculer le déplacement de points entre deux images."""
    # Cette fonction appellera plus tard l'algorithme Lucas-Kanade d'OpenCV.
    pass


def filter_good_points(old_points, new_points, status):
    """Conserver uniquement les points correctement suivis."""
    # Cette fonction filtrera les points valides selon le statut retourné.
    pass
