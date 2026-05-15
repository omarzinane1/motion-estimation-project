"""Fonctions simples d'analyse du mouvement.

Ce module calcule les déplacements, la distance, la vitesse et la direction à
partir de la trajectoire de l'objet.
"""

import numpy as np


def compute_displacements(trajectory_df):
    """Ajouter les colonnes dx et dy."""
    analysis_df = trajectory_df.copy()

    if analysis_df.empty:
        analysis_df["dx"] = []
        analysis_df["dy"] = []
        return analysis_df

    analysis_df["dx"] = analysis_df["x"].diff().fillna(0)
    analysis_df["dy"] = analysis_df["y"].diff().fillna(0)

    return analysis_df


def compute_distance(analysis_df):
    """Ajouter la colonne distance."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["distance"] = []
        return analysis_df

    analysis_df["distance"] = np.sqrt(analysis_df["dx"] ** 2 + analysis_df["dy"] ** 2)

    return analysis_df


def compute_speed(analysis_df, fps=30):
    """Ajouter les vitesses en pixels/frame et pixels/seconde."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["speed_px_per_frame"] = []
        analysis_df["speed_px_per_second"] = []
        return analysis_df

    analysis_df["speed_px_per_frame"] = analysis_df["distance"]
    analysis_df["speed_px_per_second"] = analysis_df["distance"] * fps

    return analysis_df


def compute_direction(analysis_df):
    """Ajouter la direction du mouvement en degrés."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["direction_deg"] = []
        return analysis_df

    direction_rad = np.arctan2(analysis_df["dy"], analysis_df["dx"])
    analysis_df["direction_deg"] = np.degrees(direction_rad)

    return analysis_df


def summarize_motion(analysis_df):
    """Retourner un résumé numérique du mouvement."""
    if analysis_df.empty:
        return {
            "number_of_frames": 0,
            "total_distance": 0,
            "mean_speed": 0,
            "max_speed": 0,
            "mean_direction": 0,
            "total_dx": 0,
            "total_dy": 0,
        }

    direction_values = analysis_df["direction_deg"].iloc[1:]

    return {
        "number_of_frames": int(len(analysis_df)),
        "total_distance": float(analysis_df["distance"].sum()),
        "mean_speed": float(analysis_df["speed_px_per_frame"].mean()),
        "max_speed": float(analysis_df["speed_px_per_frame"].max()),
        "mean_direction": float(direction_values.mean()) if len(direction_values) > 0 else 0,
        "total_dx": float(analysis_df["dx"].sum()),
        "total_dy": float(analysis_df["dy"].sum()),
    }
