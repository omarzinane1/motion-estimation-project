from pathlib import Path

import numpy as np
import pandas as pd


def interpret_direction(dx, dy):
    """Retourne une interpretation simple de la direction du mouvement."""
    eps = 1e-6
    if abs(dx) < eps and abs(dy) < eps:
        return "mouvement quasi nul"
    if abs(dx) > 1.5 * abs(dy):
        return "mouvement vers la droite" if dx > 0 else "mouvement vers la gauche"
    if abs(dy) > 1.5 * abs(dx):
        return "mouvement vers le bas" if dy > 0 else "mouvement vers le haut"
    return "mouvement diagonal"


def compute_motion_analysis(trajectory_df):
    """Calcule dx, dy, vitesse en pixels/frame et direction avec atan2."""
    df = trajectory_df.copy()
    df["dx"] = df["center_x"].diff().fillna(0.0)
    df["dy"] = df["center_y"].diff().fillna(0.0)
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["speed_px_frame"] = df["distance"]
    df["direction_rad"] = np.arctan2(df["dy"], df["dx"])
    df["direction_deg"] = np.degrees(df["direction_rad"])
    df["interpretation"] = [interpret_direction(dx, dy) for dx, dy in zip(df["dx"], df["dy"])]

    columns = [
        "frame_id",
        "center_x",
        "center_y",
        "dx",
        "dy",
        "distance",
        "speed_px_frame",
        "direction_rad",
        "direction_deg",
        "interpretation",
    ]
    return df[columns]


def save_motion_analysis_csv(df, output_path):
    """Sauvegarde l'analyse du mouvement."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

