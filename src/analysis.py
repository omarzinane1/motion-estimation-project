"""Motion-analysis helpers."""

from pathlib import Path

import numpy as np
import pandas as pd


def compute_displacements(trajectory_df):
    """Add dx and dy columns to a trajectory DataFrame."""
    analysis_df = trajectory_df.copy()

    if analysis_df.empty:
        analysis_df["dx"] = []
        analysis_df["dy"] = []
        return analysis_df

    analysis_df["dx"] = analysis_df["x"].diff().fillna(0)
    analysis_df["dy"] = analysis_df["y"].diff().fillna(0)

    return analysis_df


def compute_distance(analysis_df):
    """Add the Euclidean distance column."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["distance"] = []
        return analysis_df

    analysis_df["distance"] = np.sqrt(analysis_df["dx"] ** 2 + analysis_df["dy"] ** 2)

    return analysis_df


def compute_speed(analysis_df, fps=30):
    """Add speed in pixels/frame and optionally pixels/second."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["speed_px_per_frame"] = []
        analysis_df["speed_px_per_second"] = []
        return analysis_df

    if "distance" not in analysis_df.columns:
        analysis_df = compute_distance(analysis_df)

    analysis_df["speed_px_per_frame"] = analysis_df["distance"]
    if fps is None:
        analysis_df["speed_px_per_second"] = np.nan
    else:
        analysis_df["speed_px_per_second"] = analysis_df["distance"] * float(fps)

    return analysis_df


def compute_direction(analysis_df):
    """Add direction in degrees using atan2(dy, dx)."""
    analysis_df = analysis_df.copy()

    if analysis_df.empty:
        analysis_df["direction_deg"] = []
        return analysis_df

    direction_rad = np.arctan2(analysis_df["dy"], analysis_df["dx"])
    analysis_df["direction_deg"] = np.degrees(direction_rad)

    return analysis_df


def summarize_motion(analysis_df):
    """Return a numeric motion summary from a DataFrame."""
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

    if "distance" not in analysis_df.columns:
        analysis_df = compute_distance(analysis_df)
    if "speed_px_per_frame" not in analysis_df.columns:
        analysis_df = compute_speed(analysis_df, fps=None)
    if "direction_deg" not in analysis_df.columns:
        analysis_df = compute_direction(analysis_df)

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


def compute_summary_from_records(records):
    """Compute the final project summary from trajectory records."""
    if records is None or len(records) == 0:
        return {
            "number_of_frames": 0,
            "total_distance": 0.0,
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "mean_direction": 0.0,
            "total_dx": 0.0,
            "total_dy": 0.0,
            "mean_tracked_points": 0.0,
        }

    df = pd.DataFrame(records)
    speeds = df["speed_px_per_frame"] if "speed_px_per_frame" in df else pd.Series(dtype=float)
    directions = df["direction_deg"].iloc[1:] if "direction_deg" in df and len(df) > 1 else pd.Series(dtype=float)

    return {
        "number_of_frames": int(len(df)),
        "total_distance": float(speeds.sum()) if len(speeds) else 0.0,
        "mean_speed": float(speeds.mean()) if len(speeds) else 0.0,
        "max_speed": float(speeds.max()) if len(speeds) else 0.0,
        "mean_direction": float(directions.mean()) if len(directions) else 0.0,
        "total_dx": float(df["dx"].sum()) if "dx" in df else 0.0,
        "total_dy": float(df["dy"].sum()) if "dy" in df else 0.0,
        "mean_tracked_points": float(df["tracked_points"].mean()) if "tracked_points" in df else 0.0,
    }


def save_interpretation_results(summary, output_path):
    """Write a clear text interpretation of the final results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    number_of_frames = int(summary.get("number_of_frames", 0))
    total_distance = float(summary.get("total_distance", 0.0))
    mean_speed = float(summary.get("mean_speed", 0.0))
    max_speed = float(summary.get("max_speed", 0.0))
    mean_direction = float(summary.get("mean_direction", 0.0))
    mean_tracked_points = float(summary.get("mean_tracked_points", 0.0))
    total_dx = float(summary.get("total_dx", 0.0))
    total_dy = float(summary.get("total_dy", 0.0))
    outputs = summary.get("outputs", {})

    if abs(total_dx) >= abs(total_dy):
        global_comment = "La trajectoire globale est principalement horizontale."
    else:
        global_comment = "La trajectoire globale presente une composante verticale importante."

    horizontal = "vers la droite" if total_dx > 0 else "vers la gauche" if total_dx < 0 else "sans deplacement horizontal net"
    vertical = "vers le bas" if total_dy > 0 else "vers le haut" if total_dy < 0 else "sans deplacement vertical net"

    text = f"""Resultats et interpretation du suivi sans groundtruth

Chargement des frames
- Le suivi a ete realise sur {number_of_frames} frames de la sequence voiture.
- Les frames sont traitees dans leur ordre temporel, car le mouvement est estime entre images successives.

Pretraitement et histogramme
- Les images sont converties en niveaux de gris pour travailler sur la luminance.
- Le filtre gaussien reduit le bruit avant segmentation et Lucas-Kanade.
- L'amelioration de contraste aide a separer la voiture de la route, mais une egalisation trop forte peut perturber l'hypothese d'illumination constante.

Segmentation et morphologie
- La voiture est initialisee par ROI manuelle ou bbox fournie, puis segmentee dans cette ROI.
- L'ouverture supprime les petits bruits, la fermeture remplit les petits trous, et la plus grande composante connectee conserve l'objet principal.
- Si la segmentation echoue, le pipeline conserve la derniere bbox connue pour eviter un arret inutile.

Canny
- Les contours Canny servent a verifier visuellement les limites de la voiture.
- Canny n'est pas utilise comme methode principale de tracking.

Detection de mouvement par difference d'images
- La difference absolue entre deux frames met en evidence les pixels qui changent.
- Cette carte confirme les zones mobiles, mais elle reste sensible a l'eclairage et aux variations d'intensite.

Champ de mouvement Lucas-Kanade
- Le champ de mouvement est represente par les vecteurs Lucas-Kanade calcules sur les points de la voiture.
- Nombre moyen de points Lucas-Kanade suivis : {mean_tracked_points:.2f}.
- Les erreurs possibles viennent de la perte de points, du faible contraste, des ombres ou de deplacements trop grands.

Trajectoire globale
- La trajectoire globale est extraite a partir des centres successifs de la bounding box estimee.
- {global_comment} Le deplacement cumule est dx={total_dx:.2f} px et dy={total_dy:.2f} px, donc globalement {horizontal} et {vertical}.

Vitesse
- La vitesse est calculee par sqrt(dx^2 + dy^2) en pixels/frame.
- Distance totale : {total_distance:.2f} pixels.
- Vitesse moyenne : {mean_speed:.2f} pixels/frame.
- Vitesse maximale : {max_speed:.2f} pixels/frame.

Direction
- La direction moyenne est {mean_direction:.2f} degres, calculee avec atan2(dy, dx).
- L'axe y de l'image est oriente vers le bas, ce qui influence l'interpretation angulaire.

Resultats generes
- CSV trajectoire : {outputs.get("trajectory_csv", "results/trajectory.csv")}
- Resume texte : {outputs.get("interpretation_txt", "results/interpretation_results.txt")}
- Segmentation : {outputs.get("segmentation", "results/segmentation")}
- Morphologie : {outputs.get("morphology", "results/morphology")}
- Canny : {outputs.get("edge_detection", "results/edge_detection")}
- Difference d'images : {outputs.get("motion_detection", "results/motion_detection")}
- Champ de mouvement Lucas-Kanade : {outputs.get("optical_flow", "results/optical_flow")}
- Trajectoire : {outputs.get("trajectory", "results/trajectory")}
- Graphes vitesse/direction/positions : {outputs.get("graphs", "results/graphs")}
- Visualisation finale : {outputs.get("final_visualization", "results/final_visualization")}

Limites
- Les resultats peuvent etre influences par l'eclairage, le contraste, les erreurs de segmentation, les ombres, la perte de points Lucas-Kanade et l'absence de calibration metrique.
"""

    output_path.write_text(text, encoding="utf-8")
    return text
