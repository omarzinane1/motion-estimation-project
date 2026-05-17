"""Interface Tkinter pedagogique pour le projet d'estimation du mouvement."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline import run_tracking  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "car" / "car-11"
IMG_PATH = DATASET_PATH / "img"

DISPLAY_WIDTH = 620
TRAJECTORY_VIEW_WIDTH = 620
TRAJECTORY_VIEW_HEIGHT = 420
MAX_MOTION_VECTORS = 30
MOTION_ROI_WIDTH = 320
MOTION_ROI_HEIGHT = 220
RECENT_TRACKING_POINTS = 45


def get_image_files():
    """Recuperer les images du dataset."""
    image_extensions = [".jpg", ".jpeg", ".png"]

    if not IMG_PATH.exists():
        return []

    return sorted([
        file for file in IMG_PATH.iterdir()
        if file.suffix.lower() in image_extensions
    ])


def interpret_direction(dx, dy, threshold=0.5):
    """Interprete une direction dans le repere image."""
    if abs(dx) < threshold and abs(dy) < threshold:
        return "mouvement tres faible"
    if dx > 0 and abs(dy) < threshold:
        return "vers la droite"
    if dx < 0 and abs(dy) < threshold:
        return "vers la gauche"
    if dy > 0 and abs(dx) < threshold:
        return "vers le bas"
    if dy < 0 and abs(dx) < threshold:
        return "vers le haut"
    if dx > 0 and dy > 0:
        return "vers la droite et le bas"
    if dx > 0 and dy < 0:
        return "vers la droite et le haut"
    if dx < 0 and dy > 0:
        return "vers la gauche et le bas"
    if dx < 0 and dy < 0:
        return "vers la gauche et le haut"

    return "mouvement tres faible"


def _empty_points():
    """Retourner un tableau de points vide au format OpenCV."""
    return np.empty((0, 2), dtype=np.float32)


def _as_points(points):
    """Convertir une collection de points en tableau Nx2."""
    if points is None:
        return _empty_points()

    points_array = np.asarray(points, dtype=np.float32)

    if points_array.size == 0:
        return _empty_points()

    return points_array.reshape(-1, 2)


def _clip_roi(center, image_shape, width=MOTION_ROI_WIDTH, height=MOTION_ROI_HEIGHT):
    """Creer une ROI autour du centre de l'objet pour la visualisation."""
    image_height, image_width = image_shape[:2]

    if center is None:
        return 0, 0, image_width, image_height

    center_x, center_y = center
    x = int(round(center_x - width / 2))
    y = int(round(center_y - height / 2))
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))

    return x, y, width, height


def compute_motion_field_for_display(prev_frame, current_frame, previous_center=None):
    """
    Calculer des vecteurs Lucas-Kanade pour l'affichage du champ sparse.

    Le pipeline principal n'est pas modifie : cette fonction sert uniquement a
    afficher les deplacements de points caracteristiques entre deux frames.
    """
    if prev_frame is None or current_frame is None:
        return _empty_points(), _empty_points()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(prev_gray.shape, dtype=np.uint8)
    x, y, width, height = _clip_roi(previous_center, prev_frame.shape)
    mask[y:y + height, x:x + width] = 255

    old_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=90,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=mask,
    )

    if old_points is None or len(old_points) == 0:
        return _empty_points(), _empty_points()

    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        current_gray,
        old_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    if new_points is None or status is None:
        return _empty_points(), _empty_points()

    valid = status.ravel() == 1

    return old_points[valid].reshape(-1, 2), new_points[valid].reshape(-1, 2)


def draw_motion_field_view(frame, old_points, new_points, current_center=None, max_vectors=30):
    """
    Affiche le champ de mouvement sparse :
    - fleches entre anciens et nouveaux points
    - points suivis
    - centre actuel de l'objet
    """
    output = frame.copy()
    old_points = _as_points(old_points)
    new_points = _as_points(new_points)
    vector_count = min(len(old_points), len(new_points))

    cv2.putText(
        output,
        "Champ de mouvement sparse estime par Lucas-Kanade",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if vector_count == 0:
        cv2.putText(
            output,
            "Champ de mouvement non disponible pour cette frame",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        old_points = old_points[:vector_count]
        new_points = new_points[:vector_count]

        if vector_count > max_vectors:
            indices = np.linspace(0, vector_count - 1, max_vectors).astype(int)
            old_points = old_points[indices]
            new_points = new_points[indices]

        for old_point, new_point in zip(old_points, new_points):
            if np.linalg.norm(new_point - old_point) < 1.0:
                continue

            old_x, old_y = old_point.astype(int)
            new_x, new_y = new_point.astype(int)

            cv2.arrowedLine(
                output,
                (old_x, old_y),
                (new_x, new_y),
                (255, 220, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.28,
            )
            cv2.circle(output, (old_x, old_y), 3, (0, 180, 255), -1)
            cv2.circle(output, (new_x, new_y), 4, (255, 220, 0), -1)

    if current_center is not None:
        current_x, current_y = np.asarray(current_center).astype(int)
        cv2.circle(output, (current_x, current_y), 9, (0, 255, 255), -1)
        cv2.circle(output, (current_x, current_y), 12, (0, 0, 0), 2)
        cv2.putText(
            output,
            "Current",
            (current_x + 14, current_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        "Fleches = deplacement des points suivis",
        (20, output.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )

    return output


def _map_trajectory_point(point, min_x, min_y, scale, offset_x, offset_y):
    """Projeter un point de trajectoire dans le canvas 2D."""
    x = int(round(offset_x + (float(point[0]) - min_x) * scale))
    y = int(round(offset_y + (float(point[1]) - min_y) * scale))

    return x, y


def draw_trajectory_global_view(trajectory_points, current_index, width=620, height=420):
    """
    Dessine la trajectoire globale dans une image/canvas.
    Start reste fixe.
    Current evolue.
    La trajectoire rouge se construit progressivement.
    """
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    trajectory_points = _as_points(trajectory_points)

    cv2.putText(
        canvas,
        "Trajectoire globale estimee",
        (22, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (45, 45, 45),
        2,
        cv2.LINE_AA,
    )

    if len(trajectory_points) == 0:
        cv2.putText(canvas, "Aucune trajectoire disponible", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
        return canvas

    current_index = min(max(current_index, 0), len(trajectory_points) - 1)
    start_point = trajectory_points[0]
    current_point = trajectory_points[current_index]
    end_point = trajectory_points[-1]
    partial_trajectory = trajectory_points[:current_index + 1]

    padding_left = 58
    padding_right = 34
    padding_top = 58
    padding_bottom = 54
    available_width = width - padding_left - padding_right
    available_height = height - padding_top - padding_bottom

    min_x = float(np.min(trajectory_points[:, 0]))
    max_x = float(np.max(trajectory_points[:, 0]))
    min_y = float(np.min(trajectory_points[:, 1]))
    max_y = float(np.max(trajectory_points[:, 1]))
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)
    scale = min(available_width / range_x, available_height / range_y)
    draw_width = range_x * scale
    draw_height = range_y * scale
    offset_x = padding_left + (available_width - draw_width) / 2.0
    offset_y = padding_top + (available_height - draw_height) / 2.0

    plot_x1 = padding_left
    plot_y1 = padding_top
    plot_x2 = width - padding_right
    plot_y2 = height - padding_bottom
    cv2.rectangle(canvas, (plot_x1, plot_y1), (plot_x2, plot_y2), (205, 205, 205), 1)

    for grid_index in range(1, 4):
        gx = plot_x1 + grid_index * (plot_x2 - plot_x1) // 4
        gy = plot_y1 + grid_index * (plot_y2 - plot_y1) // 4
        cv2.line(canvas, (gx, plot_y1), (gx, plot_y2), (226, 226, 226), 1)
        cv2.line(canvas, (plot_x1, gy), (plot_x2, gy), (226, 226, 226), 1)

    mapped_partial = [
        _map_trajectory_point(point, min_x, min_y, scale, offset_x, offset_y)
        for point in partial_trajectory
    ]

    for index in range(1, len(mapped_partial)):
        cv2.line(canvas, mapped_partial[index - 1], mapped_partial[index], (0, 0, 255), 3, cv2.LINE_AA)

    for point in mapped_partial:
        cv2.circle(canvas, point, 3, (0, 0, 255), -1)

    mapped_start = _map_trajectory_point(start_point, min_x, min_y, scale, offset_x, offset_y)
    mapped_current = _map_trajectory_point(current_point, min_x, min_y, scale, offset_x, offset_y)
    mapped_end = _map_trajectory_point(end_point, min_x, min_y, scale, offset_x, offset_y)

    cv2.circle(canvas, mapped_start, 8, (0, 175, 0), -1)
    cv2.circle(canvas, mapped_start, 11, (0, 0, 0), 2)
    cv2.putText(canvas, "Start", (mapped_start[0] + 10, mapped_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 145, 0), 2, cv2.LINE_AA)

    cv2.circle(canvas, mapped_end, 8, (255, 0, 0), -1)
    cv2.circle(canvas, mapped_end, 11, (0, 0, 0), 2)
    cv2.putText(canvas, "End", (mapped_end[0] + 10, mapped_end[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 0, 0), 2, cv2.LINE_AA)

    cv2.circle(canvas, mapped_current, 9, (0, 255, 255), -1)
    cv2.circle(canvas, mapped_current, 12, (0, 0, 0), 2)
    cv2.putText(canvas, "Current", (mapped_current[0] + 10, mapped_current[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 145, 145), 2, cv2.LINE_AA)

    cv2.putText(canvas, "x", (plot_x2 + 10, plot_y2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y", (plot_x1 - 22, plot_y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Repere image : y vers le bas", (22, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)

    legend_x = max(22, width - 228)
    legend_y = 58
    cv2.putText(canvas, "Vert = depart", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 145, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Jaune = position actuelle", (legend_x, legend_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 145, 145), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Rouge = trajectoire", (legend_x, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Bleu = arrivee", (legend_x, legend_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 0, 0), 1, cv2.LINE_AA)

    return canvas


def draw_tracking_video_view(frame, trajectory_points, current_index):
    """Dessiner une frame de tracking simple pour le bouton video."""
    output = frame.copy()
    trajectory_points = _as_points(trajectory_points)

    if len(trajectory_points) == 0:
        return output

    current_index = min(max(current_index, 0), len(trajectory_points) - 1)
    recent_start = max(0, current_index - RECENT_TRACKING_POINTS)
    recent_points = trajectory_points[recent_start:current_index + 1].astype(int)

    for index in range(1, len(recent_points)):
        cv2.line(output, tuple(recent_points[index - 1]), tuple(recent_points[index]), (0, 0, 255), 3, cv2.LINE_AA)

    current_x, current_y = trajectory_points[current_index].astype(int)
    cv2.circle(output, (current_x, current_y), 9, (0, 255, 255), -1)
    cv2.circle(output, (current_x, current_y), 12, (0, 0, 0), 2)
    cv2.putText(output, "Current", (current_x + 14, current_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "Video tracking : Current + trajectoire recente", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return output


class TrackingApp:
    """Interface Tkinter organisee selon les trois objectifs du projet."""

    def __init__(self, root):
        self.root = root
        self.root.title("Estimation du mouvement d'un objet unique")
        self.root.minsize(1280, 860)

        self.image_files = get_image_files()
        self.trajectory_df = None
        self.analysis_df = None
        self.summary = None
        self.current_index = 0
        self.after_id = None
        self.delay = 70
        self.motion_photo = None
        self.trajectory_photo = None

        self.create_widgets()
        self.reset_view()

    def create_widgets(self):
        """Construire une interface simple en trois blocs."""
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(
            header_frame,
            text="Estimation du mouvement d'un objet unique",
            font=("Arial", 19, "bold"),
        ).pack()
        tk.Label(
            header_frame,
            text="Segmentation + Lucas-Kanade Optical Flow",
            font=("Arial", 12),
        ).pack(pady=(2, 4))
        tk.Label(
            header_frame,
            text="Cette interface presente les trois resultats demandes : champ de mouvement, trajectoire globale, vitesse et direction.",
            font=("Arial", 10),
        ).pack()
        tk.Label(
            header_frame,
            text="Groundtruth utilise uniquement pour comparaison.",
            font=("Arial", 10, "italic"),
        ).pack(pady=(2, 0))

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=12, pady=6)

        tk.Label(controls_frame, text="Frame de depart").grid(row=0, column=0, padx=5)
        self.start_entry = tk.Entry(controls_frame, width=8)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=0, column=1, padx=5)

        tk.Label(controls_frame, text="Frame de fin").grid(row=0, column=2, padx=5)
        self.end_entry = tk.Entry(controls_frame, width=8)
        self.end_entry.insert(0, "100")
        self.end_entry.grid(row=0, column=3, padx=5)

        tk.Button(controls_frame, text="Lancer l'analyse", command=self.start_analysis).grid(row=0, column=4, padx=5)
        tk.Button(controls_frame, text="Afficher video tracking", command=self.show_tracking_video).grid(row=0, column=5, padx=5)
        tk.Button(controls_frame, text="Afficher trajectoire globale", command=self.show_trajectory_graph).grid(row=0, column=6, padx=5)
        tk.Button(controls_frame, text="Afficher graphes vitesse/direction", command=self.show_speed_direction_graphs).grid(row=0, column=7, padx=5)
        tk.Button(controls_frame, text="Reinitialiser", command=self.reset_view).grid(row=0, column=8, padx=5)
        tk.Button(controls_frame, text="Quitter", command=self.root.destroy).grid(row=0, column=9, padx=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        views_frame = tk.Frame(main_frame)
        views_frame.pack(fill=tk.X)

        motion_frame = tk.LabelFrame(views_frame, text="1. Champ de mouvement", font=("Arial", 12, "bold"), padx=8, pady=8)
        motion_frame.grid(row=0, column=0, sticky="n", padx=(0, 8))
        tk.Label(motion_frame, text="Vecteurs estimes entre points suivis.", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        self.motion_image_label = tk.Label(motion_frame, text="Lancer l'analyse pour afficher le champ de mouvement.", bg="black", fg="white")
        self.motion_image_label.pack()

        trajectory_frame = tk.LabelFrame(views_frame, text="2. Trajectoire globale de l'objet", font=("Arial", 12, "bold"), padx=8, pady=8)
        trajectory_frame.grid(row=0, column=1, sticky="n", padx=(8, 0))
        tk.Label(trajectory_frame, text="Centre de l'objet calcule frame par frame.", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        self.trajectory_image_label = tk.Label(trajectory_frame, text="Lancer l'analyse pour afficher la trajectoire.", bg="white", fg="black")
        self.trajectory_image_label.pack()

        views_frame.columnconfigure(0, weight=1)
        views_frame.columnconfigure(1, weight=1)

        stats_frame = tk.LabelFrame(main_frame, text="3. Analyse vitesse et direction", font=("Arial", 12, "bold"), padx=8, pady=8)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tk.Label(stats_frame, text="Calculees a partir des deplacements dx et dy.", font=("Arial", 10)).pack(anchor="w")
        self.stats_text = tk.Text(stats_frame, height=13, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _make_photo(self, image_bgr):
        """Convertir une image OpenCV BGR en PhotoImage Tkinter."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        original_width, original_height = pil_image.size

        if original_width > 0:
            scale = DISPLAY_WIDTH / original_width
            display_height = max(1, int(original_height * scale))
            pil_image = pil_image.resize((DISPLAY_WIDTH, display_height), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(pil_image)

    def _set_view_images(self, motion_image, trajectory_image):
        """Mettre a jour les deux images principales."""
        self.motion_photo = self._make_photo(motion_image)
        self.trajectory_photo = self._make_photo(trajectory_image)
        self.motion_image_label.configure(image=self.motion_photo, text="")
        self.trajectory_image_label.configure(image=self.trajectory_photo, text="")

    def start_analysis(self):
        """Lancer le pipeline existant puis demarrer l'animation."""
        try:
            start_frame = int(self.start_entry.get())
            end_frame = int(self.end_entry.get())

            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

            self.trajectory_df, self.analysis_df, self.summary = run_tracking(
                start_frame=start_frame,
                end_frame=end_frame,
            )
            self.image_files = get_image_files()
            self.current_index = 0
            self.animate_analysis()
        except ValueError as error:
            messagebox.showerror("Erreur", str(error))
        except Exception as error:
            messagebox.showerror("Erreur", f"L'analyse n'a pas pu etre lancee.\n{error}")

    def animate_analysis(self):
        """Mettre a jour les trois blocs de l'interface."""
        if self.trajectory_df is None or self.analysis_df is None:
            return

        if self.current_index >= len(self.trajectory_df):
            self.after_id = None
            return

        row = self.trajectory_df.iloc[self.current_index]
        frame_index = int(row["frame"])

        if frame_index < 0 or frame_index >= len(self.image_files):
            self.current_index += 1
            self.after_id = self.root.after(self.delay, self.animate_analysis)
            return

        current_frame = cv2.imread(str(self.image_files[frame_index]))

        if current_frame is None:
            self.current_index += 1
            self.after_id = self.root.after(self.delay, self.animate_analysis)
            return

        current_center = (float(row["x"]), float(row["y"]))
        old_points = _empty_points()
        new_points = _empty_points()

        if self.current_index > 0:
            previous_row = self.trajectory_df.iloc[self.current_index - 1]
            previous_frame_index = int(previous_row["frame"])

            if 0 <= previous_frame_index < len(self.image_files):
                previous_frame = cv2.imread(str(self.image_files[previous_frame_index]))
                previous_center = (float(previous_row["x"]), float(previous_row["y"]))
                old_points, new_points = compute_motion_field_for_display(
                    previous_frame,
                    current_frame,
                    previous_center=previous_center,
                )

        motion_view = draw_motion_field_view(
            current_frame,
            old_points,
            new_points,
            current_center=current_center,
            max_vectors=MAX_MOTION_VECTORS,
        )
        trajectory_view = draw_trajectory_global_view(
            self.trajectory_df[["x", "y"]].values,
            self.current_index,
            width=TRAJECTORY_VIEW_WIDTH,
            height=TRAJECTORY_VIEW_HEIGHT,
        )

        self._set_view_images(motion_view, trajectory_view)
        self.update_stats(self.current_index)

        self.current_index += 1
        self.after_id = self.root.after(self.delay, self.animate_analysis)

    def update_stats(self, index):
        """Afficher les mesures de vitesse et direction."""
        row = self.analysis_df.iloc[index]
        start_row = self.trajectory_df.iloc[0]
        end_row = self.trajectory_df.iloc[-1]

        frame = int(row["frame"])
        x = float(row["x"])
        y = float(row["y"])
        dx = float(row["dx"])
        dy = float(row["dy"])
        distance = float(row["distance"])
        speed = float(row["speed_px_per_frame"])
        mean_speed = float(self.analysis_df["speed_px_per_frame"].mean())
        max_speed = float(self.analysis_df["speed_px_per_frame"].max())
        direction_deg = float(row["direction_deg"])
        total_distance = float(self.analysis_df["distance"].sum())
        total_dx = float(end_row["x"] - start_row["x"])
        total_dy = float(end_row["y"] - start_row["y"])
        global_direction_deg = float(np.degrees(np.arctan2(total_dy, total_dx)))

        lines = [
            "Analyse quantitative du mouvement",
            "Groundtruth utilise uniquement pour comparaison.",
            "Repere image : x augmente vers la droite, y augmente vers le bas.",
            "",
            f"Frame actuelle : {frame}",
            f"Position actuelle : x = {x:.2f}, y = {y:.2f}",
            f"dx actuel : {dx:.2f} pixels",
            f"dy actuel : {dy:.2f} pixels",
            f"Distance actuelle : {distance:.2f} pixels",
            f"Vitesse actuelle : {speed:.2f} pixels/frame",
            f"Vitesse moyenne : {mean_speed:.2f} pixels/frame",
            f"Vitesse maximale : {max_speed:.2f} pixels/frame",
            f"Direction actuelle : {direction_deg:.2f} degres",
            f"Interpretation actuelle : {interpret_direction(dx, dy)}",
            "",
            f"Distance totale parcourue : {total_distance:.2f} pixels",
            f"Direction globale : {global_direction_deg:.2f} degres ({interpret_direction(total_dx, total_dy)})",
        ]

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "\n".join(lines))

    def _current_display_index(self):
        """Retourner l'index actuellement affiche, ou le dernier index valide."""
        if self.trajectory_df is None:
            return 0

        return min(max(self.current_index - 1, 0), len(self.trajectory_df) - 1)

    def check_analysis_ready(self):
        """Verifier que l'analyse a ete lancee."""
        if self.trajectory_df is None or self.analysis_df is None:
            messagebox.showinfo("Information", "Veuillez d'abord lancer l'analyse.")
            return False

        return True

    def show_tracking_video(self):
        """Afficher une frame de tracking avec trajectoire recente."""
        if not self.check_analysis_ready():
            return

        display_index = self._current_display_index()
        frame_index = int(self.trajectory_df.iloc[display_index]["frame"])

        if frame_index < 0 or frame_index >= len(self.image_files):
            messagebox.showerror("Erreur", "Frame introuvable.")
            return

        frame = cv2.imread(str(self.image_files[frame_index]))

        if frame is None:
            messagebox.showerror("Erreur", "La frame n'a pas pu etre chargee.")
            return

        tracking_view = draw_tracking_video_view(
            frame,
            self.trajectory_df[["x", "y"]].values,
            display_index,
        )

        plt.figure(figsize=(8, 5))
        plt.imshow(cv2.cvtColor(tracking_view, cv2.COLOR_BGR2RGB))
        plt.title("Video tracking")
        plt.axis("off")
        plt.show()

    def show_trajectory_graph(self):
        """Afficher la trajectoire globale complete dans Matplotlib."""
        if not self.check_analysis_ready():
            return

        display_index = self._current_display_index()
        current_row = self.trajectory_df.iloc[display_index]
        start_row = self.trajectory_df.iloc[0]
        end_row = self.trajectory_df.iloc[-1]

        plt.figure(figsize=(7, 5))
        plt.plot(self.trajectory_df["x"], self.trajectory_df["y"], color="red", marker="o", label="Trajectoire estimee")
        plt.scatter([start_row["x"]], [start_row["y"]], color="green", s=90, label="Start", zorder=3)
        plt.scatter([current_row["x"]], [current_row["y"]], color="gold", edgecolors="black", s=90, label="Current", zorder=4)
        plt.scatter([end_row["x"]], [end_row["y"]], color="blue", s=90, label="End", zorder=3)
        plt.gca().invert_yaxis()
        plt.title("Trajectoire globale de l'objet")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_speed_direction_graphs(self):
        """Afficher les graphes de vitesse et direction."""
        if not self.check_analysis_ready():
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

        axes[0].plot(self.analysis_df["frame"], self.analysis_df["speed_px_per_frame"], color="tab:blue")
        axes[0].set_title("Vitesse apparente de l'objet")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("pixels/frame")
        axes[0].grid(True)
        axes[0].text(
            0.5,
            -0.22,
            "La vitesse est exprimee en pixels/frame car la camera n'est pas calibree.",
            transform=axes[0].transAxes,
            ha="center",
            fontsize=9,
        )

        axes[1].plot(self.analysis_df["frame"], self.analysis_df["direction_deg"], color="tab:green")
        axes[1].set_title("Direction du mouvement")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("degres")
        axes[1].grid(True)

        fig.tight_layout()
        plt.show()

    def reset_view(self):
        """Reinitialiser l'interface."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.trajectory_df = None
        self.analysis_df = None
        self.summary = None
        self.current_index = 0
        self.motion_photo = None
        self.trajectory_photo = None
        self.motion_image_label.configure(image="", text="Lancer l'analyse pour afficher le champ de mouvement.")
        self.trajectory_image_label.configure(image="", text="Lancer l'analyse pour afficher la trajectoire.")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(
            tk.END,
            "Cette interface presente les trois resultats demandes :\n"
            "1. Champ de mouvement : vecteurs Lucas-Kanade sparse.\n"
            "2. Trajectoire globale : centre de l'objet frame par frame.\n"
            "3. Vitesse et direction : calculees a partir de dx et dy.\n\n"
            "Groundtruth utilise uniquement pour comparaison.",
        )


def main():
    """Lancer l'application Tkinter."""
    root = tk.Tk()
    TrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
