"""Interface Tkinter pour visualiser le tracking de voiture."""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline import run_tracking  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "car" / "car-11"
IMG_PATH = DATASET_PATH / "img"


def get_image_files():
    """Récupérer les images du dataset."""
    image_extensions = [".jpg", ".jpeg", ".png"]

    if not IMG_PATH.exists():
        return []

    return sorted([
        file for file in IMG_PATH.iterdir()
        if file.suffix.lower() in image_extensions
    ])


def interpret_direction(dx, dy, threshold=0.5):
    """Retourner une interprétation simple de la direction."""
    if abs(dx) < threshold and abs(dy) < threshold:
        return "mouvement très faible"

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

    return "mouvement très faible"


class TrackingApp:
    """Interface simple pour lancer et observer le tracking."""

    def __init__(self, root):
        self.root = root
        self.root.title("Estimation du mouvement - Tracking de voiture")
        self.root.minsize(1180, 820)

        self.trajectory_df = None
        self.analysis_df = None
        self.summary = None
        self.image_files = get_image_files()
        self.current_index = 0
        self.current_photo = None
        self.after_id = None
        self.is_paused = False
        self.delay = 50

        self.create_widgets()

    def create_widgets(self):
        """Créer les éléments de l'interface."""
        title_label = tk.Label(
            self.root,
            text="Tracking de voiture avec Lucas-Kanade",
            font=("Arial", 18, "bold"),
        )
        title_label.pack(pady=8)

        inputs_frame = tk.Frame(self.root)
        inputs_frame.pack(pady=4)

        tk.Label(inputs_frame, text="Frame de départ").grid(row=0, column=0, padx=5)
        self.start_entry = tk.Entry(inputs_frame, width=10)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=0, column=1, padx=5)

        tk.Label(inputs_frame, text="Frame de fin").grid(row=0, column=2, padx=5)
        self.end_entry = tk.Entry(inputs_frame, width=10)
        self.end_entry.insert(0, "100")
        self.end_entry.grid(row=0, column=3, padx=5)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=6)

        tk.Button(buttons_frame, text="Lancer le tracking", command=self.start_tracking).grid(row=0, column=0, padx=4)
        tk.Button(buttons_frame, text="Afficher graphe vitesse", command=self.show_speed_graph).grid(row=0, column=1, padx=4)
        tk.Button(buttons_frame, text="Afficher graphe direction", command=self.show_direction_graph).grid(row=0, column=2, padx=4)
        tk.Button(buttons_frame, text="Afficher trajectoire 2D", command=self.show_trajectory_graph).grid(row=0, column=3, padx=4)

        self.pause_button = tk.Button(buttons_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=4, padx=4)

        tk.Button(buttons_frame, text="Réinitialiser", command=self.reset_view).grid(row=0, column=5, padx=4)
        tk.Button(buttons_frame, text="Quitter", command=self.root.destroy).grid(row=0, column=6, padx=4)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.image_label = tk.Label(
            main_frame,
            text="La vidéo du tracking apparaîtra ici.",
            width=100,
            height=30,
            bg="black",
            fg="white",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=8)

        self.stats_text = tk.Text(main_frame, width=48, height=30)
        self.stats_text.grid(row=0, column=1, sticky="nsew", padx=8)

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)

        graphs_frame = tk.Frame(self.root)
        graphs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.speed_fig = Figure(figsize=(5.8, 2.6), dpi=100)
        self.speed_ax = self.speed_fig.add_subplot(111)
        self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, master=graphs_frame)
        self.speed_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6)

        self.direction_fig = Figure(figsize=(5.8, 2.6), dpi=100)
        self.direction_ax = self.direction_fig.add_subplot(111)
        self.direction_canvas = FigureCanvasTkAgg(self.direction_fig, master=graphs_frame)
        self.direction_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=6)

        graphs_frame.columnconfigure(0, weight=1)
        graphs_frame.columnconfigure(1, weight=1)

        self.update_embedded_graphs(0)

    def start_tracking(self):
        """Lancer le tracking et démarrer l'animation."""
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
            self.is_paused = False
            self.pause_button.configure(text="Pause")
            self.animate_tracking()
        except ValueError as error:
            messagebox.showerror("Erreur", str(error))
        except Exception as error:
            messagebox.showerror("Erreur", f"Le tracking n'a pas pu être lancé.\n{error}")

    def animate_tracking(self):
        """Afficher la frame courante et mettre à jour l'interface."""
        if self.is_paused:
            return

        if self.trajectory_df is None or self.analysis_df is None:
            return

        if self.current_index >= len(self.trajectory_df):
            self.after_id = None
            return

        row = self.trajectory_df.iloc[self.current_index]
        frame_index = int(row["frame"])

        if frame_index < 0 or frame_index >= len(self.image_files):
            self.current_index += 1
            self.after_id = self.root.after(self.delay, self.animate_tracking)
            return

        image = cv2.imread(str(self.image_files[frame_index]))

        if image is None:
            self.current_index += 1
            self.after_id = self.root.after(self.delay, self.animate_tracking)
            return

        trajectory_points = self.trajectory_df.iloc[:self.current_index + 1][["x", "y"]].values
        center = (row["x"], row["y"])

        image = self.draw_trajectory(image, trajectory_points)
        image = self.draw_current_center(image, center)
        cv2.putText(
            image,
            f"Frame : {frame_index}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.show_image(image)
        self.update_current_stats(self.current_index)
        self.update_embedded_graphs(self.current_index)

        self.current_index += 1
        self.after_id = self.root.after(self.delay, self.animate_tracking)

    def draw_trajectory(self, image, trajectory):
        """Dessiner la trajectoire progressive en rouge."""
        output = image.copy()
        points = np.asarray(trajectory).reshape(-1, 2).astype(int)

        for index, point in enumerate(points):
            x, y = point
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

            if index > 0:
                prev_x, prev_y = points[index - 1]
                cv2.line(output, (prev_x, prev_y), (x, y), (0, 0, 255), 2)

        return output

    def draw_current_center(self, image, center):
        """Dessiner le centre courant de l'objet."""
        output = image.copy()
        x, y = [int(value) for value in center]
        cv2.circle(output, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(output, (x, y), 11, (0, 0, 0), 2)
        return output

    def show_image(self, image_bgr):
        """Afficher une image OpenCV dans Tkinter."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((760, 500), Image.Resampling.LANCZOS)

        self.current_photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.current_photo, text="")

    def update_current_stats(self, index):
        """Mettre à jour les statistiques en temps réel."""
        row = self.analysis_df.iloc[index]
        partial_df = self.analysis_df.iloc[:index + 1]

        frame = int(row["frame"])
        total_frames = len(self.analysis_df)
        x = float(row["x"])
        y = float(row["y"])
        dx = float(row["dx"])
        dy = float(row["dy"])
        speed = float(row["speed_px_per_frame"])
        direction_deg = float(row["direction_deg"])
        distance_so_far = float(partial_df["distance"].sum())
        mean_speed_so_far = float(partial_df["speed_px_per_frame"].mean())
        total_dx_so_far = float(partial_df["dx"].sum())
        total_dy_so_far = float(partial_df["dy"].sum())

        current_text = interpret_direction(dx, dy)
        global_text = interpret_direction(total_dx_so_far, total_dy_so_far)

        lines = [
            "Statistiques du mouvement",
            "",
            f"Frame actuelle : {frame} / {int(self.analysis_df['frame'].iloc[-1])}",
            f"Nombre de frames analysées : {total_frames}",
            f"Position actuelle : x = {x:.2f}, y = {y:.2f}",
            f"dx actuel : {dx:.2f} pixels",
            f"dy actuel : {dy:.2f} pixels",
            f"Vitesse actuelle : {speed:.2f} pixels/frame",
            f"Direction actuelle : {direction_deg:.2f} degrés",
            f"Interprétation actuelle : {current_text}",
            "",
            f"Distance parcourue : {distance_so_far:.2f} pixels",
            f"Vitesse moyenne : {mean_speed_so_far:.2f} pixels/frame",
            f"Direction globale : {global_text}",
        ]

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "\n".join(lines))

    def update_embedded_graphs(self, index):
        """Mettre à jour les graphes intégrés."""
        self.speed_ax.clear()
        self.direction_ax.clear()

        if self.analysis_df is not None and len(self.analysis_df) > 0:
            partial_df = self.analysis_df.iloc[:index + 1]

            self.speed_ax.plot(
                partial_df["frame"],
                partial_df["speed_px_per_frame"],
                color="tab:blue",
            )
            self.speed_ax.set_title("Vitesse en temps réel")
            self.speed_ax.set_xlabel("Frame")
            self.speed_ax.set_ylabel("pixels/frame")
            self.speed_ax.grid(True)

            self.direction_ax.plot(
                partial_df["frame"],
                partial_df["direction_deg"],
                color="tab:green",
            )
            self.direction_ax.set_title("Direction en temps réel")
            self.direction_ax.set_xlabel("Frame")
            self.direction_ax.set_ylabel("degrés")
            self.direction_ax.grid(True)
        else:
            self.speed_ax.set_title("Vitesse en temps réel")
            self.speed_ax.set_xlabel("Frame")
            self.speed_ax.set_ylabel("pixels/frame")
            self.speed_ax.grid(True)

            self.direction_ax.set_title("Direction en temps réel")
            self.direction_ax.set_xlabel("Frame")
            self.direction_ax.set_ylabel("degrés")
            self.direction_ax.grid(True)

        self.speed_fig.tight_layout()
        self.direction_fig.tight_layout()
        self.speed_canvas.draw()
        self.direction_canvas.draw()

    def toggle_pause(self):
        """Mettre l'animation en pause ou la reprendre."""
        if self.trajectory_df is None:
            messagebox.showinfo("Information", "Veuillez d'abord lancer le tracking.")
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_button.configure(text="Reprendre")
        else:
            self.pause_button.configure(text="Pause")
            self.animate_tracking()

    def reset_view(self):
        """Réinitialiser la visualisation."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.current_index = 0
        self.is_paused = False
        self.pause_button.configure(text="Pause")
        self.image_label.configure(image="", text="La vidéo du tracking apparaîtra ici.")
        self.stats_text.delete("1.0", tk.END)
        self.update_embedded_graphs(0)

    def check_tracking_ready(self):
        """Vérifier que le tracking a été lancé."""
        if self.trajectory_df is None or self.analysis_df is None:
            messagebox.showinfo("Information", "Veuillez d'abord lancer le tracking.")
            return False

        return True

    def show_speed_graph(self):
        """Afficher le graphe de vitesse dans une fenêtre Matplotlib."""
        if not self.check_tracking_ready():
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.analysis_df["frame"], self.analysis_df["speed_px_per_frame"])
        plt.title("Vitesse de l'objet au cours du temps")
        plt.xlabel("Frame")
        plt.ylabel("Vitesse (pixels/frame)")
        plt.grid(True)
        plt.show()

    def show_direction_graph(self):
        """Afficher le graphe de direction dans une fenêtre Matplotlib."""
        if not self.check_tracking_ready():
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.analysis_df["frame"], self.analysis_df["direction_deg"])
        plt.title("Direction du mouvement au cours du temps")
        plt.xlabel("Frame")
        plt.ylabel("Direction (degrés)")
        plt.grid(True)
        plt.show()

    def show_trajectory_graph(self):
        """Afficher la trajectoire 2D dans une fenêtre Matplotlib."""
        if not self.check_tracking_ready():
            return

        plt.figure(figsize=(7, 5))
        plt.plot(self.trajectory_df["x"], self.trajectory_df["y"], marker="o")
        plt.gca().invert_yaxis()
        plt.title("Trajectoire 2D de l'objet")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()


def main():
    """Lancer l'application Tkinter."""
    root = tk.Tk()
    TrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
