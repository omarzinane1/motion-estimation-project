from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MotionEstimationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Estimation du mouvement d’un objet unique — Voiture")
        self.geometry("1100x780")
        self.configure(padx=12, pady=12)

        self.photos = {}

        title = ttk.Label(
            self,
            text="Estimation du mouvement d’un objet unique — Voiture",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.motion_tab = ttk.Frame(self.notebook, padding=10)
        self.trajectory_tab = ttk.Frame(self.notebook, padding=10)
        self.analysis_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.motion_tab, text="Champ de mouvement")
        self.notebook.add(self.trajectory_tab, text="Trajectoire globale")
        self.notebook.add(self.analysis_tab, text="Vitesse et direction")

        self._build_motion_tab()
        self._build_trajectory_tab()
        self._build_analysis_tab()

    def _image_label(self, parent):
        label = ttk.Label(parent, anchor="center")
        label.pack(fill="both", expand=True, pady=10)
        return label

    def _show_image(self, label, relative_path, max_size=(1000, 560)):
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            messagebox.showwarning(
                "Fichier absent",
                f"Le fichier n'existe pas encore:\n{path}\n\nExecutez les notebooks dans l'ordre.",
            )
            return

        image = Image.open(path)
        image.thumbnail(max_size)
        photo = ImageTk.PhotoImage(image)
        self.photos[label] = photo
        label.configure(image=photo)

    def _build_motion_tab(self):
        ttk.Label(
            self.motion_tab,
            text=(
                "Le champ de mouvement est estime avec Lucas-Kanade sur des points "
                "caracteristiques. Il s'agit d'un champ sparse."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 8))

        image_label = ttk.Label(self.motion_tab, anchor="center")
        ttk.Button(
            self.motion_tab,
            text="Afficher champ de mouvement",
            command=lambda: self._show_image(image_label, "results/frames_output/motion_field_frame.png"),
        ).pack(anchor="w")
        image_label.pack(fill="both", expand=True, pady=10)

    def _build_trajectory_tab(self):
        ttk.Label(
            self.trajectory_tab,
            text=(
                "Legende : Start en vert ; Current en jaune ; End en bleu ; "
                "trajectoire en rouge."
            ),
            wraplength=950,
        ).pack(anchor="w", pady=(0, 8))

        buttons = ttk.Frame(self.trajectory_tab)
        buttons.pack(anchor="w", pady=(0, 8))

        image_label = self._image_label(self.trajectory_tab)
        ttk.Button(
            buttons,
            text="Afficher trajectoire 2D",
            command=lambda: self._show_image(image_label, "results/plots/trajectory_2d.png"),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="Afficher trajectoire sur frame",
            command=lambda: self._show_image(image_label, "results/frames_output/trajectory_on_frame.png"),
        ).pack(side="left")

    def _build_analysis_tab(self):
        stats_frame = ttk.LabelFrame(self.analysis_tab, text="Mesures estimees", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))

        stats_text = self._analysis_text()
        ttk.Label(stats_frame, text=stats_text, justify="left", wraplength=950).pack(anchor="w")

        buttons = ttk.Frame(self.analysis_tab)
        buttons.pack(anchor="w", pady=(0, 8))

        image_label = self._image_label(self.analysis_tab)
        ttk.Button(
            buttons,
            text="Afficher graphe vitesse",
            command=lambda: self._show_image(image_label, "results/plots/speed_curve.png"),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="Afficher graphe direction",
            command=lambda: self._show_image(image_label, "results/plots/direction_curve.png"),
        ).pack(side="left")

    def _analysis_text(self):
        csv_path = PROJECT_ROOT / "results/motion_analysis_estimated.csv"
        if not csv_path.exists():
            return "Le fichier results/motion_analysis_estimated.csv est absent. Executez le notebook 06."

        df = pd.read_csv(csv_path)
        if df.empty:
            return "Le fichier d'analyse est vide."

        current = df.iloc[-1]
        mean_speed = df["speed_px_frame"].mean()
        total_distance = df["distance"].sum()

        return (
            f"dx actuel : {current['dx']:.2f} px\n"
            f"dy actuel : {current['dy']:.2f} px\n"
            f"vitesse actuelle : {current['speed_px_frame']:.2f} pixels/frame\n"
            f"vitesse moyenne : {mean_speed:.2f} pixels/frame\n"
            f"direction actuelle : {current['direction_deg']:.2f} degres\n"
            f"interpretation : {current['interpretation']}\n"
            f"distance totale : {total_distance:.2f} pixels"
        )


def main():
    app = MotionEstimationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
