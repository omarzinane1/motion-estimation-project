"""Tkinter interface for the no-groundtruth car motion-estimation pipeline."""

import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.detection import select_initial_roi, validate_bbox
from src.pipeline import IMG_PATH, PROJECT_ROOT, RESULTS_PATH, run_tracking_without_groundtruth


class MotionEstimationApp:
    """Simple UI to run and inspect the classical tracking pipeline."""

    def __init__(self, root):
        self.root = root
        self.root.title("Estimation du mouvement d'un objet unique - Voiture")
        self.root.minsize(1180, 760)

        self.image_dir = tk.StringVar(value=str(IMG_PATH))
        self.preprocess_method = tk.StringVar(value="stretching")
        self.segmentation_method = tk.StringVar(value="otsu")
        self.invert_mask = tk.BooleanVar(value=True)
        self.use_segmentation = tk.BooleanVar(value=True)
        self.current_photo = None
        self.last_result = None
        self.worker_thread = None

        self._build_ui()
        self._show_original_frame()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)

        title = ttk.Label(
            main,
            text="Estimation du mouvement d'un objet unique - Voiture",
            font=("Segoe UI", 16, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        controls = ttk.Frame(main)
        controls.grid(row=1, column=0, sticky="nsw", padx=(0, 10))

        dataset_box = ttk.LabelFrame(controls, text="1. Selection dataset", padding=8)
        dataset_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(dataset_box, text="Choisir dossier images", command=self.choose_image_dir).pack(fill=tk.X)
        ttk.Label(dataset_box, textvariable=self.image_dir, wraplength=330).pack(fill=tk.X, pady=(6, 0))

        params_box = ttk.LabelFrame(controls, text="2. Parametres", padding=8)
        params_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(params_box, text="start_frame").grid(row=0, column=0, sticky="w", pady=2)
        self.start_entry = ttk.Entry(params_box, width=12)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(params_box, text="end_frame").grid(row=1, column=0, sticky="w", pady=2)
        self.end_entry = ttk.Entry(params_box, width=12)
        self.end_entry.insert(0, "100")
        self.end_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(params_box, text="initial_bbox").grid(row=2, column=0, sticky="w", pady=2)
        self.bbox_entry = ttk.Entry(params_box, width=22)
        self.bbox_entry.insert(0, "535,300,220,105")
        self.bbox_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Button(params_box, text="Selectionner ROI manuellement", command=self.select_roi).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 2)
        )

        ttk.Label(params_box, text="preprocess_method").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(
            params_box,
            textvariable=self.preprocess_method,
            values=["none", "clahe", "stretching", "equalization"],
            state="readonly",
            width=18,
        ).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(params_box, text="segmentation_method").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Combobox(
            params_box,
            textvariable=self.segmentation_method,
            values=["otsu", "adaptive"],
            state="readonly",
            width=18,
        ).grid(row=5, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(params_box, text="invert_mask", variable=self.invert_mask).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=2
        )
        ttk.Checkbutton(params_box, text="use_segmentation", variable=self.use_segmentation).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=2
        )
        params_box.columnconfigure(1, weight=1)

        actions_box = ttk.LabelFrame(controls, text="3. Actions", padding=8)
        actions_box.pack(fill=tk.X)
        ttk.Button(actions_box, text="Lancer le suivi", command=self.run_tracking).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher champ de mouvement", command=self.show_motion_field).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher trajectoire globale", command=self.show_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher vitesse", command=self.show_speed).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher direction", command=self.show_direction).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher segmentation", command=self.show_segmentation).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher Canny", command=self.show_edges).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Afficher motion mask", command=self.show_motion_mask).pack(fill=tk.X, pady=2)
        ttk.Button(actions_box, text="Ouvrir dossier resultats", command=self.open_results_folder).pack(fill=tk.X, pady=2)

        display_box = ttk.LabelFrame(main, text="4. Zone d'affichage", padding=8)
        display_box.grid(row=1, column=1, rowspan=2, sticky="nsew")
        display_box.rowconfigure(0, weight=1)
        display_box.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(display_box, text="Image/resultat", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = ttk.Label(display_box, text="Pret.")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        text_box = ttk.LabelFrame(main, text="5. Resume et interpretation", padding=8)
        text_box.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(8, 0))
        text_box.rowconfigure(0, weight=1)
        text_box.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(text_box, width=48, height=15, wrap=tk.WORD)
        self.summary_text.grid(row=0, column=0, sticky="nsew")

    def choose_image_dir(self):
        directory = filedialog.askdirectory(initialdir=str(PROJECT_ROOT))
        if directory:
            self.image_dir.set(directory)
            self._show_original_frame()

    def _image_files(self):
        image_dir = Path(self.image_dir.get())
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if not image_dir.exists():
            return []
        return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in extensions])

    def _parse_start_end(self):
        start = int(self.start_entry.get().strip() or "0")
        end_text = self.end_entry.get().strip()
        end = int(end_text) if end_text else None
        return start, end

    def _parse_bbox(self):
        text = self.bbox_entry.get().strip()
        if not text:
            return None
        parts = [part.strip() for part in text.replace(";", ",").split(",")]
        if len(parts) != 4:
            raise ValueError("Format bbox attendu: x,y,w,h")
        return tuple(int(float(part)) for part in parts)

    def select_roi(self):
        try:
            start, _ = self._parse_start_end()
            files = self._image_files()
            if not files:
                raise ValueError("Aucune image trouvee dans le dossier selectionne.")
            if start < 0 or start >= len(files):
                raise ValueError("start_frame invalide.")
            frame = cv2.imread(str(files[start]))
            if frame is None:
                raise ValueError("La frame de depart est illisible.")
            bbox = select_initial_roi(frame)
            bbox = validate_bbox(bbox, frame.shape)
            if bbox is None:
                messagebox.showinfo("ROI", "Aucune ROI valide n'a ete selectionnee.")
                return
            self.bbox_entry.delete(0, tk.END)
            self.bbox_entry.insert(0, ",".join(str(v) for v in bbox))
            self.show_cv_image(cv2.rectangle(frame.copy(), (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2))
        except Exception as error:
            messagebox.showerror("Erreur ROI", str(error))

    def run_tracking(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Suivi", "Le suivi est deja en cours.")
            return

        try:
            start, end = self._parse_start_end()
            bbox = self._parse_bbox()
            if bbox is None:
                raise ValueError("Veuillez fournir initial_bbox=(x,y,w,h) ou sélectionner une ROI manuellement.")
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return

        self.status_label.configure(text="Suivi en cours...")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "Execution du pipeline sans groundtruth...\n")

        def worker():
            try:
                result = run_tracking_without_groundtruth(
                    start_frame=start,
                    end_frame=end,
                    initial_bbox=bbox,
                    use_manual_roi=False,
                    preprocess_method=self.preprocess_method.get(),
                    segmentation_method=self.segmentation_method.get(),
                    invert_mask=self.invert_mask.get(),
                    use_segmentation=self.use_segmentation.get(),
                    save_outputs=True,
                    image_dir=self.image_dir.get(),
                )
                self.root.after(0, lambda: self._tracking_done(result))
            except Exception as error:
                self.root.after(0, lambda: self._tracking_failed(error))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _tracking_done(self, result):
        self.last_result = result
        self.status_label.configure(text="Suivi termine. Resultats sauvegardes dans results/.")
        self.load_interpretation()
        self.show_trajectory()
        messagebox.showinfo("Succes", "Le suivi est termine et les resultats ont ete sauvegardes.")

    def _tracking_failed(self, error):
        self.status_label.configure(text="Erreur pendant le suivi.")
        messagebox.showerror("Erreur", f"Le suivi n'a pas pu etre lance.\n{error}")

    def load_interpretation(self):
        path = RESULTS_PATH / "interpretation_results.txt"
        self.summary_text.delete("1.0", tk.END)
        if path.exists():
            self.summary_text.insert(tk.END, path.read_text(encoding="utf-8"))
        elif self.last_result:
            self.summary_text.insert(tk.END, str(self.last_result.get("summary", "")))
        else:
            self.summary_text.insert(tk.END, "Aucun resume disponible.")

    def _show_original_frame(self):
        files = self._image_files()
        if not files:
            return
        frame = cv2.imread(str(files[0]))
        if frame is not None:
            self.show_cv_image(frame)

    def show_cv_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        image.thumbnail((780, 560), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo, text="")

    def show_image_file(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showinfo("Resultat", f"Fichier introuvable:\n{path}")
            return
        image = Image.open(path)
        image.thumbnail((780, 560), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo, text="")
        self.status_label.configure(text=str(path))

    def _latest_file(self, directory, pattern="*.png"):
        directory = Path(directory)
        files = sorted(directory.glob(pattern))
        return files[-1] if files else None

    def show_motion_field(self):
        self.show_image_file(RESULTS_PATH / "final_visualization" / "final_motion_field_lk.png")

    def show_trajectory(self):
        self.show_image_file(RESULTS_PATH / "final_visualization" / "final_trajectory.png")

    def show_speed(self):
        self.show_image_file(RESULTS_PATH / "graphs" / "speed.png")

    def show_direction(self):
        self.show_image_file(RESULTS_PATH / "graphs" / "direction.png")

    def show_segmentation(self):
        path = self._latest_file(RESULTS_PATH / "segmentation", "mask_overlay_*.png")
        self.show_image_file(path or RESULTS_PATH / "final_visualization" / "segmentation_morphology_comparison.png")

    def show_edges(self):
        path = self._latest_file(RESULTS_PATH / "edge_detection", "edges_*.png")
        if path:
            self.show_image_file(path)
        else:
            messagebox.showinfo("Canny", "Aucun resultat Canny disponible.")

    def show_motion_mask(self):
        path = self._latest_file(RESULTS_PATH / "motion_detection", "motion_overlay_*.png")
        if path:
            self.show_image_file(path)
        else:
            messagebox.showinfo("Motion mask", "Aucun masque de mouvement disponible.")

    def open_results_folder(self):
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        os.startfile(str(RESULTS_PATH))


def main():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print("Tkinter ne peut pas demarrer dans cet environnement Python.")
        print(error)
        return
    MotionEstimationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
