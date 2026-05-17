# Projet 1 - Estimation du mouvement d'un objet unique

Ce projet traite l'estimation du mouvement d'une voiture dans une sequence d'images. Il est concu pour rester simple, clair et facilement presentable.

## Objectif

Le but est de repondre aux trois points demandes :

- estimer le champ de mouvement ;
- extraire la trajectoire globale de l'objet ;
- analyser la vitesse et la direction du mouvement.

La voiture est consideree comme un objet rigide. Nous travaillons avec des deplacements entre frames successives, et la vitesse est exprimee en pixels/frame.

## Dataset

Le dataset utilise est deja present dans :

```text
data/car/car-11/
├── img/
├── groundtruth.txt
├── full_occlusion.txt
└── out_of_view.txt
```

Le fichier `groundtruth.txt` n'est pas utilise pour suivre l'objet. Il est utilise uniquement pour explorer le dataset, verifier visuellement les annotations et comparer la trajectoire estimee a la fin.

## Methode utilisee

La methode principale est :

1. Pretraitement des images : niveaux de gris, CLAHE et GaussianBlur.
2. Segmentation dans une ROI manuelle : Otsu ou Adaptive Threshold, morphologie, puis conservation de la plus grande composante.
3. Detection de points caracteristiques avec `cv2.goodFeaturesToTrack`.
4. Estimation du champ de mouvement sparse avec Lucas-Kanade pyramidal.
5. Calcul du centre robuste de l'objet avec la mediane des points suivis.
6. Analyse de `dx`, `dy`, distance, vitesse en pixels/frame et direction avec `atan2`.

Hypotheses etudiees :

- rigidite : la voiture est consideree comme un objet rigide, donc les points suivis ont un mouvement global coherent ;
- petits deplacements : Lucas-Kanade fonctionne mieux lorsque le deplacement entre deux frames est faible ;
- illumination constante : Lucas-Kanade suppose que l'intensite d'un meme point reste presque constante entre deux frames.

## Structure

```text
motion_estimation_project/
├── data/
│   └── car/
│       └── car-11/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_segmentation_detection.ipynb
│   ├── 04_lucas_kanade_motion_field.ipynb
│   ├── 05_trajectory_extraction.ipynb
│   ├── 06_speed_direction_analysis.ipynb
│   └── 07_final_visualization.ipynb
├── src/
├── app/
├── results/
│   ├── plots/
│   ├── frames_output/
│   └── videos/
├── presentation/
├── README.md
├── requirements.txt
└── .gitignore
```

## Role des notebooks

| Notebook | Role |
|---|---|
| `01_dataset_exploration.ipynb` | Explorer les images, les annotations et les fichiers d'occlusion. |
| `02_preprocessing.ipynb` | Montrer le passage en gris, CLAHE et GaussianBlur. |
| `03_segmentation_detection.ipynb` | Segmenter la voiture dans une ROI manuelle et detecter les points. |
| `04_lucas_kanade_motion_field.ipynb` | Estimer le champ de mouvement sparse avec Lucas-Kanade. |
| `05_trajectory_extraction.ipynb` | Suivre les points, filtrer les aberrations et extraire la trajectoire. |
| `06_speed_direction_analysis.ipynb` | Calculer vitesse et direction a partir de la trajectoire. |
| `07_final_visualization.ipynb` | Rassembler les resultats finaux pour la presentation. |

## Installation

```bash
pip install -r requirements.txt
```

## Execution des notebooks

```bash
jupyter notebook
```

Ouvrir les notebooks dans l'ordre, de `01_dataset_exploration.ipynb` a `07_final_visualization.ipynb`.

## Lancement de l'interface Tkinter

```bash
python app/tkinter_app.py
```

L'interface est organisee selon les trois objectifs :

- champ de mouvement ;
- trajectoire globale ;
- vitesse et direction.

## Resultats generes

Les notebooks sauvegardent les sorties dans `results/` :

```text
results/trajectory_estimated.csv
results/motion_analysis_estimated.csv
results/plots/trajectory_2d.png
results/plots/speed_curve.png
results/plots/direction_curve.png
results/plots/comparison_groundtruth.png
results/frames_output/points_detected.png
results/frames_output/segmentation_comparison.png
results/frames_output/motion_field_frame.png
results/frames_output/trajectory_on_frame.png
```

## Limites

La solution reste volontairement pedagogique. Elle ne fait pas d'homographie, pas de stabilisation de camera, pas de deep learning et pas de tracker complexe. Les resultats peuvent etre sensibles a la ROI initiale, a l'illumination, aux occultations et aux grands deplacements.

## Conclusion

Nous avons utilise Lucas-Kanade pour estimer le deplacement des points caracteristiques de la voiture. Les vecteurs obtenus forment un champ de mouvement sparse.

Nous avons calcule le centre robuste de l'objet a chaque frame a partir des points suivis. La succession de ces centres represente la trajectoire globale de la voiture.

Nous avons calcule `dx`, `dy`, la distance, la vitesse apparente en pixels/frame et la direction avec `atan2`.
