# Estimation du mouvement d'un objet unique - Voiture

Projet de traitement d'images et vision par ordinateur portant sur l'estimation du mouvement apparent d'une voiture dans une sequence d'images.

## Objectif

Le travail demande consiste a:

- estimer le champ de mouvement;
- extraire la trajectoire globale de l'objet;
- analyser la vitesse et la direction du mouvement.

Le workflow final est autonome: il utilise uniquement les images, une ROI initiale manuelle ou une bbox fournie par l'utilisateur, puis des methodes classiques de traitement d'images. Aucune annotation n'est utilisee dans le pipeline final.

## Dataset

Les images sont chargees depuis:

```text
data/car/car-11/img/
```

L'ordre temporel des noms de fichiers est conserve, car le mouvement est estime entre frames successives.

## Methodes utilisees

- conversion en niveaux de gris pour travailler sur la luminance;
- flou gaussien pour reduire le bruit;
- amelioration du contraste par CLAHE, stretching ou egalisation;
- segmentation dans une ROI par Otsu ou seuillage adaptatif;
- morphologie: ouverture, fermeture, plus grande composante connectee;
- Canny pour verifier visuellement les contours de la voiture;
- difference d'images pour visualiser les pixels mobiles;
- Lucas-Kanade pour estimer le champ de mouvement;
- centres successifs de la bbox estimee pour extraire la trajectoire;
- vitesse en pixels/frame et direction par `atan2(dy, dx)`.

## Pipeline

```text
frames
-> preprocessing
-> histogram/contrast enhancement
-> manual ROI
-> segmentation in ROI
-> morphology
-> Canny and frame difference
-> Lucas-Kanade motion vectors
-> global trajectory
-> speed and direction analysis
-> visualizations and Tkinter interface
```

Chaque etape genere des images ou fichiers dans `results/` et une interpretation textuelle dans `results/interpretation_results.txt`.

## Notebooks

```text
notebooks/
├── 01_dataset_exploration.ipynb
├── 02_preprocessing_histogram.ipynb
├── 03_segmentation_morphology.ipynb
├── 04_edge_motion_detection.ipynb
├── 05_optical_flow_lucas_kanade.ipynb
├── 06_trajectory_extraction.ipynb
├── 07_speed_direction_analysis.ipynb
└── 08_results_visualization.ipynb
```

Chaque notebook contient un objectif, du code, les sorties sauvegardees et une interpretation.

## Interface Tkinter

L'interface permet de:

- choisir le dossier d'images;
- entrer `start_frame`, `end_frame` et `initial_bbox`;
- selectionner une ROI manuellement;
- choisir la methode de pretraitement;
- choisir Otsu ou le seuillage adaptatif;
- activer ou non l'inversion du masque et la segmentation;
- lancer le suivi;
- afficher le champ de mouvement, la trajectoire, la vitesse, la direction, Canny, segmentation et motion mask;
- lire le resume automatique des resultats.

## Installation

```bash
pip install -r requirements.txt
```

## Lancer les notebooks

```bash
jupyter notebook
```

Ouvrir les notebooks dans l'ordre depuis le dossier `notebooks/`.

## Lancer le pipeline

Exemple avec une bbox manuelle:

```bash
python -c "from src.pipeline import run_tracking_without_groundtruth; run_tracking_without_groundtruth(initial_bbox=(535,300,220,105), use_segmentation=True)"
```

La bbox doit etre adaptee si la voiture n'est pas bien encadree sur la premiere frame.

## Lancer l'interface

```bash
python app_tkinter.py
```

L'ancien point d'entree reste disponible:

```bash
python app/tkinter_app.py
```

## Resultats generes

```text
results/
├── trajectory.csv
├── interpretation_results.txt
├── preprocessing/
├── segmentation/
├── morphology/
├── edge_detection/
├── motion_detection/
├── optical_flow/
├── trajectory/
├── graphs/
└── final_visualization/
```

Sorties principales:

- comparaison pretraitement/histogramme;
- masque Otsu;
- masque adaptatif;
- masque apres morphologie;
- contours Canny;
- difference d'images;
- motion mask;
- champ de mouvement Lucas-Kanade avec vecteurs;
- trajectoire globale dessinee sur image;
- graphe de vitesse;
- graphe de direction;
- resume texte interpretable.

## Limites

Les resultats dependent du choix de la ROI initiale, de l'eclairage, du contraste, des ombres, de la qualite de segmentation, du nombre de points suivis et de l'hypothese de petits deplacements de Lucas-Kanade. Les vitesses sont exprimees en pixels/frame si aucun FPS reel n'est fourni.

## Contraintes respectees

Le workflow final ne lit pas les annotations, ne compare pas avec des boites annotees et n'utilise pas de deep learning. Les methodes restent classiques: pretraitement, histogramme, segmentation, morphologie, Canny, difference d'images, Lucas-Kanade, trajectoire, vitesse et direction.
