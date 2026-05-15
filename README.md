# Estimation du mouvement d'un objet unique

Projet universitaire de traitement d'image et de vision par ordinateur.

## Sujet du projet

Ce projet porte sur l'estimation du mouvement d'un objet unique dans une séquence d'images. La séquence utilisée pour démarrer le travail se trouve dans le dossier `data/car/car-11`.

## Objectif

L'objectif est de construire progressivement une chaîne de traitement permettant de :

- charger et explorer une séquence d'images ;
- initialiser l'objet à suivre à partir des annotations ;
- estimer son mouvement entre les frames ;
- extraire sa trajectoire globale ;
- analyser sa vitesse et sa direction ;
- visualiser les résultats obtenus.

## Méthode choisie

La méthode principale du projet sera le flot optique Lucas-Kanade. Cette approche permet de suivre des points caractéristiques entre deux images consécutives et d'estimer leur déplacement.

Dans ce projet, nous utiliserons Lucas-Kanade pour suivre les points situés sur l'objet d'intérêt, puis nous regrouperons ces déplacements pour obtenir une trajectoire globale.

## Structure du projet

```text
motion-estimation-project/
├── data/
│   └── car/
│       └── car-11/
│           ├── img/
│           ├── groundtruth.txt
│           ├── full_occlusion.txt
│           └── out_of_view.txt
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_object_detection_initialization.ipynb
│   ├── 04_optical_flow_lucas_kanade.ipynb
│   ├── 05_trajectory_extraction.ipynb
│   ├── 06_speed_direction_analysis.ipynb
│   └── 07_results_visualization.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── detection.py
│   ├── optical_flow.py
│   ├── trajectory.py
│   ├── analysis.py
│   └── visualization.py
├── results/
│   ├── frames_output/
│   ├── plots/
│   └── videos/
├── report/
│   └── README_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Rôle des notebooks

| Notebook | Rôle |
|---|---|
| `01_dataset_exploration.ipynb` | Explorer le dataset, les images et les fichiers d'annotations. |
| `02_preprocessing.ipynb` | Préparer les images avant le suivi. |
| `03_object_detection_initialization.ipynb` | Initialiser l'objet à suivre avec la boîte englobante de départ. |
| `04_optical_flow_lucas_kanade.ipynb` | Préparer le calcul du flot optique Lucas-Kanade. |
| `05_trajectory_extraction.ipynb` | Construire la trajectoire globale de l'objet. |
| `06_speed_direction_analysis.ipynb` | Analyser la vitesse, les déplacements et la direction. |
| `07_results_visualization.ipynb` | Visualiser les boîtes, points, vecteurs, trajectoires et sorties finales. |

## Lancer le projet étape par étape

1. Créer un environnement virtuel :

```bash
python -m venv .venv
```

2. Activer l'environnement :

```bash
.venv\Scripts\activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

4. Lancer Jupyter :

```bash
jupyter notebook
```

5. Ouvrir les notebooks dans l'ordre, en commençant par :

```text
notebooks/01_dataset_exploration.ipynb
```

## État actuel

La structure du projet est prête. Les notebooks et les modules Python contiennent seulement des squelettes de départ. Les algorithmes seront ajoutés progressivement dans les prochaines étapes.
