# Motion Estimation Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Motion%20Estimation-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

## Project Overview

**Motion Estimation Project** is a computer vision project focused on estimating the motion of a single moving object in image sequences. The project analyzes car sequences to detect the object, compute optical flow, track its trajectory, and extract motion characteristics such as speed and direction.

The goal is to build a clean and modular AI pipeline that transforms raw image frames into interpretable motion analysis outputs, including annotated frames, trajectory plots, vector-field visualizations, graphs, and videos.

> This repository is structured as a professional computer vision project, with clear separation between data, reusable source modules, notebooks, results, and interface components.

## Objectives

- Estimate the motion of a single object across image sequences.
- Load and preprocess sequential image frames.
- Detect and isolate the target object.
- Compute optical flow between consecutive frames.
- Track object position over time.
- Extract object trajectory from frame-level observations.
- Analyze speed, direction, displacement, and movement consistency.
- Visualize motion using arrows, paths, plots, and generated videos.
- Provide an interface concept for interactive experimentation and demonstration.

## Motion Estimation Concept

Motion estimation is the process of determining how objects or visual patterns move between consecutive image frames. In computer vision, this is commonly achieved by comparing pixel intensities, object positions, or feature points over time.

For this project, the main focus is single-object motion estimation. A moving object, such as a car, is observed across a sequence of frames. The system estimates:

- **Where the object is located** in each frame.
- **How the object moves** between frames.
- **How fast the object moves** over time.
- **Which direction the object follows**.
- **What trajectory the object forms** across the sequence.

This makes the project useful for learning and demonstrating core computer vision concepts such as optical flow, object tracking, trajectory extraction, and motion analytics.

## Features

| Feature | Description |
|---|---|
| Data loading | Load image sequences and metadata files from structured folders. |
| Preprocessing | Prepare image frames for detection, flow estimation, and tracking. |
| Object detection | Identify the target moving object in each sequence. |
| Optical flow | Estimate frame-to-frame motion using dense or sparse flow techniques. |
| Tracking | Follow object position across the full sequence. |
| Trajectory extraction | Convert object positions into a continuous movement path. |
| Speed analysis | Estimate relative object speed across frames. |
| Direction analysis | Determine the dominant direction of object motion. |
| Visualization | Generate annotated frames, arrows, trajectory plots, graphs, and videos. |
| Interface | Provide a future Streamlit or Tkinter interface for interactive use. |

## Dataset Description

The dataset contains multiple car image sequences:

```text
car-1, car-2, ..., car-20
```

Each sequence contains image frames and annotation files:

| File or Folder | Description |
|---|---|
| `img/` | Folder containing ordered image frames for the sequence. |
| `groundtruth.txt` | Ground-truth bounding box annotations for the target object. |
| `full_occlusion.txt` | Frame-level indicators for full object occlusion. |
| `out_of_view.txt` | Frame-level indicators for object visibility outside the frame. |
| `nlp.txt` | Natural language or metadata description associated with the sequence. |

Expected dataset layout:

```text
data/
└── car/
    ├── car-1/
    │   ├── img/
    │   ├── groundtruth.txt
    │   ├── full_occlusion.txt
    │   ├── out_of_view.txt
    │   └── nlp.txt
    ├── car-2/
    │   └── ...
    └── car-20/
        └── ...
```

## Project Architecture

The project follows a modular architecture designed for clarity, maintainability, and experimentation.

```text
motion-estimation-project/
|
├── data/
│   └── car/
|
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── detection.py
│   ├── optical_flow.py
│   ├── tracking.py
│   ├── analysis.py
│   └── visualization.py
|
├── notebooks/
│   ├── 01_load_and_preprocess.ipynb
│   ├── 02_detection.ipynb
│   ├── 03_optical_flow.ipynb
│   ├── 04_tracking.ipynb
│   ├── 05_analysis.ipynb
│   └── 06_final_pipeline.ipynb
|
├── app/
│   ├── app.py
│   ├── pages/
│   ├── components/
│   └── utils/
|
├── results/
│   ├── images/
│   └── videos/
|
├── requirements.txt
└── README.md
```

## Folder Structure Explanation

| Path | Purpose |
|---|---|
| `data/` | Stores dataset files and image sequences. |
| `data/car/` | Contains car sequences from `car-1` to `car-20`. |
| `src/` | Contains reusable Python modules for the computer vision pipeline. |
| `src/load_data.py` | Intended for sequence loading and dataset parsing utilities. |
| `src/preprocess.py` | Intended for image preparation and frame transformation logic. |
| `src/detection.py` | Intended for object detection and localization methods. |
| `src/optical_flow.py` | Intended for optical flow estimation methods. |
| `src/tracking.py` | Intended for object tracking and trajectory construction. |
| `src/analysis.py` | Intended for speed, direction, displacement, and motion analysis. |
| `src/visualization.py` | Intended for visual outputs such as arrows, plots, overlays, and videos. |
| `notebooks/` | Contains experimental notebooks for each major project step. |
| `app/` | Contains the future interface layer for interactive usage. |
| `results/images/` | Stores generated figures, annotated frames, and plots. |
| `results/videos/` | Stores generated motion visualization videos. |

## Workflow

The expected project pipeline is organized as follows:

```text
Dataset
   |
   v
Load image sequence and annotations
   |
   v
Preprocess frames
   |
   v
Detect or localize target object
   |
   v
Compute optical flow
   |
   v
Track object trajectory
   |
   v
Analyze speed and direction
   |
   v
Generate visualizations and reports
```

Recommended notebook order:

1. `01_load_and_preprocess.ipynb`
2. `02_detection.ipynb`
3. `03_optical_flow.ipynb`
4. `04_tracking.ipynb`
5. `05_analysis.ipynb`
6. `06_final_pipeline.ipynb`

## Technologies Used

| Technology | Role |
|---|---|
| Python | Main programming language. |
| OpenCV | Image processing, optical flow, tracking, and video generation. |
| NumPy | Numerical computation and array manipulation. |
| Matplotlib | Graphs, plots, and visual analysis. |
| Pandas | Optional annotation and metric processing. |
| Jupyter Notebook | Experimentation and step-by-step analysis. |
| Streamlit or Tkinter | Future interactive interface layer. |

## Interface Concept

The `app/` folder is reserved for a user-facing interface. The interface may be implemented with **Streamlit** or **Tkinter**, depending on the final project direction.

Planned interface capabilities:

- Select a car sequence from the dataset.
- Preview image frames.
- Run preprocessing, detection, optical flow, tracking, and analysis steps.
- Display trajectory overlays and optical flow arrows.
- Show speed and direction graphs.
- Export visual results as images or videos.

### Demo Placeholder

Add screenshots or GIFs here when the interface is available.

```text
assets/demo-placeholder.gif
assets/interface-screenshot.png
```

## Optical Flow Methods

Optical flow estimates the apparent motion of pixels or visual features between two consecutive frames. It helps describe how the visual content of a scene changes over time.

This project can support two major optical flow families:

| Method Type | Description |
|---|---|
| Sparse optical flow | Tracks selected points or features across frames. Useful for lightweight tracking and feature-based motion analysis. |
| Dense optical flow | Estimates motion for most or all pixels. Useful for rich motion-field visualization and detailed scene analysis. |

Possible optical flow approaches:

- Lucas-Kanade optical flow.
- Farneback dense optical flow.
- Feature-based point tracking.
- Motion vector visualization using arrows or color maps.

## Tracking and Analysis

The tracking module is intended to estimate the object position across the full image sequence. Tracking can rely on ground-truth annotations, detected bounding boxes, optical flow vectors, or a combination of these signals.

Motion analysis may include:

- Object center extraction.
- Frame-to-frame displacement.
- Cumulative trajectory estimation.
- Relative speed estimation.
- Direction angle calculation.
- Horizontal and vertical movement decomposition.
- Occlusion-aware tracking behavior.
- Out-of-view handling.

The final analysis should help answer questions such as:

- Is the object moving left, right, up, or down?
- How does its speed change over time?
- Is the trajectory stable or irregular?
- Which frames contain occlusion or visibility issues?

## Visualization

Visualization is a key part of this project because motion estimation results are easier to interpret visually.

Planned visual outputs:

- Annotated frames with detected object bounding boxes.
- Optical flow arrows between frames.
- Trajectory path drawn over the image sequence.
- Object center-point movement plots.
- Speed-over-time graphs.
- Direction-over-time graphs.
- Side-by-side comparison of original and processed frames.
- Exported videos showing tracking and motion overlays.

### Results Preview

Add generated outputs here once available.

| Output Type | Placeholder |
|---|---|
| Annotated frame | `results/images/annotated_frame_placeholder.png` |
| Trajectory plot | `results/images/trajectory_plot_placeholder.png` |
| Optical flow visualization | `results/images/optical_flow_placeholder.png` |
| Tracking video | `results/videos/tracking_demo_placeholder.mp4` |

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd motion-estimation-project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> The exact dependency list should be maintained in `requirements.txt` as the implementation evolves.

## Usage

### 1. Prepare the Dataset

Place the car sequences inside:

```text
data/car/
```

Expected sequence folders:

```text
data/car/car-1/
data/car/car-2/
...
data/car/car-20/
```

### 2. Run Experiments in Notebooks

Open the notebooks in order to explore the complete pipeline:

```text
notebooks/01_load_and_preprocess.ipynb
notebooks/02_detection.ipynb
notebooks/03_optical_flow.ipynb
notebooks/04_tracking.ipynb
notebooks/05_analysis.ipynb
notebooks/06_final_pipeline.ipynb
```

### 3. Launch the Interface

When the interface is implemented, it will be launched from:

```text
app/app.py
```

For a Streamlit-based interface, the expected launch command will be:

```bash
streamlit run app/app.py
```

## Expected Outputs

The project is expected to generate the following outputs after implementation:

| Output | Destination |
|---|---|
| Preprocessed frames | `results/images/` |
| Detection overlays | `results/images/` |
| Optical flow visualizations | `results/images/` |
| Trajectory plots | `results/images/` |
| Speed and direction graphs | `results/images/` |
| Annotated tracking videos | `results/videos/` |
| Final demonstration media | `results/videos/` |

No sample results are included yet. Screenshots, plots, and videos should be added after the pipeline is implemented and evaluated.

## Future Improvements

- Add support for multiple object categories.
- Compare multiple optical flow algorithms.
- Add quantitative evaluation against ground-truth annotations.
- Improve robustness during occlusion and out-of-view frames.
- Add object re-identification after temporary disappearance.
- Build a complete Streamlit dashboard.
- Add automated tests for data loading and analysis utilities.
- Add configuration files for experiment reproducibility.
- Export structured reports in CSV, JSON, or PDF format.
- Add sample demo media for portfolio presentation.
