from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _to_rgb(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show_frames_grid(frames, titles=None):
    """Affiche une grille simple de frames."""
    n = len(frames)
    if n == 0:
        raise ValueError("Aucune frame a afficher.")

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(_to_rgb(frames[i]), cmap="gray" if frames[i].ndim == 2 else None)
            if titles:
                ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    return fig


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """Dessine une boite englobante x,y,w,h."""
    output = frame.copy()
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def draw_points(frame, points, color=(0, 255, 0), radius=4):
    """Dessine des points sur une copie de la frame."""
    output = frame.copy()
    if points is None:
        return output
    for x, y in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        cv2.circle(output, (int(round(x)), int(round(y))), radius, color, -1)
    return output


def draw_motion_field(frame, points_old, points_new):
    """Dessine un champ de mouvement sparse."""
    output = frame.copy()
    old = np.asarray(points_old, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(points_new, dtype=np.float32).reshape(-1, 2)
    for (x0, y0), (x1, y1) in zip(old, new):
        cv2.arrowedLine(
            output,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (0, 0, 255),
            2,
            tipLength=0.35,
        )
        cv2.circle(output, (int(round(x0)), int(round(y0))), 3, (0, 255, 0), -1)
        cv2.circle(output, (int(round(x1)), int(round(y1))), 3, (255, 0, 0), -1)
    return output


def plot_trajectory(trajectory_df, save_path=None):
    """Trace la trajectoire 2D estimee."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trajectory_df["center_x"], trajectory_df["center_y"], color="red", linewidth=2)
    ax.scatter(trajectory_df["center_x"].iloc[0], trajectory_df["center_y"].iloc[0], color="green", label="Start")
    ax.scatter(
        trajectory_df["center_x"].iloc[-1],
        trajectory_df["center_y"].iloc[-1],
        color="blue",
        label="End",
    )
    ax.set_title("Trajectoire globale estimee")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory_on_frame(frame, trajectory_df, save_path=None):
    """Dessine la trajectoire sur une frame."""
    output = frame.copy()
    points = trajectory_df[["center_x", "center_y"]].dropna().to_numpy(dtype=np.int32)

    if len(points) >= 2:
        cv2.polylines(output, [points.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

    if len(points) > 0:
        start = tuple(points[0])
        current = tuple(points[len(points) // 2])
        end = tuple(points[-1])
        cv2.circle(output, start, 7, (0, 255, 0), -1)
        cv2.circle(output, current, 7, (0, 255, 255), -1)
        cv2.circle(output, end, 7, (255, 0, 0), -1)
        cv2.putText(output, "Start", (start[0] + 8, start[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, "Current", (current[0] + 8, current[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output, "End", (end[0] + 8, end[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    _ensure_parent(save_path)
    if save_path is not None:
        cv2.imwrite(str(save_path), output)
    return output


def plot_speed(motion_df, save_path=None):
    """Trace la vitesse apparente en pixels/frame."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(motion_df["frame_id"], motion_df["speed_px_frame"], color="tab:orange")
    ax.set_title("Vitesse apparente")
    ax.set_xlabel("Frame")
    ax.set_ylabel("pixels/frame")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_direction(motion_df, save_path=None):
    """Trace la direction du mouvement en degres."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(motion_df["frame_id"], motion_df["direction_deg"], color="tab:purple")
    ax.set_title("Direction du mouvement")
    ax.set_xlabel("Frame")
    ax.set_ylabel("degres")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison_with_groundtruth(estimated_df, gt_df, save_path=None):
    """Trace la trajectoire estimee et la trajectoire groundtruth."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(estimated_df["center_x"], estimated_df["center_y"], color="red", label="Estimee")
    ax.plot(gt_df["center_x"], gt_df["center_y"], color="blue", linestyle="--", label="Groundtruth")
    ax.set_title("Comparaison trajectoire estimee / groundtruth")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
