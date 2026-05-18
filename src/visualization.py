"""Visualization helpers for segmentation, motion field and trajectory."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detection import validate_bbox


def draw_bbox(image, bbox):
    """Draw a bounding box on a BGR image."""
    if image is None:
        return None

    output = image.copy()

    if bbox is None:
        return output

    x, y, w, h = [int(value) for value in bbox]
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output


def draw_points(image, points):
    """Draw tracked feature points on a BGR image."""
    if image is None:
        return None

    output = image.copy()

    if points is None or len(points) == 0:
        return output

    points_array = np.asarray(points).reshape(-1, 2)

    for point in points_array:
        x, y = point.astype(int)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

    return output


def draw_motion_vectors(image, old_points, new_points):
    """Draw motion vectors between two point sets."""
    if image is None:
        return None

    output = image.copy()

    if old_points is None or new_points is None:
        return output

    old_points = np.asarray(old_points).reshape(-1, 2)
    new_points = np.asarray(new_points).reshape(-1, 2)

    for old_point, new_point in zip(old_points, new_points):
        old_x, old_y = old_point.astype(int)
        new_x, new_y = new_point.astype(int)
        cv2.arrowedLine(output, (old_x, old_y), (new_x, new_y), (255, 0, 0), 2)
        cv2.circle(output, (old_x, old_y), 3, (0, 255, 0), -1)
        cv2.circle(output, (new_x, new_y), 3, (0, 0, 255), -1)

    return output


def draw_trajectory(image, trajectory):
    """Draw a global trajectory on a BGR image."""
    if image is None:
        return None

    output = image.copy()

    if trajectory is None or len(trajectory) == 0:
        return output

    trajectory_points = np.asarray(trajectory).reshape(-1, 2).astype(int)

    for index, point in enumerate(trajectory_points):
        x, y = point
        cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

        if index > 0:
            prev_x, prev_y = trajectory_points[index - 1]
            cv2.line(output, (prev_x, prev_y), (x, y), (255, 0, 0), 2)

    return output


def draw_current_center(image, center):
    """Draw the current object center."""
    if image is None:
        return None

    output = image.copy()

    if center is None:
        return output

    x, y = [int(value) for value in center]
    cv2.circle(output, (x, y), 6, (0, 255, 255), -1)

    return output


def save_frame(image, output_path):
    """Save a frame to disk."""
    if image is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return cv2.imwrite(str(output_path), image)


def create_tracking_video(image_files, trajectory_df, output_path, fps=20):
    """Create a video with the progressive trajectory."""
    if image_files is None or len(image_files) == 0:
        return False

    if trajectory_df is None or len(trajectory_df) == 0:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = None
    for image_file in image_files:
        first_frame = cv2.imread(str(image_file))
        if first_frame is not None:
            break

    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        return False

    trajectory = []
    written_frames = 0

    for _, row in trajectory_df.iterrows():
        frame_index = int(row["frame"]) if "frame" in trajectory_df.columns else written_frames

        if frame_index < 0 or frame_index >= len(image_files):
            continue

        image = cv2.imread(str(image_files[frame_index]))

        if image is None:
            continue

        center = (float(row["x"]), float(row["y"]))
        trajectory.append(center)

        output = draw_trajectory(image, trajectory)
        output = draw_current_center(output, center)
        writer.write(output)
        written_frames += 1

    writer.release()

    return written_frames > 0


def draw_mask_overlay(frame_bgr, mask, bbox=None):
    """Overlay a segmentation mask on a BGR frame in green."""
    if frame_bgr is None:
        return None

    output = frame_bgr.copy()
    if mask is None:
        return output

    overlay = output.copy()

    if bbox is None:
        mask_full = mask
        if mask_full.shape[:2] != output.shape[:2]:
            mask_full = cv2.resize(mask_full, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[mask_full > 0] = (0, 255, 0)
    else:
        bbox = validate_bbox(bbox, frame_bgr.shape)
        if bbox is None:
            return output
        x, y, w, h = bbox
        roi_mask = mask
        if roi_mask.shape[:2] != (h, w):
            roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_overlay = overlay[y:y + h, x:x + w]
        roi_overlay[roi_mask > 0] = (0, 255, 0)
        overlay[y:y + h, x:x + w] = roi_overlay

    return cv2.addWeighted(overlay, 0.35, output, 0.65, 0)


def draw_edges_overlay(frame_bgr, edges, bbox=None):
    """Overlay Canny edges on a BGR frame in red."""
    if frame_bgr is None:
        return None

    output = frame_bgr.copy()
    if edges is None:
        return output

    if bbox is None:
        edge_mask = edges
        if edge_mask.shape[:2] != output.shape[:2]:
            edge_mask = cv2.resize(edge_mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        output[edge_mask > 0] = (0, 0, 255)
    else:
        bbox = validate_bbox(bbox, frame_bgr.shape)
        if bbox is None:
            return output
        x, y, w, h = bbox
        roi_edges = edges
        if roi_edges.shape[:2] != (h, w):
            roi_edges = cv2.resize(roi_edges, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_output = output[y:y + h, x:x + w]
        roi_output[roi_edges > 0] = (0, 0, 255)
        output[y:y + h, x:x + w] = roi_output

    return output


def draw_motion_mask_overlay(frame_bgr, motion_mask, bbox=None):
    """Overlay frame-difference motion mask on a BGR frame in cyan."""
    if frame_bgr is None:
        return None

    output = frame_bgr.copy()
    if motion_mask is None:
        return output

    overlay = output.copy()

    if bbox is None:
        mask_full = motion_mask
        if mask_full.shape[:2] != output.shape[:2]:
            mask_full = cv2.resize(mask_full, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[mask_full > 0] = (0, 255, 255)
    else:
        bbox = validate_bbox(bbox, frame_bgr.shape)
        if bbox is None:
            return output
        x, y, w, h = bbox
        roi_mask = motion_mask
        if roi_mask.shape[:2] != (h, w):
            roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_overlay = overlay[y:y + h, x:x + w]
        roi_overlay[roi_mask > 0] = (0, 255, 255)
        overlay[y:y + h, x:x + w] = roi_overlay

    return cv2.addWeighted(overlay, 0.35, output, 0.65, 0)


def _to_rgb_for_plot(image):
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_comparison_grid(images_dict, output_path, title=None):
    """Save a grid comparing preprocessing, segmentation and masks."""
    if not images_dict:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = [(name, image) for name, image in images_dict.items() if image is not None]
    if not items:
        return False

    cols = min(4, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (name, image) in zip(axes, items):
        plot_image = _to_rgb_for_plot(image)
        if plot_image.ndim == 2:
            ax.imshow(plot_image, cmap="gray")
        else:
            ax.imshow(plot_image)
        ax.set_title(str(name))
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _records_to_df(records):
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame(records)


def plot_trajectory(records, output_path):
    """Save the 2D global trajectory graph."""
    df = _records_to_df(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not df.empty:
        ax.plot(df["x"], df["y"], marker="o", linewidth=2)
    ax.invert_yaxis()
    ax.set_title("Trajectoire globale")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_speed(records, output_path):
    """Save speed over time."""
    df = _records_to_df(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        ax.plot(df["frame"], df["speed_px_per_frame"], color="tab:blue", linewidth=2)
    ax.set_title("Vitesse")
    ax.set_xlabel("Frame")
    ax.set_ylabel("pixels/frame")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_direction(records, output_path):
    """Save direction over time."""
    df = _records_to_df(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        ax.plot(df["frame"], df["direction_deg"], color="tab:green", linewidth=2)
    ax.set_title("Direction")
    ax.set_xlabel("Frame")
    ax.set_ylabel("degres")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_positions(records, output_path):
    """Save x(t) and y(t) position curves."""
    df = _records_to_df(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        ax.plot(df["frame"], df["x"], label="x", linewidth=2)
        ax.plot(df["frame"], df["y"], label="y", linewidth=2)
        ax.legend()
    ax.set_title("Positions du centre")
    ax.set_xlabel("Frame")
    ax.set_ylabel("pixels")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def save_all_analysis_graphs(records, output_dir):
    """Save trajectory, speed, direction and position graphs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory(records, output_dir / "trajectory_2d.png")
    plot_speed(records, output_dir / "speed.png")
    plot_direction(records, output_dir / "direction.png")
    plot_positions(records, output_dir / "positions.png")

    return {
        "trajectory": output_dir / "trajectory_2d.png",
        "speed": output_dir / "speed.png",
        "direction": output_dir / "direction.png",
        "positions": output_dir / "positions.png",
    }
