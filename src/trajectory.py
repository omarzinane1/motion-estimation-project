"""Trajectory extraction helpers."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def compute_object_center(points):
    """Compute the mean center of tracked points."""
    if points is None or len(points) == 0:
        return None

    points_array = np.asarray(points).reshape(-1, 2)
    center_x = float(np.mean(points_array[:, 0]))
    center_y = float(np.mean(points_array[:, 1]))

    return center_x, center_y


def update_trajectory(trajectory, center):
    """Append a center to a trajectory list."""
    if center is not None:
        trajectory.append(center)

    return trajectory


def save_trajectory(trajectory_df, output_path):
    """Save a trajectory DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(output_path, index=False)


def draw_trajectory_on_image(image, trajectory):
    """Draw a trajectory polyline on an image."""
    image_with_trajectory = image.copy()

    if trajectory is None or len(trajectory) == 0:
        return image_with_trajectory

    trajectory_points = np.asarray(trajectory, dtype=np.int32)

    for index, point in enumerate(trajectory_points):
        x, y = point
        cv2.circle(image_with_trajectory, (x, y), 3, (0, 0, 255), -1)

        if index > 0:
            previous_x, previous_y = trajectory_points[index - 1]
            cv2.line(
                image_with_trajectory,
                (previous_x, previous_y),
                (x, y),
                (255, 0, 0),
                2,
            )

    return image_with_trajectory


def bbox_center(bbox):
    """Return the center of a bounding box."""
    if bbox is None:
        return None
    x, y, w, h = [float(value) for value in bbox]
    return x + w / 2.0, y + h / 2.0


def update_bbox_by_motion(bbox, dx, dy):
    """Translate a bbox by the estimated global displacement."""
    if bbox is None:
        return None
    x, y, w, h = [float(value) for value in bbox]
    return int(round(x + dx)), int(round(y + dy)), int(round(w)), int(round(h))


def add_trajectory_record(records, frame_index, bbox, dx, dy, tracked_points):
    """Append one trajectory-analysis record."""
    if records is None:
        records = []

    if bbox is None:
        return records

    x, y, w, h = [int(round(float(value))) for value in bbox]
    cx, cy = bbox_center((x, y, w, h))
    speed = float(np.sqrt(float(dx) ** 2 + float(dy) ** 2))
    direction = float(np.degrees(np.arctan2(float(dy), float(dx))))

    records.append({
        "frame": int(frame_index),
        "x": float(cx),
        "y": float(cy),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "dx": float(dx),
        "dy": float(dy),
        "speed_px_per_frame": speed,
        "direction_deg": direction,
        "tracked_points": int(tracked_points or 0),
    })

    return records


def save_trajectory_csv(records, output_path):
    """Save trajectory records to CSV and return the DataFrame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df = pd.DataFrame(records)
    trajectory_df.to_csv(output_path, index=False)
    return trajectory_df


def extract_global_trajectory(records):
    """Return centers as ``[(x1, y1), (x2, y2), ...]``."""
    if records is None:
        return []
    return [(float(record["x"]), float(record["y"])) for record in records]
