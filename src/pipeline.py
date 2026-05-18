"""Final tracking pipeline without groundtruth annotations.

Workflow:
frames -> preprocessing -> segmentation -> morphology -> Canny/difference
-> Lucas-Kanade -> global trajectory -> speed/direction -> visual outputs.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analysis import compute_summary_from_records, save_interpretation_results
from src.detection import (
    adaptive_threshold,
    bbox_from_mask,
    clean_segmentation_mask,
    detect_features_in_roi,
    filter_features_by_mask,
    largest_connected_component,
    otsu_threshold,
    segment_and_clean_car,
    select_initial_roi,
    validate_bbox,
)
from src.edge_detection import canny_on_roi
from src.motion_detection import detect_motion_between_frames, motion_detection_on_roi
from src.optical_flow import draw_optical_flow_vectors, estimate_motion_field
from src.preprocessing import compare_preprocessing_methods, preprocess_image_with_method
from src.trajectory import (
    add_trajectory_record,
    extract_global_trajectory,
    save_trajectory_csv,
    update_bbox_by_motion,
)
from src.visualization import (
    draw_bbox,
    draw_edges_overlay,
    draw_mask_overlay,
    draw_motion_mask_overlay,
    draw_trajectory,
    save_all_analysis_graphs,
    save_comparison_grid,
    save_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "car" / "car-11"
IMG_PATH = DATASET_PATH / "img"
RESULTS_PATH = PROJECT_ROOT / "results"


def _get_image_files(img_path):
    """Return image files sorted in temporal order."""
    img_path = Path(img_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not img_path.exists():
        return []

    return sorted([
        file for file in img_path.iterdir()
        if file.suffix.lower() in image_extensions
    ])


def _validate_frames(start_frame, end_frame, total_images):
    """Validate the frame interval."""
    if total_images <= 0:
        raise ValueError("Aucune image n'a ete trouvee dans le dossier images.")

    if start_frame < 0:
        raise ValueError("La frame de depart doit etre superieure ou egale a 0.")

    if end_frame is None:
        end_frame = total_images

    if end_frame > total_images:
        raise ValueError("La frame de fin depasse le nombre total d'images.")

    if end_frame <= start_frame:
        raise ValueError("La frame de fin doit etre superieure a la frame de depart.")

    return int(start_frame), int(end_frame)


def _create_result_dirs(results_path):
    """Create every results directory required by the project."""
    results_path = Path(results_path)
    dirs = {
        "root": results_path,
        "preprocessing": results_path / "preprocessing",
        "segmentation": results_path / "segmentation",
        "morphology": results_path / "morphology",
        "edge_detection": results_path / "edge_detection",
        "motion_detection": results_path / "motion_detection",
        "optical_flow": results_path / "optical_flow",
        "trajectory": results_path / "trajectory",
        "graphs": results_path / "graphs",
        "final_visualization": results_path / "final_visualization",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def _mask_for_child_bbox(parent_bbox, parent_mask, child_bbox):
    """Crop a ROI mask so it matches a child bounding box."""
    if parent_bbox is None or parent_mask is None or child_bbox is None:
        return None

    px, py, pw, ph = [int(value) for value in parent_bbox]
    cx, cy, cw, ch = [int(value) for value in child_bbox]

    local_x = max(0, cx - px)
    local_y = max(0, cy - py)
    local_x2 = min(pw, local_x + cw)
    local_y2 = min(ph, local_y + ch)

    if local_x2 <= local_x or local_y2 <= local_y:
        return None

    cropped = parent_mask[local_y:local_y2, local_x:local_x2]
    if cropped.shape[:2] != (ch, cw):
        full = np.zeros((ch, cw), dtype=np.uint8)
        full[:cropped.shape[0], :cropped.shape[1]] = cropped
        cropped = full

    return cropped


def _bbox_area(bbox):
    if bbox is None:
        return 0
    _, _, w, h = bbox
    return int(w) * int(h)


def _is_reasonable_segmented_bbox(candidate_bbox, reference_bbox):
    """Reject tiny segmentation artifacts."""
    if candidate_bbox is None or reference_bbox is None:
        return False

    candidate_area = _bbox_area(candidate_bbox)
    reference_area = _bbox_area(reference_bbox)
    if reference_area <= 0:
        return False

    return candidate_area >= 0.08 * reference_area


def _detect_points_with_optional_mask(gray_image, bbox, mask=None):
    """Detect and optionally mask feature points."""
    points = detect_features_in_roi(gray_image, bbox)
    if mask is not None:
        points = filter_features_by_mask(points, bbox, mask)
    return points


def _points_count(points):
    if points is None:
        return 0
    return int(len(points))


def _should_save_frame(frame_index, start_frame, end_frame):
    """Limit saved per-frame images while keeping representative outputs."""
    total = max(1, end_frame - start_frame)
    stride = max(1, total // 80)
    return (
        frame_index in {start_frame, start_frame + 1, end_frame - 1}
        or (frame_index - start_frame) % stride == 0
    )


def _save_initial_preprocessing(frame_bgr, dirs):
    """Save preprocessing comparison for the first frame."""
    comparisons = compare_preprocessing_methods(frame_bgr)
    images = {"original": frame_bgr}
    images.update(comparisons)
    save_comparison_grid(
        images,
        dirs["preprocessing"] / "preprocessing_histogram_comparison.png",
        title="Comparaison pretraitement et contraste",
    )
    save_comparison_grid(
        images,
        dirs["final_visualization"] / "preprocessing_histogram_comparison.png",
        title="Comparaison pretraitement et contraste",
    )


def _save_segmentation_comparison(gray_image, frame_bgr, bbox, dirs):
    """Save Otsu, adaptive and cleaned-mask comparison on the initial ROI."""
    bbox = validate_bbox(bbox, gray_image.shape)
    if bbox is None:
        return None

    x, y, w, h = bbox
    roi = gray_image[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    otsu_mask = otsu_threshold(roi, invert=True)
    adaptive_mask = adaptive_threshold(roi, invert=True)
    clean_mask = clean_segmentation_mask(otsu_mask, kernel_size=(5, 5))
    largest = largest_connected_component(clean_mask)
    segmented_bbox = bbox_from_mask(largest, parent_bbox=bbox)

    save_frame(otsu_mask, dirs["segmentation"] / "mask_otsu_initial.png")
    save_frame(adaptive_mask, dirs["segmentation"] / "mask_adaptive_initial.png")
    save_frame(clean_mask, dirs["morphology"] / "mask_cleaned_initial.png")
    save_frame(largest, dirs["morphology"] / "largest_component_initial.png")
    if segmented_bbox is not None:
        save_frame(draw_bbox(frame_bgr, segmented_bbox), dirs["segmentation"] / "bbox_from_mask_initial.png")

    save_comparison_grid(
        {
            "original": frame_bgr,
            "roi": roi,
            "otsu": otsu_mask,
            "adaptive": adaptive_mask,
            "mask cleaned": clean_mask,
            "largest component": largest,
        },
        dirs["final_visualization"] / "segmentation_morphology_comparison.png",
        title="Segmentation et morphologie",
    )

    return {
        "otsu_mask": otsu_mask,
        "adaptive_mask": adaptive_mask,
        "clean_mask": clean_mask,
        "largest_component": largest,
        "segmented_bbox": segmented_bbox,
    }


def run_tracking_without_groundtruth(
    start_frame=0,
    end_frame=None,
    fps=None,
    initial_bbox=None,
    use_manual_roi=False,
    preprocess_method="stretching",
    segmentation_method="otsu",
    invert_mask=True,
    use_segmentation=True,
    save_outputs=True,
    image_dir=None,
):
    """Run the complete car-tracking pipeline without reading annotations."""
    start_frame = int(start_frame)
    if end_frame is not None:
        end_frame = int(end_frame)
    if fps is not None:
        fps = float(fps)

    img_path = Path(image_dir) if image_dir is not None else IMG_PATH
    image_files = _get_image_files(img_path)
    start_frame, end_frame = _validate_frames(start_frame, end_frame, len(image_files))

    first_frame = cv2.imread(str(image_files[start_frame]))
    if first_frame is None:
        raise ValueError("La frame de depart n'a pas pu etre chargee.")

    if initial_bbox is None and use_manual_roi:
        initial_bbox = select_initial_roi(first_frame)

    if initial_bbox is None:
        raise ValueError("Veuillez fournir initial_bbox=(x,y,w,h) ou sélectionner une ROI manuellement.")

    initial_bbox = validate_bbox(initial_bbox, first_frame.shape)
    if initial_bbox is None:
        raise ValueError("La bounding box initiale est invalide.")

    dirs = _create_result_dirs(RESULTS_PATH)
    notes = []

    first_preprocessed = preprocess_image_with_method(first_frame, method=preprocess_method)
    prev_gray = first_preprocessed["enhanced"]

    if save_outputs:
        _save_initial_preprocessing(first_frame, dirs)
        _save_segmentation_comparison(prev_gray, first_frame, initial_bbox, dirs)

    current_bbox = initial_bbox
    current_mask = None

    if use_segmentation:
        first_segmentation = segment_and_clean_car(
            prev_gray,
            current_bbox,
            method=segmentation_method,
            invert=invert_mask,
        )
        segmented_bbox = first_segmentation["segmented_bbox"]
        if _is_reasonable_segmented_bbox(segmented_bbox, current_bbox):
            parent_bbox = current_bbox
            current_bbox = validate_bbox(segmented_bbox, prev_gray.shape)
            current_mask = _mask_for_child_bbox(parent_bbox, first_segmentation["largest_component"], current_bbox)
        else:
            notes.append("Segmentation initiale faible: bbox fournie conservee.")
            current_mask = first_segmentation["largest_component"]

        if save_outputs:
            save_frame(first_segmentation["raw_mask"], dirs["segmentation"] / f"raw_mask_{start_frame:06d}.png")
            save_frame(first_segmentation["clean_mask"], dirs["morphology"] / f"clean_mask_{start_frame:06d}.png")
            save_frame(
                draw_mask_overlay(first_frame, first_segmentation["largest_component"], bbox=initial_bbox),
                dirs["segmentation"] / f"mask_overlay_{start_frame:06d}.png",
            )

    points = _detect_points_with_optional_mask(prev_gray, current_bbox, current_mask)
    if _points_count(points) == 0:
        notes.append("Aucun point Lucas-Kanade detecte au depart: nouvelle detection sans masque.")
        points = detect_features_in_roi(prev_gray, current_bbox)

    records = []
    records = add_trajectory_record(
        records,
        frame_index=start_frame,
        bbox=current_bbox,
        dx=0.0,
        dy=0.0,
        tracked_points=_points_count(points),
    )

    trajectory_centers = extract_global_trajectory(records)
    last_optical_flow_image = draw_bbox(first_frame, current_bbox)
    last_trajectory_image = draw_trajectory(first_frame, trajectory_centers)

    if save_outputs:
        save_frame(draw_bbox(first_frame, current_bbox), dirs["final_visualization"] / "initial_bbox.png")
        edges = canny_on_roi(prev_gray, current_bbox)
        save_frame(draw_edges_overlay(first_frame, edges, bbox=current_bbox), dirs["edge_detection"] / f"edges_{start_frame:06d}.png")

    for frame_index in range(start_frame + 1, end_frame):
        frame_bgr = cv2.imread(str(image_files[frame_index]))
        if frame_bgr is None:
            notes.append(f"Frame illisible a l'indice {frame_index}: arret propre du suivi.")
            break

        preprocessed = preprocess_image_with_method(frame_bgr, method=preprocess_method)
        current_gray = preprocessed["enhanced"]
        previous_bbox = current_bbox

        motion_roi = motion_detection_on_roi(prev_gray, current_gray, previous_bbox, threshold=25)
        motion_full = detect_motion_between_frames(prev_gray, current_gray, threshold=25)

        motion_field = estimate_motion_field(prev_gray, current_gray, points)
        num_points = int(motion_field["num_points"])

        if num_points > 0:
            dx_global = float(motion_field["dx_global"])
            dy_global = float(motion_field["dy_global"])
            predicted_bbox = update_bbox_by_motion(previous_bbox, dx_global, dy_global)
            predicted_bbox = validate_bbox(predicted_bbox, current_gray.shape) or previous_bbox
        else:
            notes.append(f"Frame {frame_index}: points LK perdus, bbox precedente conservee.")
            dx_global = 0.0
            dy_global = 0.0
            predicted_bbox = previous_bbox

        current_bbox = predicted_bbox
        current_mask = None
        segmentation_result = None

        if use_segmentation:
            segmentation_result = segment_and_clean_car(
                current_gray,
                current_bbox,
                method=segmentation_method,
                invert=invert_mask,
            )
            segmented_bbox = segmentation_result["segmented_bbox"]
            if _is_reasonable_segmented_bbox(segmented_bbox, current_bbox):
                parent_bbox = current_bbox
                current_bbox = validate_bbox(segmented_bbox, current_gray.shape) or current_bbox
                current_mask = _mask_for_child_bbox(parent_bbox, segmentation_result["largest_component"], current_bbox)
            else:
                notes.append(f"Frame {frame_index}: segmentation faible, bbox predite conservee.")
                current_mask = segmentation_result["largest_component"]

        if num_points >= 5 and motion_field["good_new"] is not None:
            next_points = motion_field["good_new"].reshape(-1, 1, 2).astype(np.float32)
            if current_mask is not None:
                filtered = filter_features_by_mask(next_points, current_bbox, current_mask)
                if _points_count(filtered) >= 5:
                    next_points = filtered
                else:
                    next_points = _detect_points_with_optional_mask(current_gray, current_bbox, current_mask)
        else:
            next_points = _detect_points_with_optional_mask(current_gray, current_bbox, current_mask)

        if _points_count(next_points) < 5:
            redetected = detect_features_in_roi(current_gray, current_bbox)
            if _points_count(redetected) > 0:
                next_points = redetected
                notes.append(f"Frame {frame_index}: redetection des points dans la derniere bbox.")
            else:
                next_points = None
                notes.append(f"Frame {frame_index}: aucun point detecte, tentative a la frame suivante.")

        records = add_trajectory_record(
            records,
            frame_index=frame_index,
            bbox=current_bbox,
            dx=dx_global,
            dy=dy_global,
            tracked_points=num_points,
        )
        trajectory_centers = extract_global_trajectory(records)

        optical_flow_image = draw_optical_flow_vectors(frame_bgr, motion_field["good_old"], motion_field["good_new"])
        optical_flow_image = draw_bbox(optical_flow_image, current_bbox)
        trajectory_image = draw_trajectory(frame_bgr, trajectory_centers)
        trajectory_image = draw_bbox(trajectory_image, current_bbox)

        last_optical_flow_image = optical_flow_image
        last_trajectory_image = trajectory_image

        if save_outputs and _should_save_frame(frame_index, start_frame, end_frame):
            if segmentation_result is not None:
                save_frame(segmentation_result["raw_mask"], dirs["segmentation"] / f"raw_mask_{frame_index:06d}.png")
                save_frame(segmentation_result["clean_mask"], dirs["morphology"] / f"clean_mask_{frame_index:06d}.png")
                save_frame(
                    draw_mask_overlay(frame_bgr, segmentation_result["largest_component"], bbox=predicted_bbox),
                    dirs["segmentation"] / f"mask_overlay_{frame_index:06d}.png",
                )

            edges = canny_on_roi(current_gray, current_bbox)
            save_frame(draw_edges_overlay(frame_bgr, edges, bbox=current_bbox), dirs["edge_detection"] / f"edges_{frame_index:06d}.png")
            save_frame(motion_full["difference"], dirs["motion_detection"] / f"difference_{frame_index:06d}.png")
            save_frame(motion_full["motion_mask"], dirs["motion_detection"] / f"motion_mask_{frame_index:06d}.png")
            save_frame(
                draw_motion_mask_overlay(frame_bgr, motion_roi["motion_mask"], bbox=previous_bbox),
                dirs["motion_detection"] / f"motion_overlay_{frame_index:06d}.png",
            )
            save_frame(optical_flow_image, dirs["optical_flow"] / f"lk_vectors_{frame_index:06d}.png")
            save_frame(trajectory_image, dirs["trajectory"] / f"trajectory_{frame_index:06d}.png")

        points = next_points
        prev_gray = current_gray

    trajectory_df = save_trajectory_csv(records, RESULTS_PATH / "trajectory.csv")
    if fps is not None and not trajectory_df.empty:
        trajectory_df["speed_px_per_second"] = trajectory_df["speed_px_per_frame"] * fps
        trajectory_df.to_csv(RESULTS_PATH / "trajectory.csv", index=False)

    summary = compute_summary_from_records(records)
    summary["warnings"] = notes
    summary["outputs"] = {
        "trajectory_csv": str(RESULTS_PATH / "trajectory.csv"),
        "interpretation_txt": str(RESULTS_PATH / "interpretation_results.txt"),
        "segmentation": str(dirs["segmentation"]),
        "morphology": str(dirs["morphology"]),
        "edge_detection": str(dirs["edge_detection"]),
        "motion_detection": str(dirs["motion_detection"]),
        "optical_flow": str(dirs["optical_flow"]),
        "trajectory": str(dirs["trajectory"]),
        "graphs": str(dirs["graphs"]),
        "final_visualization": str(dirs["final_visualization"]),
    }

    if save_outputs:
        save_frame(last_optical_flow_image, dirs["final_visualization"] / "final_motion_field_lk.png")
        save_frame(last_trajectory_image, dirs["final_visualization"] / "final_trajectory.png")
        save_all_analysis_graphs(records, dirs["graphs"])
        save_interpretation_results(summary, RESULTS_PATH / "interpretation_results.txt")

    return {
        "records": records,
        "trajectory_df": trajectory_df,
        "summary": summary,
    }


def run_tracking(start_frame=0, end_frame=100, fps=None, initial_bbox=None, use_manual_roi=True):
    """Backward-compatible wrapper around the final no-groundtruth pipeline."""
    result = run_tracking_without_groundtruth(
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        initial_bbox=initial_bbox,
        use_manual_roi=use_manual_roi,
        save_outputs=True,
    )
    trajectory_df = result["trajectory_df"]
    return trajectory_df, trajectory_df.copy(), result["summary"]
