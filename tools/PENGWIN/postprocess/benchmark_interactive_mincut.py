"""Benchmark oracle-seeded interactive min-cut on a known fragment pair.

Ground truth is used only to sample localized source/sink interactions and to
score the result. The graph cost is built exclusively from ABBC predictions
and an optional predicted label-3 probability map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label as nd_label

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    Seed,
    SplitConfig,
    build_cost_field,
    core_first_partition,
    min_cut_partition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abbc-pred", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--source-gt", required=True)
    parser.add_argument("--sink-gt", required=True)
    parser.add_argument(
        "--instance-id",
        type=int,
        help="Naturally under-split predicted instance; no artificial merge is performed",
    )
    parser.add_argument("--source-instance-id", type=int)
    parser.add_argument("--sink-instance-id", type=int)
    parser.add_argument("--prob-label-3")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--case-id", default="unknown")
    parser.add_argument(
        "--split-mode",
        choices=("core-first", "full-mincut"),
        default="core-first",
    )
    parser.add_argument(
        "--seed-region",
        choices=("core", "instance"),
        default="core",
        help="Constrain oracle interactions to predicted Label 2 or the full instance",
    )
    parser.add_argument(
        "--primary-endpoint",
        choices=("instance-iou", "core-recall"),
        default="instance-iou",
        help="Primary success criterion; instance IoU evaluates the final output masks",
    )
    parser.add_argument(
        "--require-contained-interaction",
        action="store_true",
        help="Require the complete brush radius to stay inside the GT sampling region",
    )
    parser.add_argument(
        "--center-abbc-labels",
        type=int,
        nargs="+",
        help="Optionally restrict random click centers to predicted ABBC labels",
    )
    parser.add_argument("--seed-radii-mm", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--trials-per-radius", type=int, default=20)
    parser.add_argument("--seed-clearance-mm", type=float, default=0.5)
    parser.add_argument("--roi-padding-mm", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=95)
    parser.add_argument(
        "--sampling-strategy",
        choices=("deepest-plus-random", "random"),
        default="deepest-plus-random",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--core-recall-threshold", type=float, default=0.9)
    parser.add_argument("--min-split-piece-size", type=int, default=50)
    parser.add_argument("--w-prob3", type=float, default=4.0)
    parser.add_argument("--w-prob1", type=float, default=6.0)
    parser.add_argument("--w-prob2", type=float, default=1.0)
    parser.add_argument("--w-min", type=float, default=0.05)
    parser.add_argument("--core-anchor-erosion-mm", type=float, default=1.0)
    parser.add_argument("--core-anchor-min-voxels", type=int, default=20)
    parser.add_argument("--core-anchor-distinct-max-distance-mm", type=float, default=15.0)
    parser.add_argument("--core-first-max-graph-voxels", type=int, default=5_000_000)
    return parser


def _image_info(path: Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {
        "size": tuple(int(v) for v in reader.GetSize()),
        "spacing": tuple(float(v) for v in reader.GetSpacing()),
        "origin": tuple(float(v) for v in reader.GetOrigin()),
        "direction": tuple(float(v) for v in reader.GetDirection()),
    }


def _assert_aligned(reference: Path, others: list[Path]) -> dict[str, Any]:
    reference_info = _image_info(reference)
    for path in others:
        info = _image_info(path)
        if info["size"] != reference_info["size"]:
            raise ValueError(f"grid size mismatch: {path}")
        for field in ("spacing", "origin", "direction"):
            if not np.allclose(info[field], reference_info[field]):
                raise ValueError(f"grid {field} mismatch: {path}")
    return reference_info


def _label_bbox(path: Path, label_id: int | None) -> tuple[int, int, int, int, int, int]:
    image = sitk.ReadImage(str(path))
    if label_id is None:
        image = sitk.Cast(image > 0, sitk.sitkUInt8)
        label_id = 1
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    if not stats.HasLabel(label_id):
        raise ValueError(f"label {label_id} is absent from {path}")
    bbox = tuple(int(v) for v in stats.GetBoundingBox(label_id))
    del image, stats
    return bbox


def _union_roi(
    bboxes: list[tuple[int, int, int, int, int, int]],
    image_size_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    padding_mm: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    starts = np.asarray([bbox[:3] for bbox in bboxes], dtype=int)
    ends = np.asarray(
        [np.asarray(bbox[:3]) + np.asarray(bbox[3:]) for bbox in bboxes],
        dtype=int,
    )
    pad = np.ceil(padding_mm / np.asarray(spacing_xyz)).astype(int)
    lower = np.maximum(starts.min(axis=0) - pad, 0)
    upper = np.minimum(ends.max(axis=0) + pad, np.asarray(image_size_xyz))
    return tuple(int(v) for v in lower), tuple(int(v) for v in upper - lower)


def _read_roi(path: Path, index_xyz: tuple[int, int, int], size_xyz: tuple[int, int, int]) -> np.ndarray:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.SetExtractIndex(index_xyz)
    reader.SetExtractSize(size_xyz)
    return sitk.GetArrayFromImage(reader.Execute())


def _write_roi_image(
    path: Path,
    array: np.ndarray,
    grid: dict[str, Any],
    roi_index_xyz: tuple[int, int, int],
) -> None:
    image = sitk.GetImageFromArray(array)
    direction = np.asarray(grid["direction"], dtype=float).reshape(3, 3)
    offset = np.asarray(roi_index_xyz, dtype=float) * np.asarray(grid["spacing"], dtype=float)
    roi_origin = np.asarray(grid["origin"], dtype=float) + direction @ offset
    image.SetSpacing(grid["spacing"])
    image.SetOrigin(tuple(float(value) for value in roi_origin))
    image.SetDirection(grid["direction"])
    sitk.WriteImage(image, str(path))


def _sphere_seed(
    allowed_mask: np.ndarray,
    center_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    radius_voxels = np.ceil(radius_mm / spacing_zyx).astype(int)
    lower = np.maximum(center_zyx - radius_voxels, 0)
    upper = np.minimum(center_zyx + radius_voxels + 1, allowed_mask.shape)
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    grids = np.ogrid[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))]
    squared_distance = sum(
        ((grid - int(center_zyx[axis])) * spacing_zyx[axis]) ** 2
        for axis, grid in enumerate(grids)
    )
    seed = np.zeros(allowed_mask.shape, dtype=bool)
    seed[crop] = (squared_distance <= radius_mm**2) & allowed_mask[crop]
    return seed


def _sample_centers(
    allowed_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    radius_mm: float,
    clearance_mm: float,
    trials: int,
    rng: np.random.Generator,
    require_contained_seed: bool = True,
    include_deepest: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    depth = distance_transform_edt(allowed_mask, sampling=spacing_zyx)
    required_depth = clearance_mm + (radius_mm if require_contained_seed else 0.0)
    candidates = np.argwhere(depth >= required_depth)
    if candidates.size == 0:
        raise ValueError(
            f"no valid seed centers for required depth={required_depth} mm"
        )

    if include_deepest:
        deepest = np.asarray(np.unravel_index(int(np.argmax(depth)), depth.shape), dtype=int)
        remaining = candidates[~np.all(candidates == deepest, axis=1)]
        if len(remaining) < trials - 1:
            raise ValueError(
                f"requested {trials} unique centers but only {len(remaining) + 1} are available"
            )
        indices = rng.choice(len(remaining), size=trials - 1, replace=False)
        centers = [deepest]
        centers.extend(
            np.asarray(remaining[index], dtype=int) for index in np.atleast_1d(indices)
        )
    else:
        if len(candidates) < trials:
            raise ValueError(
                f"requested {trials} unique centers but only {len(candidates)} are available"
            )
        indices = rng.choice(len(candidates), size=trials, replace=False)
        centers = [np.asarray(candidates[index], dtype=int) for index in np.atleast_1d(indices)]
    return centers, depth


def _overlap_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    pred_size = int(np.count_nonzero(prediction))
    gt_size = int(np.count_nonzero(ground_truth))
    denominator = pred_size + gt_size
    dice = 2.0 * intersection / denominator if denominator else 1.0
    union = pred_size + gt_size - intersection
    iou = intersection / union if union else 1.0
    return float(dice), float(iou)


def _mask_sha256(mask: np.ndarray) -> str:
    """Return a stable digest for comparing voxel-level partitions."""
    return hashlib.sha256(np.packbits(mask, axis=None).tobytes()).hexdigest()


def _affected_pq(source_iou: float, sink_iou: float, threshold: float) -> tuple[int, int, int, float]:
    accepted = [iou for iou in (source_iou, sink_iou) if iou >= threshold]
    true_positives = len(accepted)
    false_positives = 2 - true_positives
    false_negatives = 2 - true_positives
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    pq = float(sum(accepted) / denominator) if denominator else 0.0
    return true_positives, false_positives, false_negatives, pq


def _match_binary_instance_pair(
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    source_gt: np.ndarray,
    sink_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Match two output instances to two GT instances without relying on label IDs."""
    first_source = _overlap_metrics(first_prediction, source_gt)
    second_sink = _overlap_metrics(second_prediction, sink_gt)
    second_source = _overlap_metrics(second_prediction, source_gt)
    first_sink = _overlap_metrics(first_prediction, sink_gt)
    direct_score = first_source[1] + second_sink[1]
    swapped_score = second_source[1] + first_sink[1]
    if swapped_score > direct_score:
        source_prediction = second_prediction
        sink_prediction = first_prediction
        source_dice, source_iou = second_source
        sink_dice, sink_iou = first_sink
        orientation = "swapped"
    else:
        source_prediction = first_prediction
        sink_prediction = second_prediction
        source_dice, source_iou = first_source
        sink_dice, sink_iou = second_sink
        orientation = "direct"
    return source_prediction, sink_prediction, {
        "matched_orientation": orientation,
        "source_dice": source_dice,
        "source_iou": source_iou,
        "sink_dice": sink_dice,
        "sink_iou": sink_iou,
    }


def _core_recognition_metrics(
    source_prediction: np.ndarray,
    sink_prediction: np.ndarray,
    source_evaluation_core: np.ndarray,
    sink_evaluation_core: np.ndarray,
    recall_threshold: float,
) -> dict[str, Any]:
    source_core_size = int(source_evaluation_core.sum())
    sink_core_size = int(sink_evaluation_core.sum())
    if source_core_size == 0 or sink_core_size == 0:
        raise ValueError("both evaluation fragments must contain predicted Label-2 core voxels")

    source_correct = int(np.count_nonzero(source_prediction & source_evaluation_core))
    sink_correct = int(np.count_nonzero(sink_prediction & sink_evaluation_core))
    source_recall = source_correct / source_core_size
    sink_recall = sink_correct / sink_core_size
    total_core = source_core_size + sink_core_size
    source_components = int(nd_label(source_prediction)[1])
    sink_components = int(nd_label(sink_prediction)[1])
    core_success = bool(
        source_recall >= recall_threshold
        and sink_recall >= recall_threshold
        and source_components == 1
        and sink_components == 1
    )
    return {
        "source_core_recall": float(source_recall),
        "sink_core_recall": float(sink_recall),
        "core_assignment_accuracy": float((source_correct + sink_correct) / total_core),
        "source_connected_components": source_components,
        "sink_connected_components": sink_components,
        "core_recognition_success": core_success,
        "fragment_recognition_success": core_success,
    }


def _merged_baseline(
    merged_mask: np.ndarray,
    source_gt: np.ndarray,
    sink_gt: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    source_dice, source_iou = _overlap_metrics(merged_mask, source_gt)
    sink_dice, sink_iou = _overlap_metrics(merged_mask, sink_gt)
    best_iou = max(source_iou, sink_iou)
    true_positives = int(best_iou >= threshold)
    false_positives = 1 - true_positives
    false_negatives = 2 - true_positives
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    pq = best_iou / denominator if true_positives else 0.0
    return {
        "predicted_instances": 1,
        "ground_truth_instances": 2,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "source_best_dice": source_dice,
        "source_best_iou": source_iou,
        "sink_best_dice": sink_dice,
        "sink_best_iou": sink_iou,
        "panoptic_quality": float(pq),
        "instance_segmentation_success": False,
        "fragment_recognition_success": False,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _summarize(records: list[dict[str, Any]], radii: list[float]) -> dict[str, Any]:
    per_radius: dict[str, Any] = {}
    for radius in radii:
        selected = [record for record in records if record["radius_mm"] == radius]
        successes = sum(bool(record["primary_success"]) for record in selected)
        instance_successes = sum(
            bool(record["instance_segmentation_success"]) for record in selected
        )
        core_successes = sum(bool(record["core_recognition_success"]) for record in selected)
        valid = [record for record in selected if record["status"] == "ok"]
        low, high = _wilson_interval(successes, len(selected))
        per_radius[str(radius)] = {
            "trials": len(selected),
            "valid_splits": len(valid),
            "primary_successes": successes,
            "success_rate": successes / len(selected) if selected else 0.0,
            "success_rate_wilson_95": [low, high],
            "instance_segmentation_successes": instance_successes,
            "instance_segmentation_success_rate": (
                instance_successes / len(selected) if selected else 0.0
            ),
            "core_recognition_successes": core_successes,
            "core_recognition_success_rate": (
                core_successes / len(selected) if selected else 0.0
            ),
            "unique_source_centers": len({record["source_center_zyx"] for record in selected}),
            "unique_sink_centers": len({record["sink_center_zyx"] for record in selected}),
            "unique_partitions_valid": len(
                {record["partition_sha256"] for record in valid if record["partition_sha256"]}
            ),
            "mean_pq_all_trials": float(np.mean([record["panoptic_quality"] for record in selected])),
            "mean_source_seed_gt_purity": float(
                np.mean([record["source_seed_gt_purity"] for record in selected])
            ),
            "mean_sink_seed_gt_purity": float(
                np.mean([record["sink_seed_gt_purity"] for record in selected])
            ),
            "source_center_abbc_label_counts": dict(
                Counter(str(record["source_center_abbc_label"]) for record in selected)
            ),
            "sink_center_abbc_label_counts": dict(
                Counter(str(record["sink_center_abbc_label"]) for record in selected)
            ),
            "mean_source_core_recall_valid": (
                float(np.mean([record["source_core_recall"] for record in valid])) if valid else 0.0
            ),
            "mean_sink_core_recall_valid": (
                float(np.mean([record["sink_core_recall"] for record in valid])) if valid else 0.0
            ),
            "mean_core_assignment_accuracy_valid": (
                float(np.mean([record["core_assignment_accuracy"] for record in valid]))
                if valid
                else 0.0
            ),
            "mean_source_dice_valid": (
                float(np.mean([record["source_dice"] for record in valid])) if valid else 0.0
            ),
            "mean_sink_dice_valid": (
                float(np.mean([record["sink_dice"] for record in valid])) if valid else 0.0
            ),
            "mean_cut_prob3_valid": (
                float(np.mean([record["mean_prob_label_3_on_cut"] for record in valid]))
                if valid
                else 0.0
            ),
            "mean_runtime_seconds": (
                float(np.mean([record["runtime_seconds"] for record in selected]))
                if selected
                else 0.0
            ),
            "max_runtime_seconds": (
                float(np.max([record["runtime_seconds"] for record in selected]))
                if selected
                else 0.0
            ),
            "failure_reasons": dict(Counter(record["status"] for record in selected if record["status"] != "ok")),
        }
    return per_radius


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_summary(
    path: Path,
    records: list[dict[str, Any]],
    radii: list[float],
    threshold: float,
    primary_endpoint: str,
    split_mode: str,
    case_id: str,
    baseline_pq: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = [f"{radius:g}" for radius in radii]
    success_rates = []
    lower_errors = []
    upper_errors = []
    for radius in radii:
        selected = [record for record in records if record["radius_mm"] == radius]
        successes = sum(bool(record["primary_success"]) for record in selected)
        rate = successes / len(selected)
        low, high = _wilson_interval(successes, len(selected))
        success_rates.append(rate)
        lower_errors.append(rate - low)
        upper_errors.append(high - rate)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(labels, success_rates, color="#2878b5")
    axes[0].errorbar(
        np.arange(len(labels)),
        success_rates,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#202020",
        capsize=4,
    )
    axes[0].set_ylim(0.0, 1.05)
    if primary_endpoint == "instance-iou":
        endpoint_title = f"Instance separation\n(Both matched IoU >= {threshold:g})"
    else:
        endpoint_title = f"Core recognition\n(Both core recall >= {threshold:g})"
    axes[0].set_title(endpoint_title)
    axes[0].set_ylabel("Success rate")
    axes[0].set_xlabel("Seed radius (mm)")

    positions = np.arange(len(radii)) * 3.0
    source_values = [
        [record["source_dice"] for record in records if record["radius_mm"] == radius and record["status"] == "ok"]
        for radius in radii
    ]
    sink_values = [
        [record["sink_dice"] for record in records if record["radius_mm"] == radius and record["status"] == "ok"]
        for radius in radii
    ]
    source_boxes = axes[1].boxplot(source_values, positions=positions - 0.45, widths=0.75, patch_artist=True)
    sink_boxes = axes[1].boxplot(sink_values, positions=positions + 0.45, widths=0.75, patch_artist=True)
    for box in source_boxes["boxes"]:
        box.set_facecolor("#1595a3")
    for box in sink_boxes["boxes"]:
        box.set_facecolor("#e28f2d")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Instance Dice by interaction size")
    axes[1].set_xlabel("Seed radius (mm)")
    axes[1].set_ylabel("Dice")
    axes[1].legend(
        handles=[
            Patch(facecolor="#1595a3", label="Source fragment"),
            Patch(facecolor="#e28f2d", label="Sink fragment"),
        ],
        loc="lower right",
    )

    pair_pq = [
        [
            record["panoptic_quality"]
            for record in records
            if record["radius_mm"] == radius
        ]
        for radius in radii
    ]
    support_boxes = axes[2].boxplot(pair_pq, tick_labels=labels, patch_artist=True)
    for box in support_boxes["boxes"]:
        box.set_facecolor("#c33c54")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].axhline(baseline_pq, color="#303030", linestyle="--", label="Input baseline")
    axes[2].set_title("Affected-pair panoptic quality")
    axes[2].set_xlabel("Seed radius (mm)")
    axes[2].set_ylabel("PQ")
    axes[2].legend(loc="lower right")

    mode_title = "Core-first" if split_mode == "core-first" else "Full-volume min-cut"
    case_title = case_id.replace("_", " ").title()
    figure.suptitle(f"{case_title} {mode_title} interaction protocol", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    natural_input = args.instance_id is not None
    pair_values = (args.source_instance_id, args.sink_instance_id)
    if natural_input and any(value is not None for value in pair_values):
        raise SystemExit("use --instance-id alone for a natural under-split input")
    if not natural_input and not all(value is not None for value in pair_values):
        raise SystemExit(
            "counterfactual mode requires --source-instance-id and --sink-instance-id"
        )
    if not natural_input and args.source_instance_id == args.sink_instance_id:
        raise SystemExit("source and sink instance IDs must differ")
    if args.trials_per_radius < 1:
        raise SystemExit("trials-per-radius must be positive")
    if any(radius <= 0 for radius in args.seed_radii_mm):
        raise SystemExit("seed radii must be positive")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise SystemExit("iou-threshold must be between 0 and 1")
    if not 0.0 <= args.core_recall_threshold <= 1.0:
        raise SystemExit("core-recall-threshold must be between 0 and 1")
    if args.center_abbc_labels and any(
        label_id not in (0, 1, 2, 3) for label_id in args.center_abbc_labels
    ):
        raise SystemExit("center-abbc-labels must contain only 0, 1, 2, or 3")

    paths = {
        "abbc_pred": Path(args.abbc_pred),
        "instances": Path(args.instances),
        "source_gt": Path(args.source_gt),
        "sink_gt": Path(args.sink_gt),
    }
    if args.prob_label_3:
        paths["prob_label_3"] = Path(args.prob_label_3)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")

    grid = _assert_aligned(paths["abbc_pred"], [path for key, path in paths.items() if key != "abbc_pred"])
    selected_instance_ids = (
        [args.instance_id]
        if natural_input
        else [args.source_instance_id, args.sink_instance_id]
    )
    bboxes = [
        *(_label_bbox(paths["instances"], instance_id) for instance_id in selected_instance_ids),
        _label_bbox(paths["source_gt"], None),
        _label_bbox(paths["sink_gt"], None),
    ]
    roi_index_xyz, roi_size_xyz = _union_roi(
        bboxes,
        grid["size"],
        grid["spacing"],
        args.roi_padding_mm,
    )

    abbc = _read_roi(paths["abbc_pred"], roi_index_xyz, roi_size_xyz)
    instances = _read_roi(paths["instances"], roi_index_xyz, roi_size_xyz)
    source_gt = _read_roi(paths["source_gt"], roi_index_xyz, roi_size_xyz) > 0
    sink_gt = _read_roi(paths["sink_gt"], roi_index_xyz, roi_size_xyz) > 0
    prob_label_3 = (
        _read_roi(paths["prob_label_3"], roi_index_xyz, roi_size_xyz).astype(np.float32)
        if "prob_label_3" in paths
        else None
    )

    if natural_input:
        merged_mask = instances == args.instance_id
        source_original = merged_mask
        sink_original = merged_mask
    else:
        source_original = instances == args.source_instance_id
        sink_original = instances == args.sink_instance_id
        merged_mask = source_original | sink_original
    _, connected_components = nd_label(merged_mask)
    if connected_components != 1:
        raise SystemExit(
            f"selected input has {connected_components} connected components; this would be a trivial split"
        )

    source_allowed = source_gt & source_original
    sink_allowed = sink_gt & sink_original
    source_interaction_domain = source_original.copy()
    sink_interaction_domain = sink_original.copy()
    predicted_core = (abbc == 2) & merged_mask
    if args.seed_region == "core":
        source_allowed &= predicted_core
        sink_allowed &= predicted_core
        source_interaction_domain &= predicted_core
        sink_interaction_domain &= predicted_core
    if args.center_abbc_labels:
        center_label_mask = np.isin(abbc, args.center_abbc_labels)
        source_allowed &= center_label_mask
        sink_allowed &= center_label_mask
    if not np.any(source_allowed) or not np.any(sink_allowed):
        raise SystemExit("GT/prediction intersection is empty for at least one seed side")
    source_evaluation_core = source_gt & merged_mask & (abbc == 2)
    sink_evaluation_core = sink_gt & merged_mask & (abbc == 2)
    if not np.any(source_evaluation_core) or not np.any(sink_evaluation_core):
        raise SystemExit("at least one GT fragment has no predicted Label-2 core in the merged pair")

    cfg = SplitConfig(
        w_prob3=args.w_prob3,
        w_prob1=args.w_prob1,
        w_prob2=args.w_prob2,
        w_min=args.w_min,
        min_split_piece_size=args.min_split_piece_size,
        core_anchor_erosion_mm=args.core_anchor_erosion_mm,
        core_anchor_min_voxels=args.core_anchor_min_voxels,
        core_anchor_distinct_max_distance_mm=args.core_anchor_distinct_max_distance_mm,
        core_first_max_graph_voxels=args.core_first_max_graph_voxels,
    )
    cost = (
        build_cost_field(abbc, None, merged_mask, cfg, prob_label_3=prob_label_3)
        if args.split_mode == "full-mincut"
        else None
    )
    p3 = (
        np.clip(prob_label_3, 0.0, 1.0)
        if prob_label_3 is not None
        else (abbc == 3).astype(np.float32)
    )
    spacing_zyx = np.asarray(grid["spacing"][::-1], dtype=float)
    distance_to_predicted_core = distance_transform_edt(
        ~predicted_core,
        sampling=spacing_zyx,
    )
    rng = np.random.default_rng(args.random_seed)
    records: list[dict[str, Any]] = []
    radii = [float(radius) for radius in args.seed_radii_mm]
    representative_radius = 2.0 if 2.0 in radii else radii[0]
    representative: dict[str, np.ndarray] | None = None

    for radius in args.seed_radii_mm:
        source_centers, source_depth = _sample_centers(
            source_allowed,
            spacing_zyx,
            radius,
            args.seed_clearance_mm,
            args.trials_per_radius,
            rng,
            require_contained_seed=args.require_contained_interaction,
            include_deepest=args.sampling_strategy == "deepest-plus-random",
        )
        sink_centers, sink_depth = _sample_centers(
            sink_allowed,
            spacing_zyx,
            radius,
            args.seed_clearance_mm,
            args.trials_per_radius,
            rng,
            require_contained_seed=args.require_contained_interaction,
            include_deepest=args.sampling_strategy == "deepest-plus-random",
        )

        for trial, (source_center, sink_center) in enumerate(zip(source_centers, sink_centers)):
            source_seed_mask = _sphere_seed(
                source_interaction_domain,
                source_center,
                spacing_zyx,
                radius,
            )
            sink_seed_mask = _sphere_seed(
                sink_interaction_domain,
                sink_center,
                spacing_zyx,
                radius,
            )
            sampling = (
                "deepest"
                if args.sampling_strategy == "deepest-plus-random" and trial == 0
                else "random_interior"
            )
            global_source_zyx = source_center + np.asarray(roi_index_xyz[::-1])
            global_sink_zyx = sink_center + np.asarray(roi_index_xyz[::-1])
            source_seed_voxels = int(source_seed_mask.sum())
            sink_seed_voxels = int(sink_seed_mask.sum())
            record: dict[str, Any] = {
                "radius_mm": float(radius),
                "trial": trial,
                "sampling": sampling,
                "status": "ok",
                "source_center_zyx": ";".join(str(int(v)) for v in global_source_zyx),
                "sink_center_zyx": ";".join(str(int(v)) for v in global_sink_zyx),
                "source_center_depth_mm": float(source_depth[tuple(source_center)]),
                "sink_center_depth_mm": float(sink_depth[tuple(sink_center)]),
                "source_center_abbc_label": int(abbc[tuple(source_center)]),
                "sink_center_abbc_label": int(abbc[tuple(sink_center)]),
                "source_center_distance_to_core_mm": float(
                    distance_to_predicted_core[tuple(source_center)]
                ),
                "sink_center_distance_to_core_mm": float(
                    distance_to_predicted_core[tuple(sink_center)]
                ),
                "source_seed_voxels": source_seed_voxels,
                "sink_seed_voxels": sink_seed_voxels,
                "source_seed_gt_purity": float(
                    np.count_nonzero(source_seed_mask & source_gt) / source_seed_voxels
                ),
                "sink_seed_gt_purity": float(
                    np.count_nonzero(sink_seed_mask & sink_gt) / sink_seed_voxels
                ),
                "source_seed_other_gt_voxels": int(
                    np.count_nonzero(source_seed_mask & sink_gt)
                ),
                "sink_seed_other_gt_voxels": int(
                    np.count_nonzero(sink_seed_mask & source_gt)
                ),
                "source_piece_voxels": 0,
                "sink_piece_voxels": 0,
                "matched_orientation": "",
                "source_dice": 0.0,
                "source_iou": 0.0,
                "sink_dice": 0.0,
                "sink_iou": 0.0,
                "true_positives": 0,
                "false_positives": 2,
                "false_negatives": 2,
                "both_iou_at_threshold": False,
                "source_core_recall": 0.0,
                "sink_core_recall": 0.0,
                "core_assignment_accuracy": 0.0,
                "source_connected_components": 0,
                "sink_connected_components": 0,
                "instance_segmentation_success": False,
                "core_recognition_success": False,
                "fragment_recognition_success": False,
                "primary_success": False,
                "panoptic_quality": 0.0,
                "cut_voxels": 0,
                "mean_prob_label_3_on_cut": 0.0,
                "fraction_cut_above_prob3_threshold": 0.0,
                "partition_sha256": "",
                "core_partition_method": "",
                "peripheral_partition_method": "",
                "source_core_anchor_voxels": 0,
                "sink_core_anchor_voxels": 0,
                "source_projection_distance_mm": 0.0,
                "sink_projection_distance_mm": 0.0,
                "core_source": "",
                "distinct_component_reassignment": "",
                "maxflow": 0.0,
                "runtime_seconds": 0.0,
                "error_message": "",
            }

            start = time.perf_counter()
            if np.any(source_seed_mask & sink_seed_mask):
                record["status"] = "overlapping_interactions"
                record["runtime_seconds"] = time.perf_counter() - start
                records.append(record)
                continue
            try:
                if args.split_mode == "core-first":
                    result, core_diagnostics = core_first_partition(
                        abbc_pred=abbc,
                        instance_mask=merged_mask,
                        source_interaction_mask=source_seed_mask,
                        sink_interaction_mask=sink_seed_mask,
                        prob_label_3=prob_label_3,
                        spacing_zyx=spacing_zyx,
                        cfg=cfg,
                    )
                    for key in (
                        "core_partition_method",
                        "peripheral_partition_method",
                        "source_core_anchor_voxels",
                        "sink_core_anchor_voxels",
                        "source_projection_distance_mm",
                        "sink_projection_distance_mm",
                        "core_source",
                        "distinct_component_reassignment",
                    ):
                        value = core_diagnostics.get(key)
                        record[key] = "" if value is None else value
                else:
                    result = min_cut_partition(
                        merged_mask,
                        Seed(source_seed_mask, source_center.astype(float), "manual"),
                        Seed(sink_seed_mask, sink_center.astype(float), "manual"),
                        cost,
                    )
            except (ValueError, RuntimeError) as error:
                record["status"] = "split_error"
                record["error_message"] = str(error)
                record["runtime_seconds"] = time.perf_counter() - start
                records.append(record)
                continue
            record["runtime_seconds"] = time.perf_counter() - start
            if result is None:
                record["status"] = "no_cut"
                records.append(record)
                continue

            source_size = int(result.source_mask.sum())
            sink_size = int(result.sink_mask.sum())
            record["source_piece_voxels"] = source_size
            record["sink_piece_voxels"] = sink_size
            record["maxflow"] = float(result.flow)
            if min(source_size, sink_size) < cfg.min_split_piece_size:
                record["status"] = "piece_too_small"
                records.append(record)
                continue

            source_prediction, sink_prediction, pair_metrics = _match_binary_instance_pair(
                result.source_mask,
                result.sink_mask,
                source_gt,
                sink_gt,
            )
            source_iou = pair_metrics["source_iou"]
            sink_iou = pair_metrics["sink_iou"]
            true_positives, false_positives, false_negatives, pq = _affected_pq(
                source_iou,
                sink_iou,
                args.iou_threshold,
            )
            core_metrics = _core_recognition_metrics(
                source_prediction,
                sink_prediction,
                source_evaluation_core,
                sink_evaluation_core,
                args.core_recall_threshold,
            )
            instance_success = true_positives == 2
            primary_success = (
                instance_success
                if args.primary_endpoint == "instance-iou"
                else core_metrics["core_recognition_success"]
            )
            cut_prob = p3[result.cut_mask]
            record.update(
                {
                    **pair_metrics,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "both_iou_at_threshold": instance_success,
                    "instance_segmentation_success": instance_success,
                    **core_metrics,
                    "primary_success": primary_success,
                    "panoptic_quality": pq,
                    "cut_voxels": int(result.cut_mask.sum()),
                    "mean_prob_label_3_on_cut": float(cut_prob.mean()) if cut_prob.size else 0.0,
                    "fraction_cut_above_prob3_threshold": (
                        float(np.mean(cut_prob > cfg.prob3_seed_threshold)) if cut_prob.size else 0.0
                    ),
                    "partition_sha256": _mask_sha256(source_prediction),
                }
            )
            if representative is None and trial == 0 and float(radius) == representative_radius:
                partition = np.zeros(merged_mask.shape, dtype=np.uint8)
                partition[source_prediction] = 1
                partition[sink_prediction] = 2
                core_assignment = np.zeros(merged_mask.shape, dtype=np.uint8)
                core_assignment[(abbc == 2) & source_prediction] = 1
                core_assignment[(abbc == 2) & sink_prediction] = 2
                interactions = np.zeros(merged_mask.shape, dtype=np.uint8)
                interactions[source_seed_mask] = 1
                interactions[sink_seed_mask] = 2
                final_instances = instances.copy()
                final_instances[merged_mask] = 0
                source_output_id = (
                    int(args.instance_id) if natural_input else int(args.source_instance_id)
                )
                sink_output_id = (
                    int(instances.max()) + 1
                    if natural_input
                    else int(args.sink_instance_id)
                )
                final_instances[source_prediction] = source_output_id
                final_instances[sink_prediction] = sink_output_id
                representative = {
                    "representative_partition.nii.gz": partition,
                    "representative_instances.nii.gz": final_instances,
                    "representative_core_assignment.nii.gz": core_assignment,
                    "representative_interactions.nii.gz": interactions,
                    "representative_cut.nii.gz": result.cut_mask.astype(np.uint8),
                }
            records.append(record)

    per_radius = _summarize(records, radii)
    input_baseline = _merged_baseline(
        merged_mask,
        source_gt,
        sink_gt,
        args.iou_threshold,
    )
    acceptance_targets = {"2.0": 0.8, "4.0": 0.9}
    acceptance = {
        radius: {
            "required_success_rate": required,
            "observed_success_rate": per_radius[radius]["success_rate"],
            "passed": per_radius[radius]["success_rate"] >= required,
        }
        for radius, required in acceptance_targets.items()
        if radius in per_radius
    }

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trials.csv", records)
    _plot_summary(
        output_dir / "summary.png",
        records,
        radii,
        (
            args.iou_threshold
            if args.primary_endpoint == "instance-iou"
            else args.core_recall_threshold
        ),
        args.primary_endpoint,
        args.split_mode,
        args.case_id,
        input_baseline["panoptic_quality"],
    )
    if representative is None:
        raise RuntimeError("no valid representative partition was generated")
    for filename, array in representative.items():
        _write_roi_image(output_dir / filename, array, grid, roi_index_xyz)

    protocol = {
        "case_id": args.case_id,
        "protocol": (
            "oracle localized interactions on a naturally under-split predicted instance"
            if natural_input
            else "oracle localized interactions on a counterfactually merged predicted pair"
        ),
        "input_mode": "natural_under_split" if natural_input else "counterfactual_merge",
        "gt_usage": "seed placement and evaluation only; never graph cost",
        "interaction_support": (
            "predicted Label-2 core" if args.seed_region == "core" else "predicted instance"
        ),
        "inputs": {key: str(path) for key, path in paths.items()},
        "source_instance_id": args.source_instance_id,
        "sink_instance_id": args.sink_instance_id,
        "instance_id": args.instance_id,
        "roi_index_xyz": roi_index_xyz,
        "roi_size_xyz": roi_size_xyz,
        "spacing_xyz_mm": grid["spacing"],
        "seed_radii_mm": radii,
        "trials_per_radius": args.trials_per_radius,
        "seed_clearance_mm": args.seed_clearance_mm,
        "random_seed": args.random_seed,
        "sampling_strategy": args.sampling_strategy,
        "require_contained_interaction": args.require_contained_interaction,
        "center_abbc_labels": args.center_abbc_labels,
        "representative_radius_mm": representative_radius,
        "split_mode": args.split_mode,
        "seed_region": args.seed_region,
        "primary_endpoint": args.primary_endpoint,
        "primary_success_definition": (
            f"both affected output instances match distinct GT instances at IoU >= {args.iou_threshold:g}"
            if args.primary_endpoint == "instance-iou"
            else (
                "both GT-associated predicted cores have recall >= "
                f"{args.core_recall_threshold:g} in two connected output instances"
            )
        ),
        "secondary_core_definition": (
            "both GT-associated predicted cores have recall >= "
            f"{args.core_recall_threshold:g} in two connected output instances"
        ),
        "cost_source": "soft predicted label-3 probability" if prob_label_3 is not None else "hard ABBC labels",
        "split_config": {
            "w_prob3": cfg.w_prob3,
            "w_prob1": cfg.w_prob1,
            "w_prob2": cfg.w_prob2,
            "w_min": cfg.w_min,
            "min_split_piece_size": cfg.min_split_piece_size,
            "core_anchor_erosion_mm": cfg.core_anchor_erosion_mm,
            "core_anchor_min_voxels": cfg.core_anchor_min_voxels,
            "core_anchor_distinct_max_distance_mm": (
                cfg.core_anchor_distinct_max_distance_mm
            ),
            "core_first_max_graph_voxels": cfg.core_first_max_graph_voxels,
        },
    }
    summary = {
        "protocol": protocol,
        "input_baseline": input_baseline,
        "per_radius": per_radius,
        "acceptance": acceptance,
        "acceptance_passed": all(item["passed"] for item in acceptance.values()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {output_dir / 'trials.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.png'}")
    for radius in radii:
        item = per_radius[str(radius)]
        print(
            f"radius={radius:g} mm: success={item['primary_successes']}/{item['trials']} "
            f"({item['success_rate']:.1%}), valid={item['valid_splits']}"
        )


if __name__ == "__main__":
    main()
