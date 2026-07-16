#!/usr/bin/env python3
"""Run prompt-driven splits and render compact 2D candidate visualizations."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    SplitConfig,
    core_first_partition,
)
from tools.PENGWIN.postprocess.find_interactive_split_candidates import (
    _bbox_slices,
    _hard_label_watershed,
)


COLORS = {
    "merged": "#f2be22",
    "source_gt": "#1597c5",
    "sink_gt": "#d92f67",
    "source_prompt": "#22c55e",
    "sink_prompt": "#e11dca",
    "source_output": "#1597c5",
    "sink_output": "#ef7d18",
    "boundary": "#ffffff",
}

INSTANCE_COLORS = (
    "#4e79a7",
    "#59a14f",
    "#af7aa1",
    "#e15759",
    "#76b7b2",
    "#9c755f",
    "#bab0ab",
    "#b07aa1",
    "#499894",
    "#79706e",
    "#86bcb6",
    "#8cd17d",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abbc-dir", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="CASE:PREDICTED_INSTANCE:FIRST_GT_LABEL:SECOND_GT_LABEL",
    )
    parser.add_argument("--prompt-radius-mm", type=float, default=2.0)
    parser.add_argument("--crop-padding-voxels", type=int, default=2)
    parser.add_argument("--min-core-size", type=int, default=10)
    parser.add_argument("--core-erosion-mm", type=float, default=2.0)
    parser.add_argument("--core-component-min-voxels", type=int, default=20)
    parser.add_argument("--core-distinct-max-distance-mm", type=float, default=15.0)
    parser.add_argument("--max-graph-voxels", type=int, default=5_000_000)
    parser.add_argument("--ct-window", type=float, nargs=2, default=(-200.0, 1500.0))
    parser.add_argument(
        "--slice-gallery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write separate axial, coronal, and sagittal review images",
    )
    parser.add_argument("--gallery-dpi", type=int, default=200)
    parser.add_argument("--gallery-padding-voxels", type=int, default=12)
    parser.add_argument(
        "--large-plane-boards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one large source/cut/sink comparison board per anatomical plane",
    )
    parser.add_argument("--board-dpi", type=int, default=250)
    parser.add_argument(
        "--write-volumes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write partition, prompt, and before/after NIfTI volumes",
    )
    return parser


def _parse_case_spec(value: str) -> tuple[str, int, int, int]:
    fields = value.split(":")
    if len(fields) != 4:
        raise ValueError(f"invalid case specification: {value}")
    return fields[0], int(fields[1]), int(fields[2]), int(fields[3])


def _deepest_center(mask: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        raise ValueError("prompt domain is empty")
    depth = distance_transform_edt(mask, sampling=spacing_zyx)
    return np.asarray(np.unravel_index(int(np.argmax(depth)), depth.shape), dtype=int)


def _sphere_prompt(
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
    prompt = np.zeros_like(allowed_mask, dtype=bool)
    prompt[crop] = (squared_distance <= radius_mm**2) & allowed_mask[crop]
    return prompt


def _overlap_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    predicted_size = int(prediction.sum())
    gt_size = int(ground_truth.sum())
    dice = 2.0 * intersection / (predicted_size + gt_size)
    union = predicted_size + gt_size - intersection
    iou = intersection / union if union else 0.0
    return float(dice), float(iou)


def _match_outputs(
    first: np.ndarray,
    second: np.ndarray,
    source_gt: np.ndarray,
    sink_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    first_source = _overlap_metrics(first, source_gt)
    second_sink = _overlap_metrics(second, sink_gt)
    second_source = _overlap_metrics(second, source_gt)
    first_sink = _overlap_metrics(first, sink_gt)
    if second_source[1] + first_sink[1] > first_source[1] + second_sink[1]:
        source_output, sink_output = second, first
        source_metrics, sink_metrics = second_source, first_sink
        orientation = "swapped"
    else:
        source_output, sink_output = first, second
        source_metrics, sink_metrics = first_source, second_sink
        orientation = "direct"
    return source_output, sink_output, {
        "matched_orientation": orientation,
        "source_dice": source_metrics[0],
        "source_iou": source_metrics[1],
        "sink_dice": sink_metrics[0],
        "sink_iou": sink_metrics[1],
    }


def _apply_split_to_instances(
    instances: np.ndarray,
    predicted_id: int,
    source_output: np.ndarray,
    sink_output: np.ndarray,
) -> tuple[np.ndarray, int]:
    if not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("instances must use an integer dtype")
    if source_output.shape != instances.shape or sink_output.shape != instances.shape:
        raise ValueError("split masks must have the same shape as instances")
    source_output = np.asarray(source_output, dtype=bool)
    sink_output = np.asarray(sink_output, dtype=bool)
    if np.any(source_output & sink_output):
        raise ValueError("source and sink outputs must not overlap")
    selected = instances == int(predicted_id)
    if not np.array_equal(source_output | sink_output, selected):
        raise ValueError("split outputs must exactly partition the selected instance")

    new_instance_id = int(instances.max()) + 1
    output_dtype = instances.dtype
    if new_instance_id > np.iinfo(output_dtype).max:
        output_dtype = np.dtype(np.uint32)
    instances_after = instances.astype(output_dtype, copy=True)
    instances_after[source_output] = int(predicted_id)
    instances_after[sink_output] = new_instance_id
    return instances_after, new_instance_id


def _overlay(ax: plt.Axes, mask: np.ndarray, color: str, alpha: float) -> None:
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = to_rgba(color, alpha)
    ax.imshow(overlay, origin="lower", interpolation="nearest")


def _instance_color(instance_id: int) -> str:
    return INSTANCE_COLORS[(int(instance_id) - 1) % len(INSTANCE_COLORS)]


def _all_boundaries_2d(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    vertical = (labels[:-1] != labels[1:]) & ((labels[:-1] > 0) | (labels[1:] > 0))
    horizontal = (labels[:, :-1] != labels[:, 1:]) & (
        (labels[:, :-1] > 0) | (labels[:, 1:] > 0)
    )
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    return boundary


def _overlay_instances(
    ax: plt.Axes,
    labels: np.ndarray,
    alpha: float,
    color_overrides: dict[int, str] | None = None,
) -> None:
    color_overrides = color_overrides or {}
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    for instance_id in np.unique(labels):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        color = color_overrides.get(instance_id, _instance_color(instance_id))
        overlay[labels == instance_id] = to_rgba(color, alpha)
    ax.imshow(overlay, origin="lower", interpolation="nearest")
    _overlay(ax, _all_boundaries_2d(labels), COLORS["boundary"], 0.72)


def _annotate_instance_ids(
    ax: plt.Axes,
    labels: np.ndarray,
    prefix: str,
    minimum_slice_voxels: int = 30,
) -> None:
    for instance_id in np.unique(labels):
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        mask = labels == instance_id
        if int(mask.sum()) < minimum_slice_voxels:
            continue
        depth = distance_transform_edt(mask)
        y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        text = ax.text(
            int(x),
            int(y),
            f"{prefix}{instance_id}",
            color="white",
            fontsize=6.5,
            ha="center",
            va="center",
            weight="bold",
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground="#202020")])


def _boundary_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(first, dtype=bool)
    boundary[:-1] |= first[:-1] & second[1:]
    boundary[1:] |= first[1:] & second[:-1]
    boundary[:, :-1] |= first[:, :-1] & second[:, 1:]
    boundary[:, 1:] |= first[:, 1:] & second[:, :-1]
    return boundary


def _plot_case(
    path: Path,
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    ct: np.ndarray,
    instances_before: np.ndarray,
    gt_instances: np.ndarray,
    instances_after: np.ndarray,
    source_prompt: np.ndarray,
    sink_prompt: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_output: np.ndarray,
    sink_output: np.ndarray,
    new_instance_id: int,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    ct_window: tuple[float, float],
    spacing_zyx: np.ndarray,
) -> None:
    rows = (
        ("Source-prompt axial", int(source_center[0]), source_center, COLORS["source_prompt"]),
        ("Sink-prompt axial", int(sink_center[0]), sink_center, COLORS["sink_prompt"]),
    )
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 8.8), facecolor="white")
    figure.subplots_adjust(
        left=0.045,
        right=0.99,
        top=0.82,
        bottom=0.16,
        wspace=0.025,
        hspace=0.07,
    )
    column_titles = (
        "All predicted instances (before)",
        "All fragment instance GT",
        "Selected instance and prompts",
        "All predicted instances (after)",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11)

    for row, (row_name, z_index, row_center, point_color) in enumerate(rows):
        ct_slice = ct[z_index]
        instances_before_slice = instances_before[z_index]
        gt_slice = gt_instances[z_index]
        instances_after_slice = instances_after[z_index]
        source_prompt_slice = source_prompt[z_index]
        sink_prompt_slice = sink_prompt[z_index]
        source_output_slice = source_output[z_index]
        sink_output_slice = sink_output[z_index]
        for axis in axes[row]:
            axis.imshow(
                ct_slice,
                cmap="gray",
                vmin=ct_window[0],
                vmax=ct_window[1],
                origin="lower",
                interpolation="nearest",
            )
            axis.set_aspect(float(spacing_zyx[1] / spacing_zyx[2]))
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(f"{row_name}\nz = {z_index}", fontsize=10)
        _overlay_instances(
            axes[row, 0],
            instances_before_slice,
            0.50,
            {predicted_id: COLORS["merged"]},
        )
        _annotate_instance_ids(axes[row, 0], instances_before_slice, "P")
        _overlay_instances(
            axes[row, 1],
            gt_slice,
            0.52,
            {
                source_label: COLORS["source_gt"],
                sink_label: COLORS["sink_gt"],
            },
        )
        _annotate_instance_ids(axes[row, 1], gt_slice, "G")
        _overlay_instances(
            axes[row, 2],
            instances_before_slice,
            0.22,
            {predicted_id: COLORS["merged"]},
        )
        _overlay(axes[row, 2], source_prompt_slice, COLORS["source_prompt"], 0.8)
        _overlay(axes[row, 2], sink_prompt_slice, COLORS["sink_prompt"], 0.8)
        axes[row, 2].scatter(
            int(row_center[2]),
            int(row_center[1]),
            s=72,
            facecolors="none",
            edgecolors=point_color,
            linewidths=2.2,
            marker="o",
        )
        axes[row, 2].scatter(
            int(row_center[2]),
            int(row_center[1]),
            s=22,
            color=point_color,
            marker="x",
            linewidths=1.8,
        )
        _overlay_instances(
            axes[row, 3],
            instances_after_slice,
            0.50,
            {
                predicted_id: COLORS["source_output"],
                new_instance_id: COLORS["sink_output"],
            },
        )
        _annotate_instance_ids(axes[row, 3], instances_after_slice, "P")
        boundary = _boundary_2d(source_output_slice, sink_output_slice)
        if np.any(boundary):
            axes[row, 3].contour(
                boundary.astype(np.uint8),
                levels=[0.5],
                colors=[COLORS["boundary"]],
                linewidths=0.8,
                origin="lower",
            )

    success = metrics["source_iou"] >= 0.5 and metrics["sink_iou"] >= 0.5
    core_method = diagnostics.get("core_partition_method", "unknown")
    peripheral_method = diagnostics.get("peripheral_partition_method", "unknown")
    instance_count_before = int(np.count_nonzero(np.unique(instances_before)))
    instance_count_after = int(np.count_nonzero(np.unique(instances_after)))
    figure.suptitle(
        f"{case}, predicted instance {predicted_id}: GT {source_label} vs {sink_label} | "
        f"IoU {metrics['source_iou']:.3f} / {metrics['sink_iou']:.3f} | "
        f"{'success' if success else 'failure'}\n"
        f"all predicted instances {instance_count_before} -> {instance_count_after} | "
        f"core={core_method}, periphery={peripheral_method}",
        fontsize=14,
        y=0.975,
    )
    handles = [
        Patch(color=INSTANCE_COLORS[0], label="Other ID-colored instances"),
        Patch(color=COLORS["merged"], label=f"Selected prediction P{predicted_id}"),
        Patch(color=COLORS["source_gt"], label=f"Selected GT G{source_label}"),
        Patch(color=COLORS["sink_gt"], label=f"Selected GT G{sink_label}"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["source_prompt"], label="Source prompt", markersize=8),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["sink_prompt"], label="Sink prompt", markersize=8),
        Patch(color=COLORS["source_output"], label=f"Source output P{predicted_id}"),
        Patch(color=COLORS["sink_output"], label=f"Sink output P{new_instance_id}"),
        Line2D([0], [0], color="#303030", label="Split boundary"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
    )
    figure.savefig(path, dpi=180, facecolor="white", transparent=False)
    plt.close(figure)


def _gallery_slice_specs(
    cut_mask: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
) -> list[tuple[int, str, str, int]]:
    """Choose three informative slices for each anatomical plane."""
    plane_names = ((0, "axial", "z"), (1, "coronal", "y"), (2, "sagittal", "x"))
    coordinates = np.argwhere(cut_mask)
    specs: list[tuple[int, str, str, int]] = []
    for axis, plane, coordinate_name in plane_names:
        other_axes = tuple(index for index in range(3) if index != axis)
        counts = cut_mask.sum(axis=other_axes)
        cut_index = (
            int(np.argmax(counts))
            if np.any(counts)
            else int(round((source_center[axis] + sink_center[axis]) / 2.0))
        )
        candidates = [
            ("source", int(source_center[axis])),
            ("cut", cut_index),
            ("sink", int(sink_center[axis])),
        ]
        if coordinates.size:
            axis_coordinates = coordinates[:, axis]
            for quantile in (0.25, 0.5, 0.75):
                candidates.append(
                    (
                        f"cut_q{int(quantile * 100)}",
                        int(round(float(np.quantile(axis_coordinates, quantile)))),
                    )
                )
        for index in np.argsort(counts)[::-1]:
            if counts[index] <= 0:
                break
            candidates.append(("cut_support", int(index)))

        chosen: list[tuple[str, int]] = []
        used: set[int] = set()
        for role, index in candidates:
            if index in used:
                continue
            used.add(index)
            chosen.append((role, index))
            if len(chosen) == 3:
                break
        for role, index in chosen:
            specs.append((axis, plane, f"{role}_{coordinate_name}{index:04d}", index))
    return specs


def _slice_bbox(mask: np.ndarray, padding: int) -> tuple[slice, slice]:
    if padding < 0:
        raise ValueError("gallery padding must be non-negative")
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + padding + 1, mask.shape)
    return slice(int(lower[0]), int(upper[0])), slice(int(lower[1]), int(upper[1]))


def _plot_gallery_slice(
    path: Path,
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    ct: np.ndarray,
    instances_before: np.ndarray,
    gt_instances: np.ndarray,
    instances_after: np.ndarray,
    source_prompt: np.ndarray,
    sink_prompt: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_output: np.ndarray,
    sink_output: np.ndarray,
    new_instance_id: int,
    metrics: dict[str, Any],
    ct_window: tuple[float, float],
    spacing_zyx: np.ndarray,
    plane_axis: int,
    plane_name: str,
    role: str,
    slice_index: int,
    padding: int,
    dpi: int,
) -> None:
    arrays = (
        ct,
        instances_before,
        gt_instances,
        instances_after,
        source_prompt,
        sink_prompt,
        source_output,
        sink_output,
    )
    planes = [np.take(array, slice_index, axis=plane_axis) for array in arrays]
    (
        ct_plane,
        before_plane,
        gt_plane,
        after_plane,
        source_prompt_plane,
        sink_prompt_plane,
        source_output_plane,
        sink_output_plane,
    ) = planes
    crop = _slice_bbox(
        (before_plane > 0) | (gt_plane > 0) | (after_plane > 0),
        padding,
    )
    planes = [plane[crop] for plane in planes]
    (
        ct_plane,
        before_plane,
        gt_plane,
        after_plane,
        source_prompt_plane,
        sink_prompt_plane,
        source_output_plane,
        sink_output_plane,
    ) = planes

    figure, axes = plt.subplots(1, 4, figsize=(16, 5.2), facecolor="white")
    remaining_axes = [axis for axis in range(3) if axis != plane_axis]
    aspect = float(spacing_zyx[remaining_axes[0]] / spacing_zyx[remaining_axes[1]])
    for axis in axes:
        axis.imshow(
            ct_plane,
            cmap="gray",
            vmin=ct_window[0],
            vmax=ct_window[1],
            origin="lower",
            interpolation="nearest",
        )
        axis.set_aspect(aspect)
        axis.set_xticks([])
        axis.set_yticks([])

    titles = (
        "All predicted instances (before)",
        "All fragment instance GT",
        "Selected instance and prompts",
        "All predicted instances (after)",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=11)

    _overlay_instances(
        axes[0],
        before_plane,
        0.50,
        {predicted_id: COLORS["merged"]},
    )
    _annotate_instance_ids(axes[0], before_plane, "P")
    _overlay_instances(
        axes[1],
        gt_plane,
        0.52,
        {
            source_label: COLORS["source_gt"],
            sink_label: COLORS["sink_gt"],
        },
    )
    _annotate_instance_ids(axes[1], gt_plane, "G")
    _overlay_instances(
        axes[2],
        before_plane,
        0.22,
        {predicted_id: COLORS["merged"]},
    )
    _overlay(axes[2], source_prompt_plane, COLORS["source_prompt"], 0.88)
    _overlay(axes[2], sink_prompt_plane, COLORS["sink_prompt"], 0.88)

    crop_starts = np.asarray((crop[0].start, crop[1].start), dtype=int)
    for center, color, marker in (
        (source_center, COLORS["source_prompt"], "o"),
        (sink_center, COLORS["sink_prompt"], "s"),
    ):
        if int(center[plane_axis]) != slice_index:
            continue
        plane_point = center[remaining_axes] - crop_starts
        axes[2].scatter(
            int(plane_point[1]),
            int(plane_point[0]),
            s=36,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            marker=marker,
        )

    _overlay_instances(
        axes[3],
        after_plane,
        0.50,
        {
            predicted_id: COLORS["source_output"],
            new_instance_id: COLORS["sink_output"],
        },
    )
    _annotate_instance_ids(axes[3], after_plane, "P")
    boundary = _boundary_2d(source_output_plane, sink_output_plane)
    if np.any(boundary):
        axes[3].contour(
            boundary.astype(np.uint8),
            levels=[0.5],
            colors=[COLORS["boundary"]],
            linewidths=0.9,
            origin="lower",
        )

    figure.suptitle(
        f"{case} | P{predicted_id} -> P{new_instance_id} | "
        f"{plane_name.capitalize()} {role.replace('_', ' ')} slice {slice_index}\n"
        f"Dice source/sink: {metrics['source_dice']:.3f} / "
        f"{metrics['sink_dice']:.3f}",
        fontsize=14,
        y=0.98,
    )
    figure.legend(
        handles=[
            Patch(color=INSTANCE_COLORS[0], label="Other ID-colored instances"),
            Patch(color=COLORS["merged"], label=f"Selected prediction P{predicted_id}"),
            Patch(color=COLORS["source_gt"], label=f"Selected GT G{source_label}"),
            Patch(color=COLORS["sink_gt"], label=f"Selected GT G{sink_label}"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=COLORS["source_prompt"],
                label="Source prompt",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=COLORS["sink_prompt"],
                label="Sink prompt",
                markersize=6,
            ),
            Patch(color=COLORS["source_output"], label=f"Source output P{predicted_id}"),
            Patch(color=COLORS["sink_output"], label=f"Sink output P{new_instance_id}"),
            Line2D([0], [0], color="#303030", label="Split boundary"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.99,
        top=0.78,
        bottom=0.20,
        wspace=0.035,
    )
    figure.savefig(path, dpi=dpi, facecolor="white", transparent=False)
    plt.close(figure)


def _write_slice_gallery(
    gallery_dir: Path,
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    ct: np.ndarray,
    instances_before: np.ndarray,
    gt_instances: np.ndarray,
    instances_after: np.ndarray,
    source_prompt: np.ndarray,
    sink_prompt: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_output: np.ndarray,
    sink_output: np.ndarray,
    cut_mask: np.ndarray,
    new_instance_id: int,
    metrics: dict[str, Any],
    ct_window: tuple[float, float],
    spacing_zyx: np.ndarray,
    padding: int,
    dpi: int,
) -> list[dict[str, Any]]:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for plane_axis, plane_name, role, slice_index in _gallery_slice_specs(
        cut_mask,
        source_center,
        sink_center,
    ):
        filename = f"{plane_name}_{role}.png"
        _plot_gallery_slice(
            gallery_dir / filename,
            case,
            predicted_id,
            source_label,
            sink_label,
            ct,
            instances_before,
            gt_instances,
            instances_after,
            source_prompt,
            sink_prompt,
            source_center,
            sink_center,
            source_output,
            sink_output,
            new_instance_id,
            metrics,
            ct_window,
            spacing_zyx,
            plane_axis,
            plane_name,
            role,
            slice_index,
            padding,
            dpi,
        )
        records.append(
            {
                "file": filename,
                "plane": plane_name,
                "role": role,
                "slice_index": slice_index,
            }
        )
    (gallery_dir / "manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    return records


def _plot_large_plane_board(
    path: Path,
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    ct: np.ndarray,
    instances_before: np.ndarray,
    gt_instances: np.ndarray,
    instances_after: np.ndarray,
    source_prompt: np.ndarray,
    sink_prompt: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_output: np.ndarray,
    sink_output: np.ndarray,
    new_instance_id: int,
    metrics: dict[str, Any],
    ct_window: tuple[float, float],
    spacing_zyx: np.ndarray,
    plane_axis: int,
    plane_name: str,
    slices: list[tuple[str, int]],
    padding: int,
    dpi: int,
) -> None:
    if len(slices) != 3:
        raise ValueError("large plane boards require exactly three slices")

    figure, axes = plt.subplots(3, 4, figsize=(24, 14.5), facecolor="white")
    column_titles = (
        "All predicted instances (before)",
        "All fragment instance GT",
        "Selected instance and prompts",
        "All predicted instances (after)",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=16, pad=10)

    remaining_axes = [axis for axis in range(3) if axis != plane_axis]
    aspect = float(spacing_zyx[remaining_axes[0]] / spacing_zyx[remaining_axes[1]])
    coordinate_name = ("z", "y", "x")[plane_axis]
    arrays = (
        ct,
        instances_before,
        gt_instances,
        instances_after,
        source_prompt,
        sink_prompt,
        source_output,
        sink_output,
    )

    for row, (role, slice_index) in enumerate(slices):
        planes = [np.take(array, slice_index, axis=plane_axis) for array in arrays]
        crop = _slice_bbox(
            (planes[1] > 0) | (planes[2] > 0) | (planes[3] > 0),
            padding,
        )
        planes = [plane[crop] for plane in planes]
        (
            ct_plane,
            before_plane,
            gt_plane,
            after_plane,
            source_prompt_plane,
            sink_prompt_plane,
            source_output_plane,
            sink_output_plane,
        ) = planes

        for axis in axes[row]:
            axis.imshow(
                ct_plane,
                cmap="gray",
                vmin=ct_window[0],
                vmax=ct_window[1],
                origin="lower",
                interpolation="nearest",
            )
            axis.set_aspect(aspect)
            axis.set_xticks([])
            axis.set_yticks([])

        role_name = role.rsplit("_", 1)[0].replace("_", " ").title()
        axes[row, 0].set_ylabel(
            f"{role_name}\n{coordinate_name} = {slice_index}",
            fontsize=14,
            labelpad=12,
        )
        _overlay_instances(
            axes[row, 0],
            before_plane,
            0.50,
            {predicted_id: COLORS["merged"]},
        )
        _annotate_instance_ids(axes[row, 0], before_plane, "P")
        _overlay_instances(
            axes[row, 1],
            gt_plane,
            0.52,
            {
                source_label: COLORS["source_gt"],
                sink_label: COLORS["sink_gt"],
            },
        )
        _annotate_instance_ids(axes[row, 1], gt_plane, "G")
        _overlay_instances(
            axes[row, 2],
            before_plane,
            0.22,
            {predicted_id: COLORS["merged"]},
        )
        _overlay(axes[row, 2], source_prompt_plane, COLORS["source_prompt"], 0.88)
        _overlay(axes[row, 2], sink_prompt_plane, COLORS["sink_prompt"], 0.88)

        crop_starts = np.asarray((crop[0].start, crop[1].start), dtype=int)
        for center, color, marker in (
            (source_center, COLORS["source_prompt"], "o"),
            (sink_center, COLORS["sink_prompt"], "s"),
        ):
            if int(center[plane_axis]) != slice_index:
                continue
            plane_point = center[remaining_axes] - crop_starts
            axes[row, 2].scatter(
                int(plane_point[1]),
                int(plane_point[0]),
                s=32,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                marker=marker,
            )

        _overlay_instances(
            axes[row, 3],
            after_plane,
            0.50,
            {
                predicted_id: COLORS["source_output"],
                new_instance_id: COLORS["sink_output"],
            },
        )
        _annotate_instance_ids(axes[row, 3], after_plane, "P")
        boundary = _boundary_2d(source_output_plane, sink_output_plane)
        if np.any(boundary):
            axes[row, 3].contour(
                boundary.astype(np.uint8),
                levels=[0.5],
                colors=[COLORS["boundary"]],
                linewidths=0.9,
                origin="lower",
            )

    success = metrics["source_iou"] >= 0.5 and metrics["sink_iou"] >= 0.5
    figure.suptitle(
        f"{case} | {plane_name.capitalize()} split review | "
        f"P{predicted_id} -> P{new_instance_id} | GT G{source_label}/G{sink_label}\n"
        f"Dice source/sink: {metrics['source_dice']:.3f} / "
        f"{metrics['sink_dice']:.3f} | {'success' if success else 'failure'}",
        fontsize=22,
        y=0.985,
    )
    figure.legend(
        handles=[
            Patch(color=INSTANCE_COLORS[0], label="Other ID-colored instances"),
            Patch(color=COLORS["merged"], label=f"Selected prediction P{predicted_id}"),
            Patch(color=COLORS["source_gt"], label=f"Selected GT G{source_label}"),
            Patch(color=COLORS["sink_gt"], label=f"Selected GT G{sink_label}"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=COLORS["source_prompt"],
                label="Source prompt",
                markersize=7,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=COLORS["sink_prompt"],
                label="Sink prompt",
                markersize=7,
            ),
            Patch(color=COLORS["source_output"], label=f"Source output P{predicted_id}"),
            Patch(color=COLORS["sink_output"], label=f"Sink output P{new_instance_id}"),
            Line2D([0], [0], color="#303030", label="Split boundary"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=5,
        frameon=False,
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.90,
        bottom=0.105,
        wspace=0.035,
        hspace=0.12,
    )
    figure.savefig(path, dpi=dpi, facecolor="white", transparent=False)
    plt.close(figure)


def _write_large_plane_boards(
    board_dir: Path,
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    ct: np.ndarray,
    instances_before: np.ndarray,
    gt_instances: np.ndarray,
    instances_after: np.ndarray,
    source_prompt: np.ndarray,
    sink_prompt: np.ndarray,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_output: np.ndarray,
    sink_output: np.ndarray,
    cut_mask: np.ndarray,
    new_instance_id: int,
    metrics: dict[str, Any],
    ct_window: tuple[float, float],
    spacing_zyx: np.ndarray,
    padding: int,
    dpi: int,
) -> list[dict[str, Any]]:
    board_dir.mkdir(parents=True, exist_ok=True)
    grouped_specs: dict[tuple[int, str], list[tuple[str, int]]] = {}
    for plane_axis, plane_name, role, slice_index in _gallery_slice_specs(
        cut_mask,
        source_center,
        sink_center,
    ):
        grouped_specs.setdefault((plane_axis, plane_name), []).append((role, slice_index))

    records: list[dict[str, Any]] = []
    for (plane_axis, plane_name), slices in grouped_specs.items():
        filename = f"{plane_name}_large_board.png"
        _plot_large_plane_board(
            board_dir / filename,
            case,
            predicted_id,
            source_label,
            sink_label,
            ct,
            instances_before,
            gt_instances,
            instances_after,
            source_prompt,
            sink_prompt,
            source_center,
            sink_center,
            source_output,
            sink_output,
            new_instance_id,
            metrics,
            ct_window,
            spacing_zyx,
            plane_axis,
            plane_name,
            slices,
            padding,
            dpi,
        )
        records.append(
            {
                "file": filename,
                "plane": plane_name,
                "slices": [
                    {"role": role, "slice_index": slice_index}
                    for role, slice_index in slices
                ],
            }
        )
    (board_dir / "manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    return records


def _write_roi_image(
    path: Path,
    array: np.ndarray,
    reference: sitk.Image,
    crop: tuple[slice, ...],
) -> None:
    image = sitk.GetImageFromArray(array)
    index_xyz = tuple(int(axis_slice.start) for axis_slice in crop[::-1])
    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())
    image.SetOrigin(reference.TransformIndexToPhysicalPoint(index_xyz))
    sitk.WriteImage(image, str(path))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _process_case(
    case: str,
    predicted_id: int,
    source_label: int,
    sink_label: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    abbc_path = args.abbc_dir / f"{case}.nii.gz"
    ct_path = args.ct_dir / f"{case}_0000.nii.gz"
    gt_path = args.gt_dir / f"{case}.nii.gz"
    for path in (abbc_path, ct_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    abbc_image = sitk.ReadImage(str(abbc_path))
    ct_image = sitk.ReadImage(str(ct_path))
    gt_image = sitk.ReadImage(str(gt_path))
    for name, image in (("CT", ct_image), ("GT", gt_image)):
        if image.GetSize() != abbc_image.GetSize():
            raise ValueError(f"{name} and ABBC image sizes do not match")
        if not np.allclose(image.GetSpacing(), abbc_image.GetSpacing()):
            raise ValueError(f"{name} and ABBC image spacings do not match")
        if not np.allclose(image.GetOrigin(), abbc_image.GetOrigin()):
            raise ValueError(f"{name} and ABBC image origins do not match")
        if not np.allclose(image.GetDirection(), abbc_image.GetDirection()):
            raise ValueError(f"{name} and ABBC image directions do not match")

    spacing_zyx = np.asarray(abbc_image.GetSpacing()[::-1], dtype=float)
    abbc_full = sitk.GetArrayFromImage(abbc_image)
    gt_full = sitk.GetArrayFromImage(gt_image)
    crop = _bbox_slices((abbc_full > 0) | (gt_full > 0), args.crop_padding_voxels)
    abbc = np.ascontiguousarray(abbc_full[crop], dtype=np.uint8)
    gt = np.ascontiguousarray(gt_full[crop])
    del abbc_full
    del gt_full

    ct_full = sitk.GetArrayFromImage(ct_image)
    ct = np.ascontiguousarray(ct_full[crop], dtype=np.float32)
    del ct_full

    instances = _hard_label_watershed(abbc, ct, args.min_core_size)
    merged = instances == predicted_id
    if not np.any(merged):
        raise ValueError(f"predicted instance {predicted_id} is absent")
    source_gt = gt == source_label
    sink_gt = gt == sink_label
    source_center = _deepest_center(source_gt & merged, spacing_zyx)
    sink_center = _deepest_center(sink_gt & merged, spacing_zyx)
    source_prompt = _sphere_prompt(merged, source_center, spacing_zyx, args.prompt_radius_mm)
    sink_prompt = _sphere_prompt(merged, sink_center, spacing_zyx, args.prompt_radius_mm)
    if np.any(source_prompt & sink_prompt):
        raise ValueError("source and sink prompt brushes overlap")

    cfg = SplitConfig(
        core_anchor_erosion_mm=args.core_erosion_mm,
        core_anchor_min_voxels=args.core_component_min_voxels,
        core_anchor_distinct_max_distance_mm=args.core_distinct_max_distance_mm,
        core_first_max_graph_voxels=args.max_graph_voxels,
    )
    result, diagnostics = core_first_partition(
        abbc_pred=abbc,
        instance_mask=merged,
        source_interaction_mask=source_prompt,
        sink_interaction_mask=sink_prompt,
        image=ct,
        spacing_zyx=spacing_zyx,
        cfg=cfg,
    )
    source_output, sink_output, metrics = _match_outputs(
        result.source_mask,
        result.sink_mask,
        source_gt,
        sink_gt,
    )
    instances_after, new_instance_id = _apply_split_to_instances(
        instances,
        predicted_id,
        source_output,
        sink_output,
    )

    case_dir = args.outdir / case / f"instance_{predicted_id}_gt_{source_label}_{sink_label}"
    case_dir.mkdir(parents=True, exist_ok=True)
    _plot_case(
        case_dir / "interactive_split_2d.png",
        case,
        predicted_id,
        source_label,
        sink_label,
        ct,
        instances,
        gt,
        instances_after,
        source_prompt,
        sink_prompt,
        source_center,
        sink_center,
        source_output,
        sink_output,
        new_instance_id,
        metrics,
        diagnostics,
        tuple(args.ct_window),
        spacing_zyx,
    )
    gallery_records: list[dict[str, Any]] = []
    if args.slice_gallery:
        gallery_records = _write_slice_gallery(
            case_dir / "slice_gallery",
            case,
            predicted_id,
            source_label,
            sink_label,
            ct,
            instances,
            gt,
            instances_after,
            source_prompt,
            sink_prompt,
            source_center,
            sink_center,
            source_output,
            sink_output,
            result.cut_mask,
            new_instance_id,
            metrics,
            tuple(args.ct_window),
            spacing_zyx,
            args.gallery_padding_voxels,
            args.gallery_dpi,
        )
    board_records: list[dict[str, Any]] = []
    if args.large_plane_boards:
        board_records = _write_large_plane_boards(
            case_dir / "large_boards",
            case,
            predicted_id,
            source_label,
            sink_label,
            ct,
            instances,
            gt,
            instances_after,
            source_prompt,
            sink_prompt,
            source_center,
            sink_center,
            source_output,
            sink_output,
            result.cut_mask,
            new_instance_id,
            metrics,
            tuple(args.ct_window),
            spacing_zyx,
            args.gallery_padding_voxels,
            args.board_dpi,
        )
    if args.write_volumes:
        partition = np.zeros_like(instances, dtype=np.uint8)
        partition[source_output] = 1
        partition[sink_output] = 2
        prompts = np.zeros_like(instances, dtype=np.uint8)
        prompts[source_prompt] = 1
        prompts[sink_prompt] = 2
        _write_roi_image(case_dir / "partition.nii.gz", partition, abbc_image, crop)
        _write_roi_image(case_dir / "prompts.nii.gz", prompts, abbc_image, crop)
        _write_roi_image(
            case_dir / "instances_before.nii.gz",
            instances.astype(np.uint32),
            abbc_image,
            crop,
        )
        _write_roi_image(
            case_dir / "instances_after.nii.gz",
            instances_after.astype(np.uint32),
            abbc_image,
            crop,
        )

    crop_start_zyx = np.asarray([axis_slice.start for axis_slice in crop], dtype=int)
    global_source_zyx = source_center + crop_start_zyx
    global_sink_zyx = sink_center + crop_start_zyx
    metadata = {
        "case": case,
        "predicted_instance": predicted_id,
        "new_instance": new_instance_id,
        "predicted_instance_count_before": int(np.count_nonzero(np.unique(instances))),
        "predicted_instance_count_after": int(np.count_nonzero(np.unique(instances_after))),
        "gt_instance_count_in_visualized_roi": int(np.count_nonzero(np.unique(gt))),
        "source_gt_label": source_label,
        "sink_gt_label": sink_label,
        "prompt_radius_mm": args.prompt_radius_mm,
        "source_prompt_center_zyx": global_source_zyx.tolist(),
        "sink_prompt_center_zyx": global_sink_zyx.tolist(),
        "source_prompt_center_physical_xyz_mm": abbc_image.TransformIndexToPhysicalPoint(
            tuple(int(value) for value in global_source_zyx[::-1])
        ),
        "sink_prompt_center_physical_xyz_mm": abbc_image.TransformIndexToPhysicalPoint(
            tuple(int(value) for value in global_sink_zyx[::-1])
        ),
        "instance_success": metrics["source_iou"] >= 0.5 and metrics["sink_iou"] >= 0.5,
        "slice_gallery": gallery_records,
        "large_plane_boards": board_records,
        "metrics": metrics,
        "diagnostics": {key: _json_value(value) for key, value in diagnostics.items()},
    }
    (case_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = _parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for value in args.case:
        case, predicted_id, source_label, sink_label = _parse_case_spec(value)
        print(f"Processing {case}, instance {predicted_id}, GT {source_label}/{sink_label}", flush=True)
        result = _process_case(case, predicted_id, source_label, sink_label, args)
        results.append(result)
        print(
            f"  success={result['instance_success']}, "
            f"IoU={result['metrics']['source_iou']:.3f}/{result['metrics']['sink_iou']:.3f}",
            flush=True,
        )
        gc.collect()
    (args.outdir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.outdir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
