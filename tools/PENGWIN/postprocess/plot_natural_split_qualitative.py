"""Plot CT planes for a natural under-split interaction benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.PENGWIN.postprocess.benchmark_interactive_mincut import _read_roi


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--out")
    parser.add_argument("--window-min", type=float, default=-200.0)
    parser.add_argument("--window-max", type=float, default=2000.0)
    return parser


def _plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis)


def _overlay(axis: plt.Axes, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha
    axis.imshow(rgba, origin="lower", interpolation="nearest")


def main() -> None:
    args = _parser().parse_args()
    result_dir = Path(args.result_dir)
    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    protocol = summary["protocol"]
    if protocol["input_mode"] != "natural_under_split":
        raise SystemExit("qualitative plot requires a natural_under_split result")

    roi_index = tuple(int(value) for value in protocol["roi_index_xyz"])
    roi_size = tuple(int(value) for value in protocol["roi_size_xyz"])
    ct = _read_roi(Path(args.ct), roi_index, roi_size).astype(np.float32)
    instances = _read_roi(Path(protocol["inputs"]["instances"]), roi_index, roi_size)
    source_gt = _read_roi(Path(protocol["inputs"]["source_gt"]), roi_index, roi_size) > 0
    sink_gt = _read_roi(Path(protocol["inputs"]["sink_gt"]), roi_index, roi_size) > 0
    partition = sitk.GetArrayFromImage(
        sitk.ReadImage(str(result_dir / "representative_partition.nii.gz"))
    )
    cut = sitk.GetArrayFromImage(
        sitk.ReadImage(str(result_dir / "representative_cut.nii.gz"))
    ) > 0
    interactions = sitk.GetArrayFromImage(
        sitk.ReadImage(str(result_dir / "representative_interactions.nii.gz"))
    )
    if any(array.shape != ct.shape for array in (instances, source_gt, sink_gt, partition, cut, interactions)):
        raise SystemExit("ROI arrays do not share one shape")

    input_mask = instances == int(protocol["instance_id"])
    source_prediction = partition == 1
    sink_prediction = partition == 2
    plane_names = ("Axial", "Coronal", "Sagittal")
    sink_interaction_points = np.argwhere(interactions == 2)
    if sink_interaction_points.size:
        slice_indices = np.rint(sink_interaction_points.mean(axis=0)).astype(int).tolist()
        plane_context = "through sink interaction"
    else:
        slice_indices = [
            int(np.argmax(sink_gt.sum(axis=tuple(other for other in range(3) if other != axis))))
            for axis in range(3)
        ]
        plane_context = "maximum fragment area"
    spacing_zyx = np.asarray(protocol["spacing_xyz_mm"][::-1], dtype=float)
    aspects = (
        spacing_zyx[1] / spacing_zyx[2],
        spacing_zyx[0] / spacing_zyx[2],
        spacing_zyx[0] / spacing_zyx[1],
    )

    figure, axes = plt.subplots(3, 3, figsize=(13.5, 12), facecolor="white")
    for row in axes:
        for axis in row:
            axis.set_facecolor("black")
    row_titles = ("Merged watershed input", "Fragment instance GT", "Core-first output")
    for column, (plane_axis, slice_index) in enumerate(zip(range(3), slice_indices)):
        ct_plane = _plane(ct, plane_axis, slice_index)
        for row in range(3):
            axis = axes[row, column]
            axis.imshow(
                ct_plane,
                cmap="gray",
                vmin=args.window_min,
                vmax=args.window_max,
                origin="lower",
                interpolation="nearest",
                aspect=aspects[column],
            )
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0, column].set_title(
            f"{plane_names[column]} ({plane_context}, index {slice_index})",
            fontsize=12,
        )
        _overlay(axes[0, column], _plane(input_mask, plane_axis, slice_index), (1.0, 0.78, 0.12), 0.32)
        _overlay(axes[1, column], _plane(source_gt, plane_axis, slice_index), (0.10, 0.65, 0.90), 0.28)
        _overlay(axes[1, column], _plane(sink_gt, plane_axis, slice_index), (0.88, 0.18, 0.38), 0.65)
        _overlay(
            axes[2, column],
            _plane(source_prediction, plane_axis, slice_index),
            (0.10, 0.65, 0.90),
            0.25,
        )
        _overlay(
            axes[2, column],
            _plane(sink_prediction, plane_axis, slice_index),
            (1.0, 0.48, 0.05),
            0.65,
        )
        _overlay(axes[2, column], _plane(cut, plane_axis, slice_index), (1.0, 1.0, 1.0), 0.9)
        _overlay(
            axes[2, column],
            _plane(interactions == 1, plane_axis, slice_index),
            (0.15, 0.95, 0.35),
            0.95,
        )
        _overlay(
            axes[2, column],
            _plane(interactions == 2, plane_axis, slice_index),
            (0.95, 0.15, 0.85),
            0.95,
        )

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=12)
    figure.suptitle("Case 95: natural under-split instance 2", fontsize=16)
    figure.legend(
        handles=[
            Patch(facecolor="#f6c51f", label="Merged predicted instance"),
            Patch(facecolor="#1aa6e6", label="Main tibia"),
            Patch(facecolor="#e02e61", label="Fragment 8 GT"),
            Patch(facecolor="#ff7a0d", label="Fragment 8 output"),
            Patch(facecolor="white", edgecolor="black", label="Partition boundary"),
            Patch(facecolor="#26f259", label="Source interaction"),
            Patch(facecolor="#f226d9", label="Sink interaction"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    output = Path(args.out) if args.out else result_dir / "qualitative_ct_planes.png"
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
