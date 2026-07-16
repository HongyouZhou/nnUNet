#!/usr/bin/env python3
"""Find natural under-split instances suitable for interactive split tests.

The scanner converts hard ABBC predictions to watershed instances, finds one
predicted instance that substantially covers multiple GT instances, and then
classifies whether a spacing-aware erosion leaves distinct GT-associated
Label-2 core components.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label as nd_label
from skimage.segmentation import watershed


BONE_LABEL_RANGES = (
    (1, 20, "tibia"),
    (21, 30, "fibula"),
    (31, 35, "femur"),
    (36, 40, "patella"),
    (41, 45, "fabella"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abbc-dir", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--case-glob", default="charite_*.nii.gz")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--ct-pattern", default="{case}_0000.nii.gz")
    parser.add_argument("--gt-pattern", default="{case}.nii.gz")
    parser.add_argument("--crop-padding-voxels", type=int, default=2)
    parser.add_argument("--min-core-size", type=int, default=10)
    parser.add_argument("--min-overlap-voxels", type=int, default=500)
    parser.add_argument("--min-gt-coverage", type=float, default=0.2)
    parser.add_argument("--min-gt-size", type=int, default=1000)
    parser.add_argument("--core-erosion-mm", type=float, default=2.0)
    parser.add_argument("--core-component-min-voxels", type=int, default=20)
    parser.add_argument("--same-bone-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_image(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(float(value) for value in image.GetSpacing()[::-1])
    return array, spacing_zyx


def _bbox_slices(mask: np.ndarray, padding: int) -> tuple[slice, ...]:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        raise ValueError("cannot compute a bounding box for an empty mask")
    slices = []
    for axis_coordinates, axis_size in zip(coordinates, mask.shape):
        lower = max(int(axis_coordinates.min()) - padding, 0)
        upper = min(int(axis_coordinates.max()) + padding + 1, axis_size)
        slices.append(slice(lower, upper))
    return tuple(slices)


def _bone_name(label_id: int) -> str:
    for lower, upper, name in BONE_LABEL_RANGES:
        if lower <= label_id <= upper:
            return name
    return "unknown"


def _hard_label_watershed(
    abbc: np.ndarray,
    ct: np.ndarray,
    min_core_size: int,
) -> np.ndarray:
    """Match the current hard-label watershed conversion without one-hot copies."""
    foreground = abbc > 0
    core_components, _ = nd_label(abbc == 2)
    core_sizes = np.bincount(core_components.ravel())
    keep_core = core_sizes > min_core_size
    keep_core[0] = False
    filtered_core = keep_core[core_components]
    markers, _ = nd_label(filtered_core)

    distance = distance_transform_edt(foreground)
    maximum = float(distance.max())
    if maximum > 0:
        distance /= maximum
    normalized_ct = np.clip(ct.astype(np.float32), -200.0, 1500.0)
    normalized_ct += 200.0
    normalized_ct /= 1700.0
    distance -= 2.0 * (abbc == 3)
    distance -= normalized_ct
    np.negative(distance, out=distance)
    return watershed(distance, markers, mask=foreground).astype(np.uint16)


def _overlap_metrics(
    overlap: int,
    predicted_size: int,
    gt_size: int,
) -> tuple[float, float, float]:
    coverage = overlap / gt_size if gt_size else 0.0
    dice = 2.0 * overlap / (predicted_size + gt_size)
    union = predicted_size + gt_size - overlap
    iou = overlap / union if union else 0.0
    return float(coverage), float(dice), float(iou)


def _dominant_component(
    components: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[int, int]:
    counts = np.bincount(components[gt_mask].ravel())
    if len(counts) <= 1:
        return 0, 0
    counts[0] = 0
    component_id = int(np.argmax(counts))
    return component_id, int(counts[component_id])


def _core_pair_diagnostics(
    abbc: np.ndarray,
    predicted_mask: np.ndarray,
    gt: np.ndarray,
    first_gt_label: int,
    second_gt_label: int,
    spacing_zyx: tuple[float, float, float],
    erosion_mm: float,
    minimum_component_voxels: int,
) -> dict[str, Any]:
    crop = _bbox_slices(predicted_mask, padding=1)
    predicted_crop = predicted_mask[crop]
    abbc_crop = abbc[crop]
    gt_crop = gt[crop]
    raw_core = (abbc_crop == 2) & predicted_crop
    if erosion_mm > 0:
        core_depth = distance_transform_edt(raw_core, sampling=spacing_zyx)
        robust_core = raw_core & (core_depth > erosion_mm)
    else:
        robust_core = raw_core

    components, _ = nd_label(robust_core)
    component_sizes = np.bincount(components.ravel())
    keep = component_sizes >= minimum_component_voxels
    keep[0] = False
    components[~keep[components]] = 0
    retained_ids = np.flatnonzero(keep)

    first_gt_mask = gt_crop == first_gt_label
    second_gt_mask = gt_crop == second_gt_label
    first_id, first_dominant_voxels = _dominant_component(components, first_gt_mask)
    second_id, second_dominant_voxels = _dominant_component(components, second_gt_mask)
    first_eroded_voxels = int(np.count_nonzero(robust_core & first_gt_mask))
    second_eroded_voxels = int(np.count_nonzero(robust_core & second_gt_mask))

    if first_id == 0 or second_id == 0:
        status = "missing"
    elif first_id == second_id:
        status = "shared"
    else:
        status = "separate"
    return {
        "core_status": status,
        "robust_core_components": int(len(retained_ids)),
        "first_eroded_core_voxels": first_eroded_voxels,
        "second_eroded_core_voxels": second_eroded_voxels,
        "first_dominant_core_component": first_id,
        "second_dominant_core_component": second_id,
        "first_dominant_core_voxels": first_dominant_voxels,
        "second_dominant_core_voxels": second_dominant_voxels,
    }


def _scan_case(
    case: str,
    abbc_path: Path,
    ct_path: Path,
    gt_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    abbc_full, spacing_zyx = _read_image(abbc_path)
    foreground = abbc_full > 0
    crop = _bbox_slices(foreground, args.crop_padding_voxels)
    full_shape = tuple(int(value) for value in abbc_full.shape)
    abbc = np.ascontiguousarray(abbc_full[crop], dtype=np.uint8)
    del abbc_full, foreground

    ct_full, ct_spacing_zyx = _read_image(ct_path)
    if tuple(ct_full.shape) != full_shape or not np.allclose(ct_spacing_zyx, spacing_zyx):
        raise ValueError("CT and ABBC grids do not match")
    ct = np.ascontiguousarray(ct_full[crop], dtype=np.float32)
    del ct_full

    gt_full, gt_spacing_zyx = _read_image(gt_path)
    if tuple(gt_full.shape) != full_shape or not np.allclose(gt_spacing_zyx, spacing_zyx):
        raise ValueError("GT and ABBC grids do not match")
    gt_sizes = np.bincount(gt_full.ravel())
    gt = np.ascontiguousarray(gt_full[crop])
    del gt_full

    instances = _hard_label_watershed(abbc, ct, args.min_core_size)
    del ct
    candidates: list[dict[str, Any]] = []
    predicted_ids = np.unique(instances)
    predicted_ids = predicted_ids[predicted_ids > 0]
    for predicted_id in predicted_ids:
        predicted_mask = instances == predicted_id
        predicted_size = int(predicted_mask.sum())
        labels, overlaps = np.unique(gt[predicted_mask], return_counts=True)
        eligible: list[dict[str, Any]] = []
        for gt_label, overlap in zip(labels, overlaps):
            gt_label = int(gt_label)
            overlap = int(overlap)
            if gt_label == 0 or gt_label >= len(gt_sizes):
                continue
            gt_size = int(gt_sizes[gt_label])
            coverage, dice, iou = _overlap_metrics(overlap, predicted_size, gt_size)
            if (
                overlap >= args.min_overlap_voxels
                and gt_size >= args.min_gt_size
                and coverage >= args.min_gt_coverage
            ):
                eligible.append(
                    {
                        "gt_label": gt_label,
                        "bone": _bone_name(gt_label),
                        "gt_voxels": gt_size,
                        "overlap_voxels": overlap,
                        "coverage": coverage,
                        "merged_prediction_dice": dice,
                        "merged_prediction_iou": iou,
                    }
                )

        for first, second in itertools.combinations(eligible, 2):
            if args.same_bone_only and first["bone"] != second["bone"]:
                continue
            diagnostics = _core_pair_diagnostics(
                abbc,
                predicted_mask,
                gt,
                first["gt_label"],
                second["gt_label"],
                spacing_zyx,
                args.core_erosion_mm,
                args.core_component_min_voxels,
            )
            candidates.append(
                {
                    "case": case,
                    "predicted_instance": int(predicted_id),
                    "predicted_voxels": predicted_size,
                    "first_gt_label": first["gt_label"],
                    "second_gt_label": second["gt_label"],
                    "bone": first["bone"],
                    "first_gt_voxels": first["gt_voxels"],
                    "second_gt_voxels": second["gt_voxels"],
                    "first_overlap_voxels": first["overlap_voxels"],
                    "second_overlap_voxels": second["overlap_voxels"],
                    "first_gt_coverage": first["coverage"],
                    "second_gt_coverage": second["coverage"],
                    "first_merged_dice": first["merged_prediction_dice"],
                    "second_merged_dice": second["merged_prediction_dice"],
                    "first_merged_iou": first["merged_prediction_iou"],
                    "second_merged_iou": second["merged_prediction_iou"],
                    "minimum_gt_coverage": min(first["coverage"], second["coverage"]),
                    **diagnostics,
                }
            )

    return {
        "case": case,
        "status": "ok",
        "shape_zyx": full_shape,
        "crop_shape_zyx": tuple(int(value) for value in abbc.shape),
        "spacing_zyx_mm": spacing_zyx,
        "gt_instances": int(np.count_nonzero(gt_sizes[1:])),
        "predicted_instances": int(len(predicted_ids)),
        "candidates": candidates,
    }


def _write_aggregate(outdir: Path, case_results: list[dict[str, Any]]) -> None:
    candidates = [
        candidate
        for result in case_results
        for candidate in result.get("candidates", [])
    ]
    merged_gt_labels: dict[tuple[str, int], set[int]] = {}
    for candidate in candidates:
        key = (candidate["case"], candidate["predicted_instance"])
        labels = merged_gt_labels.setdefault(key, set())
        labels.add(candidate["first_gt_label"])
        labels.add(candidate["second_gt_label"])
    for candidate in candidates:
        key = (candidate["case"], candidate["predicted_instance"])
        labels = sorted(merged_gt_labels[key])
        candidate["merged_gt_count"] = len(labels)
        candidate["merged_gt_labels"] = ";".join(str(label) for label in labels)

    status_order = {"shared": 0, "missing": 1, "separate": 2}
    candidates.sort(
        key=lambda item: (
            status_order.get(item["core_status"], 3),
            -item["minimum_gt_coverage"],
            -min(item["first_overlap_voxels"], item["second_overlap_voxels"]),
        )
    )
    summary = {
        "cases": len(case_results),
        "successful_cases": sum(result.get("status") == "ok" for result in case_results),
        "failed_cases": [
            {"case": result["case"], "error": result.get("error", "")}
            for result in case_results
            if result.get("status") != "ok"
        ],
        "candidate_pairs": len(candidates),
        "candidate_merged_instances": len(merged_gt_labels),
        "two_gt_merged_instances": sum(
            len(labels) == 2 for labels in merged_gt_labels.values()
        ),
        "core_status_counts": {
            status: sum(candidate["core_status"] == status for candidate in candidates)
            for status in ("separate", "shared", "missing")
        },
        "two_gt_core_status_counts": {
            status: sum(
                candidate["merged_gt_count"] == 2
                and candidate["core_status"] == status
                for candidate in candidates
            )
            for status in ("separate", "shared", "missing")
        },
        "candidates": candidates,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (outdir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        if candidates:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)


def main() -> None:
    args = _parser().parse_args()
    for path in (args.abbc_dir, args.ct_dir, args.gt_dir):
        if not path.is_dir():
            raise SystemExit(f"missing input directory: {path}")
    if args.crop_padding_voxels < 0:
        raise SystemExit("crop-padding-voxels must be non-negative")
    if not 0 <= args.min_gt_coverage <= 1:
        raise SystemExit("min-gt-coverage must be between 0 and 1")

    args.outdir.mkdir(parents=True, exist_ok=True)
    per_case_dir = args.outdir / "per_case"
    per_case_dir.mkdir(exist_ok=True)
    if args.cases:
        cases = [case.removesuffix(".nii.gz") for case in args.cases]
    else:
        cases = sorted(path.name.removesuffix(".nii.gz") for path in args.abbc_dir.glob(args.case_glob))
    if not cases:
        raise SystemExit("no ABBC cases matched")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        result_path = per_case_dir / f"{case}.json"
        if result_path.is_file() and not args.overwrite:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            print(f"[{index}/{len(cases)}] {case}: cached", flush=True)
            continue

        abbc_path = args.abbc_dir / f"{case}.nii.gz"
        ct_path = args.ct_dir / args.ct_pattern.format(case=case)
        gt_path = args.gt_dir / args.gt_pattern.format(case=case)
        print(f"[{index}/{len(cases)}] {case}: scanning", flush=True)
        try:
            missing = [str(path) for path in (abbc_path, ct_path, gt_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing files: {missing}")
            result = _scan_case(case, abbc_path, ct_path, gt_path, args)
        except Exception as error:  # Preserve progress across heterogeneous cases.
            result = {"case": case, "status": "error", "error": repr(error), "candidates": []}
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(
            f"[{index}/{len(cases)}] {case}: {result['status']}, "
            f"candidates={len(result.get('candidates', []))}",
            flush=True,
        )
        _write_aggregate(args.outdir, results)

    _write_aggregate(args.outdir, results)
    print(f"Wrote {args.outdir / 'summary.json'}", flush=True)
    print(f"Wrote {args.outdir / 'candidates.csv'}", flush=True)


if __name__ == "__main__":
    main()
