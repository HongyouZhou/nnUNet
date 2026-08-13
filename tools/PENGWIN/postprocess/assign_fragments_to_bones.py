#!/usr/bin/env python3
"""Assign fragment instance IDs to coarse anatomical bone classes.

This module intentionally does not alter the instance label image. It overlays
TotalSegmentator semantic masks with an existing fragment instance map and
writes a JSON lookup that a downstream application can use to display names
such as ``tibia_7``.

The NIfTI path is processed in z chunks so that long CT volumes do not require
all semantic masks and the instance map to be resident in memory at once.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np


BONE_ORDER = ("tibia", "fibula", "femur", "patella", "fabella")

# The aggregate label map is retained for consumers that only accept one
# NIfTI.  The named binary masks below are the canonical REPAIR/KOMO case
# representation.
BONE_LABEL_BLOCKS: dict[str, tuple[int, int]] = {
    "tibia": (1, 20),
    "fibula": (21, 10),
    "femur": (31, 5),
    "patella": (36, 5),
    "fabella": (41, 5),
}

# TotalSegmentator's appendicular-bones task emits unsided masks, while an
# existing REPAIR postprocessing step may replace them with sided masks.
DEFAULT_MASK_FILENAMES: dict[str, tuple[str, ...]] = {
    "tibia": ("tibia.nii.gz", "tibia_left.nii.gz", "tibia_right.nii.gz"),
    "fibula": ("fibula.nii.gz", "fibula_left.nii.gz", "fibula_right.nii.gz"),
    "femur": ("femur.nii.gz", "femur_left.nii.gz", "femur_right.nii.gz"),
    "patella": ("patella.nii.gz", "patella_left.nii.gz", "patella_right.nii.gz"),
    "fabella": ("fabella.nii.gz", "fabella_left.nii.gz", "fabella_right.nii.gz"),
}


def build_named_segmentations(
    instances: np.ndarray,
    assignments: Mapping[str, Mapping[str, Any]],
    laterality: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert postprocessed instances to the KOMO named-mask convention.

    The largest instance for each bone keeps the base name (for example
    ``tibia_L.nii.gz``). Remaining instances are numbered by decreasing size
    as ``tibia_L_fragment_1.nii.gz`` and so on.  Every named mask uses 255 as
    foreground, matching ``all_komo_cases.zip``.
    """

    side = str(laterality).strip().upper()
    if side not in {"L", "R"}:
        raise ValueError(f"laterality must be L or R, got {laterality!r}")

    instance_array = _validate_instance_chunk(np.asarray(instances))
    grouped: dict[str, list[tuple[int, int]]] = {bone: [] for bone in BONE_ORDER}
    present_ids, present_counts = np.unique(
        instance_array[instance_array > 0], return_counts=True
    )
    for raw_id, raw_count in zip(present_ids, present_counts):
        instance_id = int(raw_id)
        assignment = assignments.get(str(instance_id))
        if assignment is None:
            raise ValueError(f"post-processing has no bone assignment for instance {instance_id}")
        bone = str(assignment.get("bone", "unknown")).lower()
        if bone not in BONE_LABEL_BLOCKS:
            raise ValueError(
                f"instance {instance_id} has no usable bone name after post-processing: {bone}"
            )
        grouped[bone].append((instance_id, int(raw_count)))

    named_masks: dict[str, np.ndarray] = {}
    aggregate = np.zeros(instance_array.shape, dtype=np.uint8)
    for bone in BONE_ORDER:
        ranked = sorted(grouped[bone], key=lambda item: (-item[1], item[0]))
        block_start, capacity = BONE_LABEL_BLOCKS[bone]
        if len(ranked) > capacity:
            raise ValueError(
                f"post-processing assigned {len(ranked)} {bone} instances; "
                f"the case format supports {capacity}"
            )
        for rank, (instance_id, _count) in enumerate(ranked):
            stem = f"{bone}_{side}" if rank == 0 else f"{bone}_{side}_fragment_{rank}"
            mask = instance_array == instance_id
            named_masks[f"{stem}.nii.gz"] = mask.astype(np.uint8) * 255
            aggregate[mask] = block_start + rank

    if not named_masks:
        raise ValueError("post-processing produced no named bone segmentations")
    return named_masks, aggregate


def build_aggregate_label_names(
    named_masks: Mapping[str, np.ndarray], laterality: str
) -> dict[int, str]:
    """Return aggregate label names matching the named-mask filenames."""

    side = str(laterality).strip().upper()
    if side not in {"L", "R"}:
        raise ValueError(f"laterality must be L or R, got {laterality!r}")

    labels: dict[int, str] = {0: "background"}
    pattern = re.compile(
        rf"^({'|'.join(BONE_ORDER)})_{side}(?:_fragment_([1-9][0-9]*))?\.nii\.gz$"
    )
    for filename in named_masks:
        match = pattern.fullmatch(filename)
        if match is None:
            raise ValueError(f"Named segmentation does not match side {side}: {filename}")
        bone, fragment_text = match.groups()
        rank = 0 if fragment_text is None else int(fragment_text)
        block_start, capacity = BONE_LABEL_BLOCKS[bone]
        if rank >= capacity:
            raise ValueError(f"Named segmentation exceeds {bone} label block: {filename}")
        labels[block_start + rank] = filename.removesuffix(".nii.gz")
    return dict(sorted(labels.items()))


def export_named_segmentations(
    instance_path: str | Path,
    report: Mapping[str, Any],
    output_dir: str | Path,
    laterality: str,
    *,
    aggregate_filename: str = "seg.nii.gz",
) -> tuple[list[Path], Path]:
    """Write named binary masks plus a bone-aware aggregate label map."""

    reference = nib.load(str(instance_path))
    instances = np.rint(np.asanyarray(reference.dataobj)).astype(np.int64)
    assignments = report.get("instances")
    if not isinstance(assignments, Mapping):
        raise ValueError("post-processing report does not contain instance assignments")
    named_masks, aggregate = build_named_segmentations(
        instances, assignments, laterality
    )
    label_names = build_aggregate_label_names(named_masks, laterality)

    root = Path(output_dir)
    segmentation_dir = root / "segmentations"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    for stale in segmentation_dir.glob("*.nii.gz"):
        stale.unlink()

    written: list[Path] = []
    for filename, mask in named_masks.items():
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        path = segmentation_dir / filename
        nib.save(nib.Nifti1Image(mask, reference.affine, header), str(path))
        written.append(path)

    aggregate_header = reference.header.copy()
    aggregate_header.set_data_dtype(np.uint8)
    aggregate_header.extensions.clear()
    aggregate_header.extensions.append(
        nib.nifti1.Nifti1Extension(
            6,
            json.dumps(
                {
                    "schema": "repair.segmentation.labels.v1",
                    "side": str(laterality).strip().upper(),
                    "labels": {str(value): name for value, name in label_names.items()},
                },
                sort_keys=True,
            ).encode("utf-8"),
        )
    )
    aggregate_path = root / aggregate_filename
    nib.save(
        nib.Nifti1Image(aggregate, reference.affine, aggregate_header),
        str(aggregate_path),
    )
    return sorted(written), aggregate_path


@dataclass(frozen=True)
class AssignmentConfig:
    """Decision thresholds for assigning a bone name to a fragment."""

    min_overlap: float = 0.5
    min_margin: float = 0.2

    def validate(self) -> None:
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ValueError("min_overlap must be in [0, 1]")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("min_margin must be in [0, 1]")


def discover_totalsegmentator_masks(
    totalseg_dir: str | Path,
) -> dict[str, list[Path]]:
    """Find supported TotalSegmentator masks without requiring every class."""

    root = Path(totalseg_dir)
    if not root.is_dir():
        return {bone: [] for bone in BONE_ORDER}

    return {
        bone: [root / filename for filename in DEFAULT_MASK_FILENAMES[bone] if (root / filename).is_file()]
        for bone in BONE_ORDER
    }


def _update_counts(target: Counter[int], labels: np.ndarray) -> None:
    if labels.size == 0:
        return
    unique, counts = np.unique(labels, return_counts=True)
    target.update({int(label): int(count) for label, count in zip(unique, counts)})


def _validate_instance_chunk(chunk: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(chunk)):
        raise ValueError("instance image contains non-finite values")
    if np.any(chunk < 0):
        raise ValueError("instance IDs must be non-negative")
    rounded = np.rint(chunk)
    if not np.array_equal(chunk, rounded):
        raise ValueError("instance IDs must be discrete integers")
    return rounded.astype(np.int64, copy=False)


def _validate_mask_chunk(chunk: np.ndarray, mask_path: Path) -> np.ndarray:
    if not np.all(np.isfinite(chunk)):
        raise ValueError(f"bone mask contains non-finite values: {mask_path}")
    return chunk > 0


def _validate_mask_grid(
    reference: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    mask_path: Path,
    affine_atol: float,
) -> None:
    if mask.shape != reference.shape:
        raise ValueError(
            f"bone mask shape does not match instances: {mask_path} has {mask.shape}, "
            f"expected {reference.shape}"
        )
    if not np.allclose(mask.affine, reference.affine, rtol=1e-5, atol=affine_atol):
        raise ValueError(f"bone mask affine does not match instances: {mask_path}")


def compute_nifti_overlap_counts(
    instance_path: str | Path,
    mask_paths: Mapping[str, Sequence[str | Path]],
    *,
    chunk_depth: int = 128,
    affine_atol: float = 1e-4,
) -> tuple[Counter[int], dict[str, Counter[int]]]:
    """Compute per-fragment voxel and semantic-overlap counts in chunks.

    Multiple mask files for one class are unioned within each chunk. This
    supports both unsided masks and left/right masks without double counting.
    """

    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    if affine_atol < 0:
        raise ValueError("affine_atol must be non-negative")

    instance_image = nib.load(str(instance_path))
    if len(instance_image.shape) != 3:
        raise ValueError(f"instance image must be 3D, got shape {instance_image.shape}")

    loaded_masks: dict[str, list[tuple[Path, nib.spatialimages.SpatialImage]]] = {
        bone: [] for bone in BONE_ORDER
    }
    for bone in BONE_ORDER:
        for path_like in mask_paths.get(bone, ()):
            path = Path(path_like)
            mask_image = nib.load(str(path))
            _validate_mask_grid(instance_image, mask_image, path, affine_atol)
            loaded_masks[bone].append((path, mask_image))

    instance_counts: Counter[int] = Counter()
    overlap_counts = {bone: Counter() for bone in BONE_ORDER}
    depth = instance_image.shape[2]

    for start in range(0, depth, chunk_depth):
        stop = min(start + chunk_depth, depth)
        spatial_slice = (slice(None), slice(None), slice(start, stop))
        instances = _validate_instance_chunk(np.asanyarray(instance_image.dataobj[spatial_slice]))
        foreground = instances > 0
        _update_counts(instance_counts, instances[foreground])

        if not np.any(foreground):
            continue

        for bone in BONE_ORDER:
            union_mask = np.zeros(instances.shape, dtype=bool)
            for mask_path, mask_image in loaded_masks[bone]:
                mask_chunk = np.asanyarray(mask_image.dataobj[spatial_slice])
                union_mask |= _validate_mask_chunk(mask_chunk, mask_path)
            _update_counts(overlap_counts[bone], instances[foreground & union_mask])

    return instance_counts, overlap_counts


def compute_array_overlap_counts(
    instances: np.ndarray,
    bone_masks: Mapping[str, np.ndarray],
) -> tuple[Counter[int], dict[str, Counter[int]]]:
    """In-memory overlap counter used by unit tests and small volumes."""

    if instances.ndim != 3:
        raise ValueError(f"instances must be 3D, got shape {instances.shape}")
    instance_labels = _validate_instance_chunk(np.asarray(instances))
    foreground = instance_labels > 0
    instance_counts: Counter[int] = Counter()
    _update_counts(instance_counts, instance_labels[foreground])

    overlap_counts = {bone: Counter() for bone in BONE_ORDER}
    for bone in BONE_ORDER:
        mask = bone_masks.get(bone)
        if mask is None:
            continue
        mask_array = np.asarray(mask)
        if mask_array.shape != instance_labels.shape:
            raise ValueError(
                f"{bone} mask shape {mask_array.shape} does not match instances {instance_labels.shape}"
            )
        mask_bool = _validate_mask_chunk(mask_array, Path(f"<{bone} array>"))
        _update_counts(overlap_counts[bone], instance_labels[foreground & mask_bool])

    return instance_counts, overlap_counts


def assign_from_overlap_counts(
    instance_counts: Mapping[int, int],
    overlap_counts: Mapping[str, Mapping[int, int]],
    config: AssignmentConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply overlap and margin thresholds to precomputed counts."""

    cfg = config or AssignmentConfig()
    cfg.validate()
    assignments: dict[str, dict[str, Any]] = {}

    for instance_id in sorted(int(key) for key in instance_counts if int(key) > 0):
        voxel_count = int(instance_counts[instance_id])
        if voxel_count <= 0:
            raise ValueError(f"instance {instance_id} has a non-positive voxel count")

        overlap_voxels = {
            bone: int(overlap_counts.get(bone, {}).get(instance_id, 0)) for bone in BONE_ORDER
        }
        overlap_fraction = {
            bone: overlap_voxels[bone] / voxel_count for bone in BONE_ORDER
        }
        ranked = sorted(
            BONE_ORDER,
            key=lambda bone: (-overlap_fraction[bone], BONE_ORDER.index(bone)),
        )
        best_bone, second_bone = ranked[:2]
        best_overlap = overlap_fraction[best_bone]
        second_overlap = overlap_fraction[second_bone]
        margin = best_overlap - second_overlap

        if best_overlap < cfg.min_overlap:
            assigned_bone = "unknown"
            reason = "overlap_below_threshold"
        elif margin < cfg.min_margin:
            assigned_bone = "unknown"
            reason = "ambiguous_overlap_margin"
        else:
            assigned_bone = best_bone
            reason = "assigned"

        assignments[str(instance_id)] = {
            "name": f"{assigned_bone}_{instance_id}",
            "bone": assigned_bone,
            "instance_id": instance_id,
            "voxel_count": voxel_count,
            "overlap_voxels": overlap_voxels,
            "overlap_fraction": {
                bone: round(overlap_fraction[bone], 8) for bone in BONE_ORDER
            },
            "best_bone": best_bone,
            "best_overlap": round(best_overlap, 8),
            "second_best_bone": second_bone,
            "second_best_overlap": round(second_overlap, 8),
            "margin": round(margin, 8),
            "reason": reason,
        }

    return assignments


def build_assignment_report(
    instance_path: str | Path,
    totalseg_dir: str | Path,
    *,
    config: AssignmentConfig | None = None,
    chunk_depth: int = 128,
    affine_atol: float = 1e-4,
) -> dict[str, Any]:
    """Build the complete JSON-serializable postprocessing report."""

    cfg = config or AssignmentConfig()
    cfg.validate()
    discovered = discover_totalsegmentator_masks(totalseg_dir)
    instance_counts, overlap_counts = compute_nifti_overlap_counts(
        instance_path,
        discovered,
        chunk_depth=chunk_depth,
        affine_atol=affine_atol,
    )
    assignments = assign_from_overlap_counts(instance_counts, overlap_counts, cfg)

    class_counts = Counter(item["bone"] for item in assignments.values())
    return {
        "schema_version": 1,
        "method": "totalsegmentator_fragment_overlap",
        "instance_image": str(Path(instance_path).absolute()),
        "totalsegmentator_directory": str(Path(totalseg_dir).absolute()),
        "parameters": asdict(cfg) | {
            "chunk_depth": chunk_depth,
            "affine_atol": affine_atol,
        },
        "mask_files": {
            bone: [str(path.absolute()) for path in discovered[bone]] for bone in BONE_ORDER
        },
        "missing_bone_masks": [bone for bone in BONE_ORDER if not discovered[bone]],
        "summary": {
            "total_instances": len(assignments),
            "assigned_instances": len(assignments) - class_counts["unknown"],
            "class_counts": {
                bone: int(class_counts[bone]) for bone in (*BONE_ORDER, "unknown")
            },
        },
        "instances": assignments,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign fragment instance IDs to tibia/fibula/femur/patella using "
            "TotalSegmentator mask overlap. The instance NIfTI is not modified."
        )
    )
    parser.add_argument("--instances", required=True, type=Path, help="Fragment instance NIfTI")
    parser.add_argument(
        "--totalseg-dir",
        required=True,
        type=Path,
        help="Directory containing TotalSegmentator bone masks",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output instance_classes.json")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=128,
        help="Number of z slices processed together (default: 128)",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    report = build_assignment_report(
        args.instances,
        args.totalseg_dir,
        config=AssignmentConfig(
            min_overlap=args.min_overlap,
            min_margin=args.min_margin,
        ),
        chunk_depth=args.chunk_depth,
        affine_atol=args.affine_atol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    summary = report["summary"]
    print(
        f"Assigned {summary['assigned_instances']}/{summary['total_instances']} fragments; "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
