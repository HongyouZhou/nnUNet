"""Build overlap-aware Dataset778 cortical-continuity sidecars.

The command is a dry run unless ``--execute`` is supplied.  An executed build
requires an explicit, non-existing output path and is committed atomically from
a staging directory.  Source files are opened read-only and their declared
SHA-256 hashes are verified before materialisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import (
    NiftiGeometry,
    NrrdLayout,
    inspect_nifti,
    inspect_nrrd,
    materialize_image,
    sha256_file,
    write_json,
    write_nifti_crop_like,
)
from .schema import (
    DATASET_ID,
    DATASET_NAME,
    MANIFEST_SCHEMA,
    N_FOLDS,
    SCHEMA_VERSION,
    SPLIT_RANDOM_STATE,
    BuildConfig,
    attach_manifest_hash,
    canonical_json_hash,
    validate_manifest,
)
from .splits import SplitResult, make_stratified_fivefold
from .targets import CaseTargets, cortical_base_name, derive_case_targets


DEFAULT_SOURCE = Path(
    "/home/hongyou/dev/data/segmentation/derived/"
    "cortical_dataset_68_final_20260728"
)
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_EXPOSED_CASES = frozenset(
    """
    15 16 18 38 41 48 95 103 118 144 154 224 241 255 267 272 368 411
    426 497 624 676 1385 1602 1753 1918 2103 2196 2348 2394 2799 3058
    3161 5069 5097
    """.split()
)
_FULL_VOLUME_METRIC_EXCLUSIONS = frozenset(
    {("38", "seg_tibia_right_8")}
)


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    nnunet_id: str
    cohort: str
    ct_path: Path
    fragment_path: Path
    cortical_path: Path
    declared_hashes: Mapping[str, str]
    fragment_count: int
    cortical_count: int
    fragment_without_cortical: tuple[str, ...]
    ct_geometry: NiftiGeometry
    fragment_layout: NrrdLayout
    cortical_layout: NrrdLayout
    expected_voxels: Mapping[tuple[str, str], int]

    def split_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "nnunet_id": self.nnunet_id,
            "patient_id": self.case_id,
            "group_id": self.case_id,
            "cohort": self.cohort,
            "spacing_xyz_mm": list(self.ct_geometry.spacing_xyz_mm),
            "fragment_count": self.fragment_count,
            "cortical_count": self.cortical_count,
            "fragment_without_cortical_count": len(
                self.fragment_without_cortical
            ),
            "cortical_layer_count": self.cortical_layout.layer_count,
        }


@dataclass(frozen=True)
class SourceIndex:
    root: Path
    cases: tuple[SourceCase, ...]
    source_hashes: Mapping[str, str]
    dataset_summary: Mapping[str, Any]
    validation_report: Mapping[str, Any]


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _inside_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source manifest path escapes dataset root: {relative}") from error
    return candidate


def _nnunet_id(case_id: str) -> str:
    if not _SAFE_CASE_ID.fullmatch(case_id):
        raise ValueError(f"case ID is unsafe for a file stem: {case_id!r}")
    return f"charite_{case_id}"


def load_source_index(source_root: Path) -> SourceIndex:
    root = source_root.expanduser().resolve(strict=True)
    required_index = (
        "manifest.csv",
        "labels.csv",
        "dataset_summary.json",
        "validation_report.json",
    )
    missing = [name for name in required_index if not (root / name).is_file()]
    if missing:
        raise ValueError(f"source dataset is missing index files: {missing}")
    source_hashes = {
        name.replace(".", "_") + "_sha256": sha256_file(root / name)
        for name in required_index
    }
    # Preserve stable public field names for the manifest contract.
    source_hashes = {
        "manifest_csv_sha256": sha256_file(root / "manifest.csv"),
        "labels_csv_sha256": sha256_file(root / "labels.csv"),
        "dataset_summary_sha256": sha256_file(root / "dataset_summary.json"),
        "validation_report_sha256": sha256_file(root / "validation_report.json"),
    }
    summary = _read_json(root / "dataset_summary.json")
    validation = _read_json(root / "validation_report.json")
    if validation.get("status") != "PASS":
        raise ValueError("source validation_report.json is not PASS")

    manifest_rows = _read_csv(root / "manifest.csv")
    label_rows = _read_csv(root / "labels.csv")
    if int(summary.get("cases", len(manifest_rows))) != len(manifest_rows):
        raise ValueError("source dataset_summary case count differs from manifest.csv")
    if len({row["case_id"] for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("source manifest contains duplicate case IDs")

    labels_by_case: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        labels_by_case.setdefault(row["case_id"], []).append(row)
    cases: list[SourceCase] = []
    for row in manifest_rows:
        case_id = row["case_id"]
        rows = labels_by_case.get(case_id, [])
        fragment_names = {
            item["name"] for item in rows if item["role"] == "fragment"
        }
        cortical_names = {
            item["name"] for item in rows if item["role"] == "cortical"
        }
        cortical_bases = {cortical_base_name(name) for name in cortical_names}
        no_cortex = tuple(sorted(fragment_names - cortical_bases))
        expected_voxels = {
            (item["role"], item["name"]): int(item["voxels_in_dataset"])
            for item in rows
        }
        if len(expected_voxels) != len(rows):
            raise ValueError(f"{case_id} labels.csv contains duplicate role/name rows")

        ct_path = _inside_root(root, row["ct_file"])
        fragment_path = _inside_root(root, row["fragment_file"])
        cortical_path = _inside_root(root, row["cortical_file"])
        ct_geometry = inspect_nifti(ct_path)
        fragment_layout = inspect_nrrd(fragment_path)
        cortical_layout = inspect_nrrd(cortical_path)
        if (
            fragment_layout.spatial_shape_xyz != ct_geometry.shape_xyz
            or cortical_layout.spatial_shape_xyz != ct_geometry.shape_xyz
        ):
            raise ValueError(f"{case_id} NRRD shape differs from CT shape")
        fragment_count = int(row["fragment_segments"])
        cortical_count = int(row["cortical_segments"])
        if len(fragment_layout.segments) != fragment_count:
            raise ValueError(f"{case_id} fragment header count differs from manifest")
        if len(cortical_layout.segments) != cortical_count:
            raise ValueError(f"{case_id} cortical header count differs from manifest")
        if len(fragment_names) != fragment_count or len(cortical_names) != cortical_count:
            raise ValueError(f"{case_id} labels.csv counts differ from manifest")
        cases.append(
            SourceCase(
                case_id=case_id,
                nnunet_id=_nnunet_id(case_id),
                cohort=row["cohort"],
                ct_path=ct_path,
                fragment_path=fragment_path,
                cortical_path=cortical_path,
                declared_hashes={
                    "ct": row["ct_sha256"],
                    "fragment": row["fragment_sha256"],
                    "cortical": row["cortical_sha256"],
                },
                fragment_count=fragment_count,
                cortical_count=cortical_count,
                fragment_without_cortical=no_cortex,
                ct_geometry=ct_geometry,
                fragment_layout=fragment_layout,
                cortical_layout=cortical_layout,
                expected_voxels=expected_voxels,
            )
        )
    return SourceIndex(
        root=root,
        cases=tuple(sorted(cases, key=lambda case: case.case_id)),
        source_hashes=source_hashes,
        dataset_summary=summary,
        validation_report=validation,
    )


def _select_cases(
    cases: Sequence[SourceCase], requested: Sequence[str] | None
) -> tuple[SourceCase, ...]:
    if not requested:
        return tuple(cases)
    requested_set = set(requested)
    if len(requested_set) != len(requested):
        raise ValueError("--case contains duplicates")
    by_id = {case.case_id: case for case in cases}
    missing = sorted(requested_set - set(by_id))
    if missing:
        raise ValueError(f"requested case IDs are absent: {missing}")
    return tuple(case for case in cases if case.case_id in requested_set)


def plan_build(
    source: SourceIndex,
    config: BuildConfig,
    *,
    requested_cases: Sequence[str] | None = None,
) -> tuple[dict[str, Any], SplitResult, tuple[SourceCase, ...]]:
    split = make_stratified_fivefold([case.split_record() for case in source.cases])
    selected = _select_cases(source.cases, requested_cases)
    plan = {
        "mode": "dry_run",
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "source": str(source.root),
        "source_cases": len(source.cases),
        "selected_cases": len(selected),
        "partial_build": len(selected) != len(source.cases),
        "cohort_counts": dict(
            sorted(Counter(case.cohort for case in selected).items())
        ),
        "fragment_count": sum(case.fragment_count for case in selected),
        "cortical_count": sum(case.cortical_count for case in selected),
        "fragment_without_cortical_count": sum(
            len(case.fragment_without_cortical) for case in selected
        ),
        "multilayer_cortical_cases": sum(
            case.cortical_layout.layer_count > 1 for case in selected
        ),
        "target_config": config.as_dict(),
        "split": split.audit,
        "writes_source_data": False,
        "execute_requires_explicit_output": True,
    }
    return plan, split, selected


def _verify_source_case(case: SourceCase) -> None:
    observed = {
        "ct": sha256_file(case.ct_path),
        "fragment": sha256_file(case.fragment_path),
        "cortical": sha256_file(case.cortical_path),
    }
    mismatches = {
        role: {"declared": case.declared_hashes[role], "observed": digest}
        for role, digest in observed.items()
        if digest != case.declared_hashes[role]
    }
    if mismatches:
        raise ValueError(f"{case.case_id} source SHA-256 mismatch: {mismatches}")


def _validate_target_counts(case: SourceCase, targets: CaseTargets) -> None:
    if targets.audit["fragment_count"] != case.fragment_count:
        raise ValueError(f"{case.case_id} generated fragment count differs")
    if targets.audit["cortical_count"] != case.cortical_count:
        raise ValueError(f"{case.case_id} generated cortical count differs")
    for record in targets.instance_records:
        expected_fragment = case.expected_voxels[
            ("fragment", record.fragment_name)
        ]
        if record.fragment_voxels != expected_fragment:
            raise ValueError(
                f"{case.case_id} {record.fragment_name} fragment voxel count "
                f"{record.fragment_voxels} != {expected_fragment}"
            )
        if record.cortical_name is not None:
            expected_cortical = case.expected_voxels[
                ("cortical", record.cortical_name)
            ]
            if record.cortical_voxels != expected_cortical:
                raise ValueError(
                    f"{case.case_id} {record.cortical_name} cortical voxel count "
                    f"{record.cortical_voxels} != {expected_cortical}"
                )


def _write_case(
    staging: Path,
    case: SourceCase,
    targets: CaseTargets,
    *,
    fold: int,
    config: BuildConfig,
    image_mode: str,
) -> dict[str, Any]:
    stem = case.nnunet_id
    relative_paths = {
        "image": f"imagesTr/{stem}_0000.nii.gz",
        "labels": f"labelsTr/{stem}.nii.gz",
        "cortical_instances": f"corticalInstancesTr/{stem}.nii.gz",
        "cortical_overlap": f"corticalOverlapTr/{stem}.nii.gz",
        "fragment_instances": f"fragmentInstancesTr/{stem}.nii.gz",
        "fragment_overlap": f"fragmentOverlapTr/{stem}.nii.gz",
        "valid_mask": f"validMasksTr/{stem}.nii.gz",
        "support": f"supportTr/{stem}.nii.gz",
        "rim_contact": f"rimContactTr/{stem}.nii.gz",
        "metadata": f"metadataTr/{stem}.json",
    }
    image_method = materialize_image(
        case.ct_path, staging / relative_paths["image"], mode=image_mode
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["labels"],
        targets.separator,
        targets.bbox_xyz,
        dtype_name="uint8",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["cortical_instances"],
        targets.cortical_instances,
        targets.bbox_xyz,
        dtype_name="int16",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["cortical_overlap"],
        targets.cortical_overlap,
        targets.bbox_xyz,
        dtype_name="uint8",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["fragment_instances"],
        targets.fragment_instances,
        targets.bbox_xyz,
        dtype_name="int16",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["fragment_overlap"],
        targets.fragment_overlap,
        targets.bbox_xyz,
        dtype_name="uint8",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["valid_mask"],
        targets.validity,
        targets.bbox_xyz,
        dtype_name="uint8",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["support"],
        targets.fragment_support,
        targets.bbox_xyz,
        dtype_name="uint8",
    )
    write_nifti_crop_like(
        case.ct_geometry,
        staging / relative_paths["rim_contact"],
        targets.rim_contact,
        targets.bbox_xyz,
        dtype_name="uint8",
        fill_value=2,
    )
    output_hashes = {
        role: sha256_file(staging / relative)
        for role, relative in relative_paths.items()
        if role != "metadata"
    }
    excluded_instances = [
        record.fragment_name
        for record in targets.instance_records
        if (case.case_id, record.fragment_name)
        in _FULL_VOLUME_METRIC_EXCLUSIONS
    ]
    instance_metadata = []
    for record in targets.instance_records:
        item = record.as_dict()
        item["exclude_full_volume_metrics"] = (
            case.case_id,
            record.fragment_name,
        ) in _FULL_VOLUME_METRIC_EXCLUSIONS
        instance_metadata.append(item)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "nnunet_id": stem,
        "patient_id": case.case_id,
        "group_id": case.case_id,
        "cohort": case.cohort,
        "fold": fold,
        "development_exposed": case.case_id in _EXPOSED_CASES,
        "shape_xyz": list(case.ct_geometry.shape_xyz),
        "spacing_xyz_mm": list(case.ct_geometry.spacing_xyz_mm),
        "crop_bbox_xyz": [list(axis) for axis in targets.bbox_xyz],
        "target_config": config.as_dict(),
        "valid_mask_bits": {
            "semantic_c": 1,
            "cortical_relation": 2,
            "rim_contact": 4,
        },
        "label_values": {
            "background": 0,
            "cortical_body": 1,
            "cortical_separator": 2,
            "ignore": 3,
        },
        "rim_contact_values": {"negative": 0, "positive": 1, "ignore": 2},
        "fragment_without_cortical": list(case.fragment_without_cortical),
        "instances": instance_metadata,
        "exclude_full_volume_metrics": excluded_instances,
        "audit": dict(targets.audit),
        "source_files": {
            "ct": {
                "path": str(case.ct_path),
                "sha256": case.declared_hashes["ct"],
            },
            "fragment": {
                "path": str(case.fragment_path),
                "sha256": case.declared_hashes["fragment"],
            },
            "cortical": {
                "path": str(case.cortical_path),
                "sha256": case.declared_hashes["cortical"],
            },
        },
        "sidecars": {
            role: {"path": relative_paths[role], "sha256": digest}
            for role, digest in output_hashes.items()
        },
        "image_materialization": {
            "method": image_method,
            "shares_source_storage": image_method in {"hardlink", "symlink"},
            "risk": (
                "downstream in-place writes may mutate or follow the source"
                if image_method in {"hardlink", "symlink"}
                else None
            ),
        },
    }
    write_json(staging / relative_paths["metadata"], metadata)
    metadata_hash = sha256_file(staging / relative_paths["metadata"])
    output_hashes["metadata"] = metadata_hash
    return {
        "case_id": case.case_id,
        "nnunet_id": stem,
        "patient_id": case.case_id,
        "group_id": case.case_id,
        "cohort": case.cohort,
        "fold": fold,
        "development_exposed": case.case_id in _EXPOSED_CASES,
        "spacing_xyz_mm": list(case.ct_geometry.spacing_xyz_mm),
        "shape_xyz": list(case.ct_geometry.shape_xyz),
        "fragment_count": case.fragment_count,
        "cortical_count": case.cortical_count,
        "fragment_without_cortical_count": len(
            case.fragment_without_cortical
        ),
        "exclude_full_volume_metrics": excluded_instances,
        "source_files": {
            "ct": {
                "path": str(case.ct_path.relative_to(case.ct_path.parents[3]))
                if len(case.ct_path.parents) > 3
                else case.ct_path.name,
                "sha256": case.declared_hashes["ct"],
            },
            "fragment": {
                "path": case.fragment_path.name,
                "sha256": case.declared_hashes["fragment"],
            },
            "cortical": {
                "path": case.cortical_path.name,
                "sha256": case.declared_hashes["cortical"],
            },
        },
        "sidecars": {
            role: {"path": relative_paths[role], "sha256": digest}
            for role, digest in output_hashes.items()
        },
        "audit": dict(targets.audit),
    }


def _filtered_splits(
    split: SplitResult, selected: Sequence[SourceCase]
) -> list[dict[str, list[str]]]:
    identifiers = {case.nnunet_id for case in selected}
    return [
        {
            "train": [
                item for item in record["train"] if item in identifiers
            ],
            "val": [item for item in record["val"] if item in identifiers],
        }
        for record in split.splits_final
    ]


def execute_build(
    source: SourceIndex,
    selected: Sequence[SourceCase],
    split: SplitResult,
    config: BuildConfig,
    output: Path,
    *,
    image_mode: str,
    distance_backend: str,
    chunk_spatial_voxels: int,
) -> Mapping[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    try:
        output.relative_to(source.root)
    except ValueError:
        pass
    else:
        raise ValueError("output must not be inside the immutable source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    index_before = dict(source.source_hashes)
    try:
        for directory in (
            "imagesTr",
            "labelsTr",
            "corticalInstancesTr",
            "corticalOverlapTr",
            "fragmentInstancesTr",
            "fragmentOverlapTr",
            "validMasksTr",
            "supportTr",
            "rimContactTr",
            "metadataTr",
        ):
            (staging / directory).mkdir()

        case_records: list[dict[str, Any]] = []
        aggregate: Counter[str] = Counter()
        for position, case in enumerate(selected, start=1):
            print(
                f"[{position:02d}/{len(selected):02d}] building {case.case_id}",
                flush=True,
            )
            _verify_source_case(case)
            targets = derive_case_targets(
                case.fragment_layout,
                case.cortical_layout,
                case.ct_geometry.spacing_xyz_mm,
                config,
                distance_backend=distance_backend,
                chunk_spatial_voxels=chunk_spatial_voxels,
            )
            _validate_target_counts(case, targets)
            record = _write_case(
                staging,
                case,
                targets,
                fold=split.fold_by_case[case.case_id],
                config=config,
                image_mode=image_mode,
            )
            case_records.append(record)
            aggregate.update(
                {
                    key: int(value)
                    for key, value in targets.audit.items()
                    if isinstance(value, int)
                }
            )

        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels": {
                "background": 0,
                "cortical_body": 1,
                "cortical_separator": 2,
                "ignore": 3,
            },
            "numTraining": len(selected),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIOWithReorient",
            "name": DATASET_NAME,
            "description": (
                "Cortical body/separator semantic labels with overlap-aware "
                "instance, validity, support, and rim/contact sidecars"
            ),
            "continuity_sidecars": {
                "cortical_instances": "corticalInstancesTr",
                "cortical_overlap": "corticalOverlapTr",
                "fragment_instances": "fragmentInstancesTr",
                "fragment_overlap": "fragmentOverlapTr",
                "valid_masks": "validMasksTr",
                "support": "supportTr",
                "rim_contact": "rimContactTr",
            },
        }
        write_json(staging / "dataset.json", dataset_json)
        write_json(
            staging / "splits_final.json", _filtered_splits(split, selected)
        )
        write_json(staging / "manifest.schema.json", MANIFEST_SCHEMA)

        manifest = attach_manifest_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "dataset_name": DATASET_NAME,
                "source": {
                    "root": str(source.root),
                    **source.source_hashes,
                    "source_validation_status": source.validation_report["status"],
                    "source_mutated": False,
                },
                "target_config": config.as_dict(),
                "split": {
                    "n_folds": N_FOLDS,
                    "random_state": SPLIT_RANDOM_STATE,
                    "strategy": split.audit["strategy"],
                    "stratify_by": "cohort",
                    "shuffle": True,
                },
                "partial_build": len(selected) != len(source.cases),
                "cases": case_records,
            }
        )
        validate_manifest(manifest)
        write_json(staging / "continuity_manifest.json", manifest)

        index_after = {
            "manifest_csv_sha256": sha256_file(source.root / "manifest.csv"),
            "labels_csv_sha256": sha256_file(source.root / "labels.csv"),
            "dataset_summary_sha256": sha256_file(
                source.root / "dataset_summary.json"
            ),
            "validation_report_sha256": sha256_file(
                source.root / "validation_report.json"
            ),
        }
        if index_after != index_before:
            raise RuntimeError("source dataset index changed during build")
        audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "dataset_id": DATASET_ID,
            "cases": len(selected),
            "partial_build": len(selected) != len(source.cases),
            "aggregate_targets": dict(sorted(aggregate.items())),
            "source_index_unchanged": True,
            "source_data_written": False,
            "split": split.audit,
            "manifest_content_sha256": manifest["content_sha256"],
            "dataset_json_sha256": sha256_file(staging / "dataset.json"),
            "splits_final_sha256": sha256_file(staging / "splits_final.json"),
        }
        write_json(staging / "audit_report.json", audit)
        os.replace(staging, output)
        return audit
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Dataset778 cortical-continuity targets (dry-run by default)"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        help="explicit non-existing output directory; required with --execute",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="materialize Dataset778; omission performs a read-only dry run",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="optional case ID subset for an explicitly partial smoke build",
    )
    parser.add_argument("--separator-mm", type=float, default=1.0)
    parser.add_argument("--rim-positive-mm", type=float, default=2.0)
    parser.add_argument("--rim-negative-mm", type=float, default=4.0)
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "symlink"),
        default="copy",
        help=(
            "CT materialization mode (default: copy). hardlink/symlink share "
            "source storage and are explicit expert-risk options"
        ),
    )
    parser.add_argument(
        "--distance-backend",
        choices=("auto", "numpy", "scipy"),
        default="auto",
    )
    parser.add_argument(
        "--chunk-spatial-voxels", type=int, default=500_000
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.output is None:
        raise SystemExit("--execute requires an explicit --output")
    if not args.execute and args.output is not None:
        print(
            "note: --output is ignored in default dry-run mode; add --execute to write",
            file=sys.stderr,
        )
    if args.chunk_spatial_voxels <= 0:
        raise SystemExit("--chunk-spatial-voxels must be positive")
    config = BuildConfig(
        separator_mm=args.separator_mm,
        rim_positive_mm=args.rim_positive_mm,
        rim_negative_mm=args.rim_negative_mm,
    )
    source = load_source_index(args.source)
    plan, split, selected = plan_build(
        source, config, requested_cases=args.cases
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    audit = execute_build(
        source,
        selected,
        split,
        config,
        args.output,
        image_mode=args.image_mode,
        distance_backend=args.distance_backend,
        chunk_spatial_voxels=args.chunk_spatial_voxels,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
