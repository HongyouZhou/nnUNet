"""Read-only audit for a materialized Dataset778 build."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import inspect_nifti, read_nifti_array, sha256_file, write_json
from .schema import DATASET_ID, N_FOLDS, validate_manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_dataset(root: Path, *, verify_values: bool = True) -> Mapping[str, Any]:
    root = root.expanduser().resolve(strict=True)
    manifest_path = root / "continuity_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("continuity_manifest.json is missing")
    manifest = _read_json(manifest_path)
    validate_manifest(manifest)
    if manifest["dataset_id"] != DATASET_ID:
        raise ValueError("audit target is not Dataset778")

    required_top_level = (
        "dataset.json",
        "splits_final.json",
        "manifest.schema.json",
    )
    missing = [name for name in required_top_level if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Dataset778 is missing files: {missing}")
    dataset_json = _read_json(root / "dataset.json")
    if int(dataset_json["numTraining"]) != len(manifest["cases"]):
        raise ValueError("dataset.json numTraining differs from manifest")
    if dataset_json.get("labels", {}).get("ignore") != 3:
        raise ValueError("dataset.json must declare the nnU-Net ignore label as 3")
    splits = _read_json(root / "splits_final.json")
    if not isinstance(splits, list) or len(splits) != N_FOLDS:
        raise ValueError(f"splits_final.json must contain {N_FOLDS} folds")

    expected_ids = {case["nnunet_id"] for case in manifest["cases"]}
    validation_ids: list[str] = []
    for fold, record in enumerate(splits):
        train = set(record["train"])
        validation = set(record["val"])
        if train & validation:
            raise ValueError(f"fold {fold} train and validation overlap")
        if train | validation != expected_ids:
            raise ValueError(f"fold {fold} does not cover every case")
        validation_ids.extend(record["val"])
    if Counter(validation_ids) != Counter({identifier: 1 for identifier in expected_ids}):
        raise ValueError("each case must be validation exactly once")

    aggregate: Counter[str] = Counter()
    for case in manifest["cases"]:
        shapes: set[tuple[int, int, int]] = set()
        for role, artifact in case["sidecars"].items():
            path = (root / artifact["path"]).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"{case['case_id']} sidecar escapes dataset") from error
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"{case['case_id']} {role} SHA-256 mismatch")
            if role != "metadata":
                shapes.add(inspect_nifti(path).shape_xyz)
        if shapes != {tuple(case["shape_xyz"])}:
            raise ValueError(f"{case['case_id']} sidecar grids differ")

        if verify_values:
            label = read_nifti_array(root / case["sidecars"]["labels"]["path"])
            instances = read_nifti_array(
                root / case["sidecars"]["cortical_instances"]["path"]
            )
            overlap = read_nifti_array(
                root / case["sidecars"]["cortical_overlap"]["path"]
            )
            fragment_instances = read_nifti_array(
                root / case["sidecars"]["fragment_instances"]["path"]
            )
            fragment_overlap = read_nifti_array(
                root / case["sidecars"]["fragment_overlap"]["path"]
            )
            valid = read_nifti_array(
                root / case["sidecars"]["valid_mask"]["path"]
            )
            support = read_nifti_array(
                root / case["sidecars"]["support"]["path"]
            )
            if not set(np.unique(label)).issubset({0, 1, 2, 3}):
                raise ValueError(f"{case['case_id']} labelsTr has invalid values")
            cortex = (label == 1) | (label == 2)
            if np.any(cortex & (support == 0)):
                raise ValueError(f"{case['case_id']} cortex lies outside support")
            if np.any((label == 3) & (((valid & 1) != 0) | (support == 0))):
                raise ValueError(
                    f"{case['case_id']} ignore label is not an invalid fragment voxel"
                )
            if np.any((overlap > 0) & (instances != 0)):
                raise ValueError(
                    f"{case['case_id']} overlap voxels received instance ownership"
                )
            if np.any((overlap > 0) & ((valid & 2) != 0)):
                raise ValueError(
                    f"{case['case_id']} overlap voxels have relation supervision"
                )
            if np.any(cortex & ((valid & 1) == 0)):
                raise ValueError(
                    f"{case['case_id']} cortical voxels lack semantic validity"
                )
            if np.any((fragment_overlap > 0) & (fragment_instances != 0)):
                raise ValueError(
                    f"{case['case_id']} fragment-overlap voxels received ownership"
                )
            reconstructed_support = (fragment_instances > 0) | (
                fragment_overlap > 0
            )
            if np.any(reconstructed_support != (support > 0)):
                raise ValueError(
                    f"{case['case_id']} fragment owner/overlap differs from support"
                )
        aggregate.update(
            {
                key: int(value)
                for key, value in case.get("audit", {}).items()
                if isinstance(value, int)
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "dataset_id": DATASET_ID,
        "cases": len(manifest["cases"]),
        "manifest_content_sha256": manifest["content_sha256"],
        "verified_values": verify_values,
        "aggregate_targets": dict(sorted(aggregate.items())),
        "source_data_written": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a Dataset778 build")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional explicit JSON report; omission is read-only",
    )
    parser.add_argument(
        "--skip-value-audit",
        action="store_true",
        help="verify manifests, hashes, and grids without loading voxel payloads",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_dataset(
        args.dataset, verify_values=not args.skip_value_audit
    )
    if args.output is not None:
        write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
