"""Versioned manifest and target configuration schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
DATASET_ID = 778
DATASET_NAME = "Dataset778_ChariteCorticalContinuity"
N_FOLDS = 5
SPLIT_RANDOM_STATE = 20260729


@dataclass(frozen=True)
class BuildConfig:
    """Parameters that alter generated supervision.

    Distances are voxel-centre distances in physical millimetres on the source
    grid.  All values are embedded in the dataset manifest and per-case
    metadata so a target cannot be separated from the configuration that
    produced it.
    """

    separator_mm: float = 1.0
    rim_positive_mm: float = 2.0
    rim_negative_mm: float = 4.0
    split_random_state: int = SPLIT_RANDOM_STATE
    n_folds: int = N_FOLDS

    def __post_init__(self) -> None:
        values = (self.separator_mm, self.rim_positive_mm, self.rim_negative_mm)
        if any(value <= 0 for value in values):
            raise ValueError("all physical target thresholds must be positive")
        if not self.separator_mm <= self.rim_positive_mm < self.rim_negative_mm:
            raise ValueError(
                "expected separator_mm <= rim_positive_mm < rim_negative_mm"
            )
        if self.n_folds != N_FOLDS:
            raise ValueError(f"Dataset778 uses exactly {N_FOLDS} folds")
        if self.split_random_state != SPLIT_RANDOM_STATE:
            raise ValueError(
                f"Dataset778 split random_state is fixed at {SPLIT_RANDOM_STATE}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "charite-cortical-continuity-manifest-v1",
    "title": "Charite cortical continuity Dataset778 manifest",
    "type": "object",
    "required": [
        "schema_version",
        "dataset_id",
        "dataset_name",
        "source",
        "target_config",
        "split",
        "cases",
        "content_sha256",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "dataset_id": {"const": DATASET_ID},
        "dataset_name": {"const": DATASET_NAME},
        "source": {
            "type": "object",
            "required": [
                "dataset_summary_sha256",
                "manifest_csv_sha256",
                "labels_csv_sha256",
                "validation_report_sha256",
            ],
        },
        "target_config": {
            "type": "object",
            "required": [
                "separator_mm",
                "rim_positive_mm",
                "rim_negative_mm",
            ],
        },
        "split": {
            "type": "object",
            "required": ["n_folds", "random_state", "strategy"],
        },
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "case_id",
                    "nnunet_id",
                    "patient_id",
                    "group_id",
                    "cohort",
                    "fold",
                    "spacing_xyz_mm",
                    "shape_xyz",
                    "fragment_count",
                    "cortical_count",
                    "source_files",
                    "sidecars",
                ],
            },
        },
        "content_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON encoding used for hashes and on-disk manifests."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without its self-referential ``content_sha256`` field."""

    payload = dict(manifest)
    payload.pop("content_sha256", None)
    return canonical_json_hash(payload)


def attach_manifest_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["content_sha256"] = manifest_content_hash(result)
    return result


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate invariants that JSON Schema alone cannot express.

    This dependency-light validator is intentionally strict and is used by the
    builder before committing its staging directory.
    """

    required = set(MANIFEST_SCHEMA["required"])
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest is missing required fields: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    if manifest["dataset_id"] != DATASET_ID or manifest["dataset_name"] != DATASET_NAME:
        raise ValueError("manifest does not describe Dataset778")

    target_config = _require_mapping(manifest["target_config"], "target_config")
    BuildConfig(
        separator_mm=float(target_config["separator_mm"]),
        rim_positive_mm=float(target_config["rim_positive_mm"]),
        rim_negative_mm=float(target_config["rim_negative_mm"]),
        split_random_state=int(
            target_config.get("split_random_state", SPLIT_RANDOM_STATE)
        ),
        n_folds=int(target_config.get("n_folds", N_FOLDS)),
    )

    split = _require_mapping(manifest["split"], "split")
    if int(split.get("n_folds", -1)) != N_FOLDS:
        raise ValueError(f"split must contain exactly {N_FOLDS} folds")
    if int(split.get("random_state", -1)) != SPLIT_RANDOM_STATE:
        raise ValueError("split random_state differs from the frozen value")

    cases = _require_sequence(manifest["cases"], "cases")
    if not cases:
        raise ValueError("manifest must contain at least one case")
    case_ids: set[str] = set()
    nnunet_ids: set[str] = set()
    groups_by_fold: dict[str, set[int]] = {}
    for index, raw_case in enumerate(cases):
        case = _require_mapping(raw_case, f"cases[{index}]")
        for field in MANIFEST_SCHEMA["properties"]["cases"]["items"]["required"]:
            if field not in case:
                raise ValueError(f"cases[{index}] is missing {field}")
        case_id = str(case["case_id"])
        nnunet_id = str(case["nnunet_id"])
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id {case_id}")
        if nnunet_id in nnunet_ids:
            raise ValueError(f"duplicate nnunet_id {nnunet_id}")
        case_ids.add(case_id)
        nnunet_ids.add(nnunet_id)
        fold = int(case["fold"])
        if not 0 <= fold < N_FOLDS:
            raise ValueError(f"{case_id} has invalid fold {fold}")
        group_id = str(case["group_id"])
        groups_by_fold.setdefault(group_id, set()).add(fold)
        spacing = tuple(float(item) for item in case["spacing_xyz_mm"])
        shape = tuple(int(item) for item in case["shape_xyz"])
        if len(spacing) != 3 or any(item <= 0 for item in spacing):
            raise ValueError(f"{case_id} has invalid spacing")
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValueError(f"{case_id} has invalid shape")
        _require_mapping(case["source_files"], f"{case_id}.source_files")
        _require_mapping(case["sidecars"], f"{case_id}.sidecars")

    leaking = sorted(group for group, folds in groups_by_fold.items() if len(folds) > 1)
    if leaking:
        raise ValueError(f"group IDs assigned to multiple validation folds: {leaking}")

    observed_hash = str(manifest["content_sha256"])
    expected_hash = manifest_content_hash(manifest)
    if observed_hash != expected_hash:
        raise ValueError("manifest content_sha256 does not match canonical content")
