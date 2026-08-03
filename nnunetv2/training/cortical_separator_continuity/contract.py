"""Frozen preprocessing and loss contract for Dataset778 separator pilots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTINUITY_CONTRACT_VERSION = 1
CONTINUITY_PLANS_KEY = "cortical_separator_continuity_prior"
FORMAL_TARGET_SPACING_ZYX = (0.5, 0.5, 0.5)
FORMAL_PREPROCESSOR = "ChariteCorticalSeparatorContinuityPreprocessor"

SEMANTIC_CHANNEL = 0
SUPPORT_CHANNEL = 1
VALIDITY_CHANNEL = 2
NORMAL_SURFACE_CHANNEL = 3
HU_CODE_CHANNEL = 4
INSTANCE_CHANNEL = 5
TARGET_CHANNEL_LAYOUT = (
    "semantic",
    "support",
    "validity",
    "normal_surface",
    "hu_code",
    "cortical_instance_id",
)
TARGET_CHANNELS = len(TARGET_CHANNEL_LAYOUT)

SEMANTIC_VALID_BIT = 1
RELATION_VALID_BIT = 2
SEPARATOR_LABEL = 2
IGNORE_LABEL = 3

HU_MIN = -2048
HU_MAX = 4095
HU_CODE_OFFSET = 2049
HU_PADDING_CODE = 0

SEPARATOR_SAMPLING_KEY = "continuity_separator"
SURFACE_SAMPLING_KEY = "continuity_normal_surface"
SAMPLING_WEIGHTS = {
    "separator": 0.40,
    "surface": 0.40,
    "random": 0.20,
}

CONTINUITY_WEIGHT = 0.1
DENSITY_WEIGHT = 0.1

# One representative from each undirected edge in the 26-neighbourhood,
# followed by axis-aligned 1 mm edges on the frozen 0.5 mm grid.
LOCAL_HALF_OFFSETS_ZYX = (
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
)
LIFTED_OFFSETS_ZYX = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
PAIR_OFFSETS_ZYX = LOCAL_HALF_OFFSETS_ZYX + LIFTED_OFFSETS_ZYX


def path_offsets_for_pair(offset: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Return the discrete path, including both endpoints, for one pair."""

    value = tuple(int(component) for component in offset)
    if value in LOCAL_HALF_OFFSETS_ZYX:
        return ((0, 0, 0), value)
    if value in LIFTED_OFFSETS_ZYX:
        midpoint = tuple(component // 2 for component in value)
        return ((0, 0, 0), midpoint, value)
    raise ValueError(f"Offset is not in the frozen 16-edge contract: {value}")


def encode_hu(hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(hu), HU_MIN, HU_MAX).astype(np.int16, copy=False)
    return (clipped.astype(np.int32) + HU_CODE_OFFSET).astype(np.int16)


def decode_hu_code(code: Any) -> Any:
    return code - HU_CODE_OFFSET


def density_score_from_hu(hu: Any, q25: float, q50: float, q75: float) -> Any:
    scale = max((float(q75) - float(q25)) / 2.0, 1.0)
    value = (hu - float(q50)) / scale
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.sigmoid(value)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DensityCalibration:
    fold: int
    train_cases: tuple[str, ...]
    q25_hu: float
    q50_hu: float
    q75_hu: float
    splits_sha256: str
    plans_sha256: str
    source_manifest_sha256: str
    voxel_count: int
    schema_version: int = CONTINUITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTINUITY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported calibration schema {self.schema_version}")
        if self.fold not in range(5):
            raise ValueError(f"Calibration fold must be 0..4, got {self.fold}")
        if not self.train_cases or len(set(self.train_cases)) != len(self.train_cases):
            raise ValueError("Calibration train_cases must be unique and non-empty")
        if not self.q25_hu <= self.q50_hu <= self.q75_hu:
            raise ValueError("Calibration HU quantiles are not monotonic")
        if self.voxel_count <= 0:
            raise ValueError("Calibration requires at least one cortical voxel")
        for name in ("splits_sha256", "plans_sha256", "source_manifest_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} is not a lowercase SHA-256 digest")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DensityCalibration":
        return cls(
            schema_version=int(value["schema_version"]),
            fold=int(value["fold"]),
            train_cases=tuple(str(item) for item in value["train_cases"]),
            q25_hu=float(value["q25_hu"]),
            q50_hu=float(value["q50_hu"]),
            q75_hu=float(value["q75_hu"]),
            splits_sha256=str(value["splits_sha256"]),
            plans_sha256=str(value["plans_sha256"]),
            source_manifest_sha256=str(value["source_manifest_sha256"]),
            voxel_count=int(value["voxel_count"]),
        )

    @classmethod
    def load(cls, path: Path) -> "DensityCalibration":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"Calibration must contain a JSON object: {path}")
        return cls.from_dict(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fold": self.fold,
            "train_cases": list(self.train_cases),
            "q25_hu": self.q25_hu,
            "q50_hu": self.q50_hu,
            "q75_hu": self.q75_hu,
            "voxel_count": self.voxel_count,
            "splits_sha256": self.splits_sha256,
            "plans_sha256": self.plans_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "density_formula": "sigmoid((HU-Q50)/max((Q75-Q25)/2,1HU))",
            "hu_clip": [HU_MIN, HU_MAX],
        }


def continuity_plans_contract(*, data_identifier: str) -> dict[str, Any]:
    offsets = []
    for index, offset in enumerate(PAIR_OFFSETS_ZYX):
        offsets.append(
            {
                "index": index,
                "kind": "local_26_half" if index < 13 else "lifted_1mm",
                "voxel_offset_zyx": list(offset),
                "path_offsets_zyx": [list(value) for value in path_offsets_for_pair(offset)],
            }
        )
    return {
        "schema_version": CONTINUITY_CONTRACT_VERSION,
        "data_identifier": data_identifier,
        "target_spacing_mm_zyx": list(FORMAL_TARGET_SPACING_ZYX),
        "target_channel_layout": list(TARGET_CHANNEL_LAYOUT),
        "pair_offsets": offsets,
        "pair_target": {"same_instance": 0, "different_instance": 1},
        "pair_validity": {
            "positive_instance_ids_at_both_endpoints": True,
            "relation_valid_bit": RELATION_VALID_BIT,
            "ignore_label": IGNORE_LABEL,
            "lifted_path_must_be_inside_patch": True,
        },
        "path_cut_probability": "max(separator_probability_along_discrete_path)",
        "normal_surface": {
            "kind": "union_of_per_instance_inner_boundaries",
            "separator_exclusion_mm": 2.0,
            "semantic_valid_bit": SEMANTIC_VALID_BIT,
        },
        "density": {
            "formula": "sigmoid((HU-Q50)/max((Q75-Q25)/2,1HU))",
            "pair_formula": "(D(endpoint_i)+D(endpoint_j))/2",
            "candidate_endpoints": "separator|normal_surface",
            "includes_gap_voxels": False,
        },
        "loss": {
            "base": "DC_and_CE_loss",
            "continuity_weight": CONTINUITY_WEIGHT,
            "density_weight": DENSITY_WEIGHT,
            "auxiliary_resolution": "finest_only",
            "class_reduction": "separate_same_and_different_means",
        },
        "sampling": SAMPLING_WEIGHTS,
        "network": {"input_channels": 1, "softmax_classes": 3},
    }


def validate_continuity_plans_contract(plans: Mapping[str, Any], data_identifier: str) -> dict[str, Any]:
    observed = plans.get(CONTINUITY_PLANS_KEY)
    expected = continuity_plans_contract(data_identifier=data_identifier)
    if observed != expected:
        raise RuntimeError(
            f"Plans contain a missing or incompatible {CONTINUITY_PLANS_KEY} contract; "
            "regenerate the independent continuity-prior plans"
        )
    return expected
