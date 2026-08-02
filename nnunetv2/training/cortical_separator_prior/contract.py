"""Versioned density-prior contracts shared by preprocessing and training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PRIOR_CONTRACT_VERSION = 1
PRIOR_PLANS_KEY = "cortical_separator_density_prior"

SEMANTIC_CHANNEL = 0
SUPPORT_CHANNEL = 1
VALIDITY_CHANNEL = 2
NORMAL_SURFACE_CHANNEL = 3
HU_CODE_CHANNEL = 4
TARGET_CHANNELS = 5

HU_MIN = -2048
HU_MAX = 4095
HU_CODE_OFFSET = 2049
HU_PADDING_CODE = 0

SEMANTIC_VALID_BIT = 1
SEPARATOR_LABEL = 2
IGNORE_LABEL = 3

PRIOR_SEPARATOR_KEY = "separator_prior_contact"
PRIOR_SURFACE_KEY = "separator_prior_normal_surface"
PRIOR_SURFACE_HU_KEY = "separator_prior_normal_surface_hu"

CONTROL_SAMPLING_WEIGHTS = {
    "separator": 0.40,
    "surface": 0.40,
    "random": 0.20,
}
PRIOR_SAMPLING_WEIGHTS = {
    "separator": 0.40,
    "surface_low": 0.20,
    "surface_high": 0.20,
    "random": 0.20,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_hu(hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(hu), HU_MIN, HU_MAX).astype(np.int16, copy=False)
    return (clipped.astype(np.int32) + HU_CODE_OFFSET).astype(np.int16)


def decode_hu_code(code: Any) -> Any:
    return code - HU_CODE_OFFSET


def density_score_from_hu(hu: Any, q25: float, q50: float, q75: float) -> Any:
    scale = max((float(q75) - float(q25)) / 2.0, 1.0)
    value = (hu - float(q50)) / scale
    if isinstance(value, np.ndarray):
        value = np.clip(value, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-value))
    try:
        import torch
    except ModuleNotFoundError:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
    if isinstance(value, torch.Tensor):
        return torch.sigmoid(value)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


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
    schema_version: int = PRIOR_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRIOR_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported density calibration schema {self.schema_version}"
            )
        if self.fold not in range(5):
            raise ValueError(f"Density calibration fold must be 0..4, got {self.fold}")
        if not self.train_cases or len(set(self.train_cases)) != len(self.train_cases):
            raise ValueError("Density calibration train_cases must be unique and non-empty")
        if not self.q25_hu <= self.q50_hu <= self.q75_hu:
            raise ValueError("Density calibration HU quantiles are not monotonic")
        if self.voxel_count <= 0:
            raise ValueError("Density calibration requires at least one cortical voxel")
        for name in ("splits_sha256", "plans_sha256", "source_manifest_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
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
            raise ValueError(f"Density calibration must contain an object: {path}")
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


def prior_plans_contract(*, data_identifier: str) -> dict[str, Any]:
    return {
        "schema_version": PRIOR_CONTRACT_VERSION,
        "data_identifier": data_identifier,
        "target_channels": [
            "semantic",
            "fragment_support",
            "validity",
            "normal_surface",
            "hu_code",
        ],
        "hu_code": {
            "minimum_hu": HU_MIN,
            "maximum_hu": HU_MAX,
            "offset": HU_CODE_OFFSET,
            "padding_code": HU_PADDING_CODE,
        },
        "normal_surface": {
            "kind": "fragment_support_inner_boundary",
            "separator_exclusion_mm": 2.0,
            "semantic_valid_bit": SEMANTIC_VALID_BIT,
        },
        "loss": {
            "base": "DC_and_CE_loss",
            "density_auxiliary_weight": 0.5,
            "separator_positive_weight": "1+2D",
            "normal_surface_negative_weight": "1+(1-D)",
            "auxiliary_resolution": "finest_only",
        },
        "sampling": {
            "control": CONTROL_SAMPLING_WEIGHTS,
            "density_prior": PRIOR_SAMPLING_WEIGHTS,
        },
    }
