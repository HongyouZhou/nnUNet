from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


AffinityKind = Literal["logits", "probabilities"]


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")


def _validate_spacing(spacing_zyx: np.ndarray) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,):
        raise ValueError(f"spacing_zyx must have shape (3,), got {spacing.shape}")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain finite positive values")
    return spacing


@dataclasses.dataclass(frozen=True)
class ContinuityInputs:
    """Dense model evidence on one common working grid."""

    provisional_instances: np.ndarray
    cortex_probability: np.ndarray
    affinity: np.ndarray
    affinity_offsets_zyx: np.ndarray
    spacing_zyx: np.ndarray
    affinity_kind: AffinityKind = "logits"
    rim_probability: Optional[np.ndarray] = None
    volume_probability: Optional[np.ndarray] = None

    def validated(self) -> "ContinuityInputs":
        instances = np.asarray(self.provisional_instances)
        cortex = np.asarray(self.cortex_probability)
        affinity = np.asarray(self.affinity)
        offsets = np.asarray(self.affinity_offsets_zyx)
        spacing = _validate_spacing(self.spacing_zyx)

        if instances.ndim != 3:
            raise ValueError(
                f"provisional_instances must be 3D, got shape {instances.shape}"
            )
        if not np.issubdtype(instances.dtype, np.integer):
            raise ValueError("provisional_instances must have an integer dtype")
        if np.any(instances < 0):
            raise ValueError("provisional_instances must be non-negative")
        if cortex.shape != instances.shape:
            raise ValueError(
                "cortex_probability must have the same 3D shape as "
                "provisional_instances"
            )
        if affinity.ndim != 4 or affinity.shape[1:] != instances.shape:
            raise ValueError(
                "affinity must have shape (E, *provisional_instances.shape)"
            )
        if offsets.shape != (affinity.shape[0], 3):
            raise ValueError(
                f"affinity_offsets_zyx must have shape ({affinity.shape[0]}, 3)"
            )
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("affinity_offsets_zyx must contain integer offsets")
        if np.any(np.all(offsets == 0, axis=1)):
            raise ValueError("zero affinity offsets are not allowed")
        if len({tuple(int(v) for v in row) for row in offsets}) != len(offsets):
            raise ValueError("affinity_offsets_zyx must not contain duplicates")
        if self.affinity_kind not in ("logits", "probabilities"):
            raise ValueError("affinity_kind must be 'logits' or 'probabilities'")
        _require_finite("cortex_probability", cortex)
        _require_finite("affinity", affinity)
        if np.any((cortex < 0) | (cortex > 1)):
            raise ValueError("cortex_probability must be in [0, 1]")
        if self.affinity_kind == "probabilities" and np.any(
            (affinity < 0) | (affinity > 1)
        ):
            raise ValueError("probability affinities must be in [0, 1]")

        for name, optional in (
            ("rim_probability", self.rim_probability),
            ("volume_probability", self.volume_probability),
        ):
            if optional is None:
                continue
            array = np.asarray(optional)
            if array.shape != instances.shape:
                raise ValueError(f"{name} must have the same shape as instances")
            _require_finite(name, array)
            if np.any((array < 0) | (array > 1)):
                raise ValueError(f"{name} must be in [0, 1]")

        return ContinuityInputs(
            provisional_instances=instances,
            cortex_probability=cortex.astype(np.float32, copy=False),
            affinity=affinity.astype(np.float32, copy=False),
            affinity_offsets_zyx=offsets.astype(np.int16, copy=False),
            spacing_zyx=spacing,
            affinity_kind=self.affinity_kind,
            rim_probability=(
                None
                if self.rim_probability is None
                else np.asarray(self.rim_probability, dtype=np.float32)
            ),
            volume_probability=(
                None
                if self.volume_probability is None
                else np.asarray(self.volume_probability, dtype=np.float32)
            ),
        )


@dataclasses.dataclass(frozen=True)
class InstanceDecision:
    instance_id: int
    status: Literal["unchanged", "accepted", "abstained"]
    reason: str
    cortical_voxels: int
    raw_clusters: int
    output_ids: tuple[int, ...]
    normalized_energy_gain: float = 0.0
    graph_edges: int = 0
    unseeded_volume_components: int = 0
    diagnostic_detail: Optional[str] = None
    split_features: Optional[dict[str, float]] = None
    split_probability: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["output_ids"] = list(self.output_ids)
        return value


@dataclasses.dataclass(frozen=True)
class ContinuityResult:
    full_instances: np.ndarray
    cortical_instances: np.ndarray
    raw_cortical_clusters: np.ndarray
    decisions: tuple[InstanceDecision, ...]

    def validate_against(self, provisional_instances: np.ndarray) -> None:
        provisional = np.asarray(provisional_instances)
        for name, array in (
            ("full_instances", self.full_instances),
            ("cortical_instances", self.cortical_instances),
            ("raw_cortical_clusters", self.raw_cortical_clusters),
        ):
            if np.asarray(array).shape != provisional.shape:
                raise ValueError(f"{name} shape does not match provisional instances")
        if not np.array_equal(self.full_instances > 0, provisional > 0):
            raise ValueError(
                "full_instances must preserve the exact provisional foreground support"
            )


def save_inputs_npz(path: str | Path, inputs: ContinuityInputs) -> None:
    data = inputs.validated()
    payload: dict[str, np.ndarray] = {
        "provisional_instances": data.provisional_instances,
        "cortex_probability": data.cortex_probability,
        "affinity": data.affinity,
        "affinity_offsets_zyx": data.affinity_offsets_zyx,
        "spacing_zyx": data.spacing_zyx,
        "affinity_kind": np.asarray(data.affinity_kind),
    }
    if data.rim_probability is not None:
        payload["rim_probability"] = data.rim_probability
    if data.volume_probability is not None:
        payload["volume_probability"] = data.volume_probability
    np.savez_compressed(Path(path), **payload)


def load_inputs_npz(path: str | Path) -> ContinuityInputs:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "provisional_instances",
            "cortex_probability",
            "affinity",
            "affinity_offsets_zyx",
            "spacing_zyx",
            "affinity_kind",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"input bundle is missing arrays: {sorted(missing)}")
        result = ContinuityInputs(
            provisional_instances=payload["provisional_instances"],
            cortex_probability=payload["cortex_probability"],
            affinity=payload["affinity"],
            affinity_offsets_zyx=payload["affinity_offsets_zyx"],
            spacing_zyx=payload["spacing_zyx"],
            affinity_kind=str(payload["affinity_kind"].item()),
            rim_probability=(
                payload["rim_probability"]
                if "rim_probability" in payload.files
                else None
            ),
            volume_probability=(
                payload["volume_probability"]
                if "volume_probability" in payload.files
                else None
            ),
        )
    return result.validated()


def save_result(
    arrays_path: str | Path,
    diagnostics_path: str | Path,
    result: ContinuityResult,
) -> None:
    np.savez_compressed(
        Path(arrays_path),
        full_instances=result.full_instances,
        cortical_instances=result.cortical_instances,
        raw_cortical_clusters=result.raw_cortical_clusters,
    )
    diagnostics = {
        "schema_version": 1,
        "decisions": [decision.to_dict() for decision in result.decisions],
    }
    Path(diagnostics_path).write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
