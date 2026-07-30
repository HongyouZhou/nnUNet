from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CORTICAL_CONTINUITY_SCHEMA_VERSION = 1
SCHEMA_PLANS_KEY = "cortical_continuity_head_schema"
_COORDINATE_ORDER = "zyx"
AXIAL_19_DIRECTION_SET = "axial19"
DENSE_39_DIRECTION_SET = "dense39"
SUPPORTED_DIRECTION_SETS = (AXIAL_19_DIRECTION_SET, DENSE_39_DIRECTION_SET)


@dataclass(frozen=True)
class FlatHead:
    """One named interval in a flat network output tensor."""

    name: str
    start: int
    stop: int
    activation: str

    @property
    def channels(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FlatHead":
        return cls(
            name=str(value["name"]),
            start=int(value["start"]),
            stop=int(value["stop"]),
            activation=str(value["activation"]),
        )


@dataclass(frozen=True)
class AffinityOffset:
    """A directed edge offset in preprocessed array order (z, y, x)."""

    voxel_offset_zyx: tuple[int, int, int]
    physical_offset_mm_zyx: tuple[float, float, float]
    family: str
    requested_distance_mm: float | None

    @property
    def actual_distance_mm(self) -> float:
        return math.sqrt(sum(i * i for i in self.physical_offset_mm_zyx))

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel_offset_zyx": list(self.voxel_offset_zyx),
            "physical_offset_mm_zyx": list(self.physical_offset_mm_zyx),
            "family": self.family,
            "requested_distance_mm": self.requested_distance_mm,
            "actual_distance_mm": self.actual_distance_mm,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AffinityOffset":
        voxel_offset = tuple(int(i) for i in value["voxel_offset_zyx"])
        physical_offset = tuple(float(i) for i in value["physical_offset_mm_zyx"])
        if len(voxel_offset) != 3 or len(physical_offset) != 3:
            raise ValueError("Affinity offsets must have exactly three z-y-x components")
        requested = value.get("requested_distance_mm")
        return cls(
            voxel_offset_zyx=voxel_offset,
            physical_offset_mm_zyx=physical_offset,
            family=str(value["family"]),
            requested_distance_mm=None if requested is None else float(requested),
        )


@dataclass(frozen=True)
class CorticalContinuityHeadSchema:
    """Versioned contract for a flat ``C + A`` network output.

    ``cortex`` is one sigmoid logit. ``affinity`` has one sigmoid logit per
    directed offset. Activations are intentionally applied only after
    sliding-window and fold logits have been averaged.
    """

    version: int
    spacing_mm_zyx: tuple[float, float, float]
    heads: tuple[FlatHead, ...]
    affinity_offsets: tuple[AffinityOffset, ...]
    direction_set: str = AXIAL_19_DIRECTION_SET
    coordinate_order: str = _COORDINATE_ORDER
    directional_affinity_mirror_tta: bool = False

    def __post_init__(self) -> None:
        if self.version != CORTICAL_CONTINUITY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cortical-continuity schema version {self.version}; "
                f"expected {CORTICAL_CONTINUITY_SCHEMA_VERSION}"
            )
        _validate_spacing(self.spacing_mm_zyx)
        if self.coordinate_order != _COORDINATE_ORDER:
            raise ValueError(f"coordinate_order must be {_COORDINATE_ORDER!r}")
        if self.directional_affinity_mirror_tta:
            raise ValueError(
                "Directional-affinity mirror TTA is not implemented; the schema must keep it disabled"
            )
        if self.direction_set not in SUPPORTED_DIRECTION_SETS:
            raise ValueError(
                f"Unsupported direction_set {self.direction_set!r}; expected one of {SUPPORTED_DIRECTION_SETS}"
            )

        expected_names = ("cortex", "affinity")
        if tuple(i.name for i in self.heads) != expected_names:
            raise ValueError(f"Expected flat heads {expected_names}, got {tuple(i.name for i in self.heads)}")
        cursor = 0
        for head in self.heads:
            if head.start != cursor or head.stop <= head.start:
                raise ValueError("Flat head slices must be positive, contiguous, and start at channel zero")
            if head.activation != "sigmoid":
                raise ValueError(f"Unsupported activation {head.activation!r} for head {head.name!r}")
            cursor = head.stop
        if self.head("cortex").channels != 1:
            raise ValueError("The cortex head must contain exactly one channel")
        if self.head("affinity").channels != len(self.affinity_offsets):
            raise ValueError("Affinity head width does not match affinity_offsets")
        max_offsets = 19 if self.direction_set == AXIAL_19_DIRECTION_SET else 39
        if not 13 <= len(self.affinity_offsets) <= max_offsets:
            raise ValueError(
                f"Schema v1 direction set {self.direction_set!r} requires 13 local offsets "
                f"and at most {max_offsets - 13} unique lifted offsets"
            )

        voxel_offsets = [i.voxel_offset_zyx for i in self.affinity_offsets]
        if len(set(voxel_offsets)) != len(voxel_offsets):
            raise ValueError("Affinity voxel offsets must be unique")
        if any(i == (0, 0, 0) for i in voxel_offsets):
            raise ValueError("The zero affinity offset is invalid")
        if sum(i.family == "local_26_half" for i in self.affinity_offsets) != 13:
            raise ValueError("Schema v1 must contain the 13-edge half of the local 26-neighbourhood")

    @property
    def total_channels(self) -> int:
        return self.heads[-1].stop

    @property
    def num_affinity_channels(self) -> int:
        return len(self.affinity_offsets)

    def head(self, name: str) -> FlatHead:
        for head in self.heads:
            if head.name == name:
                return head
        raise KeyError(name)

    def split(self, flat_logits: Any, channel_axis: int) -> dict[str, Any]:
        """Split NumPy or tensor-like flat logits without applying activation."""

        ndim = int(flat_logits.ndim)
        axis = channel_axis if channel_axis >= 0 else ndim + channel_axis
        if not 0 <= axis < ndim:
            raise ValueError(f"channel_axis {channel_axis} is invalid for a {ndim}D value")
        if int(flat_logits.shape[axis]) != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} flat channels on axis {axis}, "
                f"got {flat_logits.shape[axis]}"
            )

        result: dict[str, Any] = {}
        for head in self.heads:
            index = [slice(None)] * ndim
            index[axis] = slice(head.start, head.stop)
            result[head.name] = flat_logits[tuple(index)]
        return result

    def activate(self, flat_logits: Any, channel_axis: int) -> dict[str, Any]:
        """Split logits and apply the activation declared by each head."""

        split_logits = self.split(flat_logits, channel_axis)
        return {head.name: _sigmoid(split_logits[head.name]) for head in self.heads}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "coordinate_order": self.coordinate_order,
            "direction_set": self.direction_set,
            "spacing_mm_zyx": list(self.spacing_mm_zyx),
            "directional_affinity_mirror_tta": self.directional_affinity_mirror_tta,
            "heads": [i.to_dict() for i in self.heads],
            "affinity_offsets": [i.to_dict() for i in self.affinity_offsets],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorticalContinuityHeadSchema":
        spacing = tuple(float(i) for i in value["spacing_mm_zyx"])
        if len(spacing) != 3:
            raise ValueError("spacing_mm_zyx must have exactly three values")
        return cls(
            version=int(value["version"]),
            coordinate_order=str(value.get("coordinate_order", _COORDINATE_ORDER)),
            direction_set=str(value.get("direction_set", AXIAL_19_DIRECTION_SET)),
            spacing_mm_zyx=spacing,
            directional_affinity_mirror_tta=bool(
                value.get("directional_affinity_mirror_tta", False)
            ),
            heads=tuple(FlatHead.from_dict(i) for i in value["heads"]),
            affinity_offsets=tuple(AffinityOffset.from_dict(i) for i in value["affinity_offsets"]),
        )

    def dumps(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def loads(cls, value: str) -> "CorticalContinuityHeadSchema":
        return cls.from_dict(json.loads(value))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CorticalContinuityHeadSchema":
        return cls.loads(Path(path).read_text(encoding="utf-8"))


def build_cortical_continuity_schema(
    spacing_mm_zyx: Sequence[float],
    lifted_distances_mm: Sequence[float] = (3.0, 6.0),
    direction_set: str = AXIAL_19_DIRECTION_SET,
) -> CorticalContinuityHeadSchema:
    """Build schema v1 with a frozen, named directional-offset profile.

    ``axial19`` is the formal default: 13 local edges plus at most six
    3/6 mm axial lifted edges. ``dense39`` expands both lifted scales to all
    13 half-neighbourhood directions. The latter is reserved for the
    predeclared O2 graph-correctness fallback rather than silent tuning.
    """

    spacing = tuple(float(i) for i in spacing_mm_zyx)
    _validate_spacing(spacing)
    if tuple(float(i) for i in lifted_distances_mm) != (3.0, 6.0):
        raise ValueError("Schema v1 fixes lifted_distances_mm to (3.0, 6.0)")
    if direction_set not in SUPPORTED_DIRECTION_SETS:
        raise ValueError(f"direction_set must be one of {SUPPORTED_DIRECTION_SETS}")

    offsets: list[AffinityOffset] = []
    seen: set[tuple[int, int, int]] = set()
    for offset in _local_half_neighbourhood():
        offsets.append(
            AffinityOffset(
                voxel_offset_zyx=offset,
                physical_offset_mm_zyx=tuple(offset[i] * spacing[i] for i in range(3)),
                family="local_26_half",
                requested_distance_mm=None,
            )
        )
        seen.add(offset)

    for distance_mm in lifted_distances_mm:
        if not math.isfinite(float(distance_mm)) or float(distance_mm) <= 0:
            raise ValueError("Lifted distances must be positive and finite")
        base_directions = (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            if direction_set == AXIAL_19_DIRECTION_SET
            else _local_half_neighbourhood()
        )
        for base_direction in base_directions:
            unit_length_mm = math.sqrt(
                sum((base_direction[i] * spacing[i]) ** 2 for i in range(3))
            )
            voxel_multiplier = max(
                1,
                int(math.floor(float(distance_mm) / unit_length_mm + 0.5)),
            )
            offset = tuple(voxel_multiplier * i for i in base_direction)
            if offset in seen:
                continue
            offsets.append(
                AffinityOffset(
                    voxel_offset_zyx=offset,
                    physical_offset_mm_zyx=tuple(offset[i] * spacing[i] for i in range(3)),
                    family=(
                        "lifted_axial"
                        if direction_set == AXIAL_19_DIRECTION_SET
                        else "lifted_dense"
                    ),
                    requested_distance_mm=float(distance_mm),
                )
            )
            seen.add(offset)

    cortex = FlatHead(name="cortex", start=0, stop=1, activation="sigmoid")
    affinity = FlatHead(
        name="affinity",
        start=1,
        stop=1 + len(offsets),
        activation="sigmoid",
    )
    return CorticalContinuityHeadSchema(
        version=CORTICAL_CONTINUITY_SCHEMA_VERSION,
        spacing_mm_zyx=spacing,
        direction_set=direction_set,
        heads=(cortex, affinity),
        affinity_offsets=tuple(offsets),
    )


def schema_from_plans(plans: Mapping[str, Any]) -> CorticalContinuityHeadSchema:
    try:
        value = plans[SCHEMA_PLANS_KEY]
    except KeyError as exc:
        raise KeyError(
            f"Plans do not contain required {SCHEMA_PLANS_KEY!r}; "
            "build and freeze the cortical-continuity schema before training"
        ) from exc
    return CorticalContinuityHeadSchema.from_dict(value)


def _local_half_neighbourhood() -> tuple[tuple[int, int, int], ...]:
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        first_nonzero = next(i for i in offset if i != 0)
        if first_nonzero > 0:
            offsets.append(offset)
    offsets.sort()
    if len(offsets) != 13:
        raise AssertionError("Expected 13 offsets in the half 26-neighbourhood")
    return tuple(offsets)


def _validate_spacing(spacing: Sequence[float]) -> None:
    if len(spacing) != 3:
        raise ValueError("spacing_mm_zyx must have exactly three values")
    if any(not math.isfinite(float(i)) or float(i) <= 0 for i in spacing):
        raise ValueError("spacing_mm_zyx values must be positive and finite")


def _sigmoid(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
        work = array.astype(dtype, copy=False)
        result = np.empty_like(work)
        positive = work >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-work[positive]))
        exp_value = np.exp(work[~positive])
        result[~positive] = exp_value / (1.0 + exp_value)
        return result
    sigmoid_method = getattr(value, "sigmoid", None)
    if callable(sigmoid_method):
        return sigmoid_method()
    raise TypeError(f"Cannot apply sigmoid to {type(value).__name__}")
