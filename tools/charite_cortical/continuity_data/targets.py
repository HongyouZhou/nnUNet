"""Overlap-aware cortical continuity target construction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .io import NrrdLayout, Segment, iter_spatial_vectors
from .schema import BuildConfig


SEMANTIC_VALID = np.uint8(1)
RELATION_VALID = np.uint8(2)
RIM_VALID = np.uint8(4)

SEPARATOR_BACKGROUND = np.uint8(0)
SEPARATOR_CORTEX = np.uint8(1)
SEPARATOR_CONTACT = np.uint8(2)
SEPARATOR_IGNORE = np.uint8(3)

RIM_NEGATIVE = np.uint8(0)
RIM_POSITIVE = np.uint8(1)
RIM_IGNORE = np.uint8(2)

_NATURAL_TOKEN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class InstanceRecord:
    local_id: int
    fragment_name: str
    cortical_name: str | None
    fragment_layer: int
    fragment_label_value: int
    cortical_layer: int | None
    cortical_label_value: int | None
    fragment_voxels: int
    cortical_voxels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_id": self.local_id,
            "fragment_name": self.fragment_name,
            "cortical_name": self.cortical_name,
            "fragment_layer": self.fragment_layer,
            "fragment_label_value": self.fragment_label_value,
            "cortical_layer": self.cortical_layer,
            "cortical_label_value": self.cortical_label_value,
            "fragment_voxels": self.fragment_voxels,
            "cortical_voxels": self.cortical_voxels,
        }


@dataclass(frozen=True)
class CaseTargets:
    bbox_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    cortex_union: np.ndarray[Any, np.dtype[np.uint8]]
    cortical_instances: np.ndarray[Any, np.dtype[np.int16]]
    cortical_overlap: np.ndarray[Any, np.dtype[np.uint8]]
    fragment_instances: np.ndarray[Any, np.dtype[np.int16]]
    fragment_overlap: np.ndarray[Any, np.dtype[np.uint8]]
    fragment_support: np.ndarray[Any, np.dtype[np.uint8]]
    validity: np.ndarray[Any, np.dtype[np.uint8]]
    separator: np.ndarray[Any, np.dtype[np.uint8]]
    rim_contact: np.ndarray[Any, np.dtype[np.uint8]]
    instance_records: tuple[InstanceRecord, ...]
    audit: Mapping[str, Any]


@dataclass
class _MembershipArrays:
    bbox_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    cortex_union: np.ndarray[Any, np.dtype[np.bool_]]
    cortical_instances: np.ndarray[Any, np.dtype[np.int16]]
    cortical_overlap: np.ndarray[Any, np.dtype[np.bool_]]
    fragment_instances: np.ndarray[Any, np.dtype[np.int16]]
    fragment_overlap: np.ndarray[Any, np.dtype[np.bool_]]
    support: np.ndarray[Any, np.dtype[np.bool_]]
    known_support: np.ndarray[Any, np.dtype[np.bool_]]
    unknown_support: np.ndarray[Any, np.dtype[np.bool_]]
    fragment_counts: np.ndarray[Any, np.dtype[np.int64]]
    cortical_counts: np.ndarray[Any, np.dtype[np.int64]]
    fragments: tuple[Segment, ...]
    cortical_by_base: Mapping[str, Segment]
    local_id_by_name: Mapping[str, int]


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token
        for token in _NATURAL_TOKEN.split(value.casefold())
    )


def cortical_base_name(name: str) -> str:
    suffix = "_cortical"
    if not name.endswith(suffix):
        raise ValueError(f"cortical segment lacks canonical {suffix} suffix: {name}")
    return name[: -len(suffix)]


def _union_bbox(
    layouts: Sequence[NrrdLayout],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    spatial_shape = layouts[0].spatial_shape_xyz
    if any(layout.spatial_shape_xyz != spatial_shape for layout in layouts[1:]):
        raise ValueError("fragment and cortical NRRD spatial shapes differ")
    extents = [
        segment.extent_xyz
        for layout in layouts
        for segment in layout.segments
        if segment.extent_xyz is not None
    ]
    if len(extents) != sum(len(layout.segments) for layout in layouts):
        return tuple((0, size) for size in spatial_shape)  # type: ignore[return-value]
    starts = [
        min(extent[axis * 2] for extent in extents if extent is not None)
        for axis in range(3)
    ]
    stops = [
        max(extent[axis * 2 + 1] for extent in extents if extent is not None) + 1
        for axis in range(3)
    ]
    starts = [max(0, value) for value in starts]
    stops = [min(spatial_shape[axis], value) for axis, value in enumerate(stops)]
    if any(stop <= start for start, stop in zip(starts, stops)):
        raise ValueError("segment extents produce an empty spatial crop")
    return tuple((start, stop) for start, stop in zip(starts, stops))  # type: ignore[return-value]


def _layer_luts(
    layout: NrrdLayout, local_id_by_name: Mapping[str, int], *, cortical: bool
) -> tuple[np.ndarray[Any, np.dtype[np.int32]], ...]:
    per_layer: list[dict[int, int]] = [dict() for _ in range(layout.layer_count)]
    for segment in layout.segments:
        name = cortical_base_name(segment.name) if cortical else segment.name
        if name not in local_id_by_name:
            raise ValueError(f"segment {segment.name!r} has no stable local ID")
        per_layer[segment.layer][segment.label_value] = local_id_by_name[name]
    result: list[np.ndarray[Any, np.dtype[np.int32]]] = []
    for values in per_layer:
        maximum = max(values, default=0)
        lut = np.zeros(maximum + 1, dtype=np.int32)
        for label, local_id in values.items():
            lut[label] = local_id
        result.append(lut)
    return tuple(result)


def _mapped_values(
    raw: np.ndarray[Any, Any],
    lut: np.ndarray[Any, np.dtype[np.int32]],
    *,
    layer: int,
) -> np.ndarray[Any, np.dtype[np.int32]]:
    mapped = np.zeros(raw.shape, dtype=np.int32)
    in_range = raw < len(lut)
    mapped[in_range] = lut[raw[in_range]]
    undeclared = (raw != 0) & (mapped == 0)
    if np.any(undeclared):
        values = np.unique(raw[undeclared]).tolist()
        raise ValueError(f"layer {layer} contains undeclared non-zero labels {values}")
    return mapped


def _active_local_indices(
    start: int,
    vectors: np.ndarray[Any, Any],
    spatial_shape: tuple[int, int, int],
    bbox_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, Any]]:
    active_rows = np.flatnonzero(np.any(vectors != 0, axis=1))
    if not active_rows.size:
        return np.empty(0, dtype=np.int64), vectors[:0]
    global_indices = start + active_rows
    size_x, size_y, _ = spatial_shape
    x = global_indices % size_x
    quotient = global_indices // size_x
    y = quotient % size_y
    z = quotient // size_y
    (x0, x1), (y0, y1), (z0, z1) = bbox_xyz
    inside = (
        (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
        & (z >= z0)
        & (z < z1)
    )
    if not np.all(inside):
        raise ValueError("non-zero NRRD payload lies outside declared segment extents")
    crop_x = x1 - x0
    crop_y = y1 - y0
    local = (x - x0) + crop_x * ((y - y0) + crop_y * (z - z0))
    return local.astype(np.int64, copy=False), vectors[active_rows]


def _extract_memberships(
    fragment_layout: NrrdLayout,
    cortical_layout: NrrdLayout,
    *,
    chunk_spatial_voxels: int,
) -> _MembershipArrays:
    if fragment_layout.spatial_shape_xyz != cortical_layout.spatial_shape_xyz:
        raise ValueError("fragment and cortical grids differ")
    fragments = tuple(sorted(fragment_layout.segments, key=lambda item: _natural_key(item.name)))
    fragment_names = [segment.name for segment in fragments]
    if len(fragment_names) != len(set(fragment_names)):
        raise ValueError("fragment names are not unique")
    local_id_by_name = {name: index + 1 for index, name in enumerate(fragment_names)}

    cortical_by_base: dict[str, Segment] = {}
    for cortical_segment in cortical_layout.segments:
        base = cortical_base_name(cortical_segment.name)
        if base not in local_id_by_name:
            raise ValueError(f"cortical segment lacks exact fragment: {cortical_segment.name}")
        if base in cortical_by_base:
            raise ValueError(f"multiple cortical segments map to {base}")
        cortical_by_base[base] = cortical_segment

    bbox = _union_bbox((fragment_layout, cortical_layout))
    crop_shape = tuple(stop - start for start, stop in bbox)
    flat_size = math.prod(crop_shape)
    cortex_union = np.zeros(crop_shape, dtype=bool, order="F")
    cortex_ids = np.zeros(crop_shape, dtype=np.int16, order="F")
    cortex_overlap = np.zeros(crop_shape, dtype=bool, order="F")
    fragment_ids = np.zeros(crop_shape, dtype=np.int16, order="F")
    fragment_overlap = np.zeros(crop_shape, dtype=bool, order="F")
    support = np.zeros(crop_shape, dtype=bool, order="F")
    known_support = np.zeros(crop_shape, dtype=bool, order="F")
    unknown_support = np.zeros(crop_shape, dtype=bool, order="F")
    cortex_union_flat = cortex_union.reshape(flat_size, order="F")
    cortex_ids_flat = cortex_ids.reshape(flat_size, order="F")
    cortex_overlap_flat = cortex_overlap.reshape(flat_size, order="F")
    fragment_ids_flat = fragment_ids.reshape(flat_size, order="F")
    fragment_overlap_flat = fragment_overlap.reshape(flat_size, order="F")
    support_flat = support.reshape(flat_size, order="F")
    known_flat = known_support.reshape(flat_size, order="F")
    unknown_flat = unknown_support.reshape(flat_size, order="F")

    fragment_luts = _layer_luts(
        fragment_layout, local_id_by_name, cortical=False
    )
    cortical_luts = _layer_luts(cortical_layout, local_id_by_name, cortical=True)
    number_of_ids = len(fragments)
    fragment_counts = np.zeros(number_of_ids + 1, dtype=np.int64)
    cortical_counts = np.zeros(number_of_ids + 1, dtype=np.int64)
    cortical_ids_set = {
        local_id_by_name[name] for name in cortical_by_base
    }
    known_id = np.zeros(number_of_ids + 1, dtype=bool)
    known_id[list(cortical_ids_set)] = True

    for start, vectors in iter_spatial_vectors(
        fragment_layout, chunk_spatial_voxels=chunk_spatial_voxels
    ):
        local, active = _active_local_indices(
            start, vectors, fragment_layout.spatial_shape_xyz, bbox
        )
        for layer, lut in enumerate(fragment_luts):
            mapped = _mapped_values(active[:, layer], lut, layer=layer)
            foreground = mapped > 0
            if not np.any(foreground):
                continue
            locations = local[foreground]
            ids = mapped[foreground]
            support_flat[locations] = True
            known_flat[locations[known_id[ids]]] = True
            unknown_flat[locations[~known_id[ids]]] = True
            fragment_counts += np.bincount(ids, minlength=number_of_ids + 1)
            old = fragment_ids_flat[locations]
            already_ambiguous = fragment_overlap_flat[locations]
            conflict = (old != 0) & (old != ids)
            new_ambiguous = already_ambiguous | conflict
            assign = (old == 0) & ~new_ambiguous
            if np.any(assign):
                fragment_ids_flat[locations[assign]] = ids[assign].astype(
                    np.int16
                )
            if np.any(new_ambiguous):
                ambiguous_locations = locations[new_ambiguous]
                fragment_overlap_flat[ambiguous_locations] = True
                fragment_ids_flat[ambiguous_locations] = 0

    for start, vectors in iter_spatial_vectors(
        cortical_layout, chunk_spatial_voxels=chunk_spatial_voxels
    ):
        local, active = _active_local_indices(
            start, vectors, cortical_layout.spatial_shape_xyz, bbox
        )
        for layer, lut in enumerate(cortical_luts):
            mapped = _mapped_values(active[:, layer], lut, layer=layer)
            foreground = mapped > 0
            if not np.any(foreground):
                continue
            locations = local[foreground]
            ids = mapped[foreground]
            cortical_counts += np.bincount(ids, minlength=number_of_ids + 1)
            cortex_union_flat[locations] = True

            old = cortex_ids_flat[locations]
            already_ambiguous = cortex_overlap_flat[locations]
            conflict = (old != 0) & (old != ids)
            new_ambiguous = already_ambiguous | conflict
            assign = (old == 0) & ~new_ambiguous
            if np.any(assign):
                cortex_ids_flat[locations[assign]] = ids[assign].astype(np.int16)
            if np.any(new_ambiguous):
                ambiguous_locations = locations[new_ambiguous]
                cortex_overlap_flat[ambiguous_locations] = True
                cortex_ids_flat[ambiguous_locations] = 0

    return _MembershipArrays(
        bbox_xyz=bbox,
        cortex_union=cortex_union,
        cortical_instances=cortex_ids,
        cortical_overlap=cortex_overlap,
        fragment_instances=fragment_ids,
        fragment_overlap=fragment_overlap,
        support=support,
        known_support=known_support,
        unknown_support=unknown_support,
        fragment_counts=fragment_counts,
        cortical_counts=cortical_counts,
        fragments=fragments,
        cortical_by_base=cortical_by_base,
        local_id_by_name=local_id_by_name,
    )


def _bbox_distance_mm(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
    spacing: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    left_min = left.min(axis=0)
    left_max = left.max(axis=0)
    right_min = right.min(axis=0)
    right_max = right.max(axis=0)
    gap = np.maximum(0, np.maximum(left_min - right_max, right_min - left_max))
    return float(np.linalg.norm(gap * spacing))


def _distance_numpy(
    instance_ids: np.ndarray[Any, np.dtype[np.int16]],
    spacing_xyz_mm: Sequence[float],
    max_distance_mm: float,
    *,
    pair_chunk: int = 512,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    result = np.full(instance_ids.shape, np.inf, dtype=np.float32)
    identifiers = [int(item) for item in np.unique(instance_ids) if item > 0]
    coordinates = {
        identifier: np.argwhere(instance_ids == identifier)
        for identifier in identifiers
    }
    per_id_distance = {
        identifier: np.full(len(coords), np.inf, dtype=np.float64)
        for identifier, coords in coordinates.items()
    }
    limit_squared = max_distance_mm * max_distance_mm
    for left_position, left_id in enumerate(identifiers):
        left = coordinates[left_id]
        for right_id in identifiers[left_position + 1 :]:
            right = coordinates[right_id]
            if _bbox_distance_mm(left, right, spacing) > max_distance_mm:
                continue
            right_best = np.full(len(right), np.inf, dtype=np.float64)
            for start in range(0, len(left), pair_chunk):
                left_chunk = left[start : start + pair_chunk]
                delta = (
                    left_chunk[:, None, :].astype(np.float64)
                    - right[None, :, :].astype(np.float64)
                ) * spacing
                squared = np.sum(delta * delta, axis=2)
                per_id_distance[left_id][start : start + len(left_chunk)] = np.minimum(
                    per_id_distance[left_id][start : start + len(left_chunk)],
                    np.sqrt(np.min(squared, axis=1)),
                )
                right_best = np.minimum(right_best, np.sqrt(np.min(squared, axis=0)))
            per_id_distance[right_id] = np.minimum(
                per_id_distance[right_id], right_best
            )
    for identifier, coords in coordinates.items():
        values = per_id_distance[identifier]
        values[values * values > limit_squared] = np.inf
        result[tuple(coords.T)] = values.astype(np.float32)
    return result


def _distance_scipy(
    instance_ids: np.ndarray[Any, np.dtype[np.int16]],
    spacing_xyz_mm: Sequence[float],
    max_distance_mm: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    from scipy.ndimage import distance_transform_edt

    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    result = np.full(instance_ids.shape, np.inf, dtype=np.float32)
    expansion = np.ceil(max_distance_mm / spacing).astype(int)
    for identifier in (int(item) for item in np.unique(instance_ids) if item > 0):
        coords = np.argwhere(instance_ids == identifier)
        starts = np.maximum(0, coords.min(axis=0) - expansion)
        stops = np.minimum(instance_ids.shape, coords.max(axis=0) + expansion + 1)
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
        roi = instance_ids[slices]
        other = (roi > 0) & (roi != identifier)
        if not np.any(other):
            continue
        distances = distance_transform_edt(~other, sampling=spacing)
        own = roi == identifier
        values = distances[own]
        values[values > max_distance_mm] = np.inf
        target = result[slices]
        target[own] = values.astype(np.float32)
    return result


def different_instance_distance(
    instance_ids: np.ndarray[Any, np.dtype[np.int16]],
    spacing_xyz_mm: Sequence[float],
    max_distance_mm: float,
    *,
    backend: str = "auto",
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Distance from each exclusive cortical voxel to another instance."""

    if instance_ids.ndim != 3:
        raise ValueError("instance_ids must be a 3-D array")
    spacing = tuple(float(item) for item in spacing_xyz_mm)
    if len(spacing) != 3 or any(item <= 0 for item in spacing):
        raise ValueError("spacing_xyz_mm must contain three positive values")
    if max_distance_mm <= 0:
        raise ValueError("max_distance_mm must be positive")
    if backend not in {"auto", "numpy", "scipy"}:
        raise ValueError("distance backend must be auto, numpy, or scipy")
    if backend in {"auto", "scipy"}:
        try:
            return _distance_scipy(instance_ids, spacing, max_distance_mm)
        except ModuleNotFoundError:
            if backend == "scipy":
                raise
    return _distance_numpy(instance_ids, spacing, max_distance_mm)


def derive_case_targets(
    fragment_layout: NrrdLayout,
    cortical_layout: NrrdLayout,
    spacing_xyz_mm: Sequence[float],
    config: BuildConfig,
    *,
    distance_backend: str = "auto",
    chunk_spatial_voxels: int = 500_000,
) -> CaseTargets:
    """Build all Dataset778 sidecars for one case."""

    membership = _extract_memberships(
        fragment_layout,
        cortical_layout,
        chunk_spatial_voxels=chunk_spatial_voxels,
    )
    if not np.all(membership.cortex_union <= membership.support):
        outside = int(np.count_nonzero(membership.cortex_union & ~membership.support))
        raise ValueError(f"{outside} cortical voxels lie outside fragment support")

    semantic_valid = (
        membership.known_support & ~membership.unknown_support
    ) | membership.cortex_union
    relation_valid = membership.cortex_union & ~membership.cortical_overlap
    distances = different_instance_distance(
        membership.cortical_instances,
        spacing_xyz_mm,
        config.rim_negative_mm,
        backend=distance_backend,
    )

    # Keep scan background at 0: nnU-Net's fingerprint extractor treats every
    # positive segmentation value as foreground. Only fragment voxels whose
    # cortical state is genuinely unknown receive the explicit ignore label.
    separator = np.zeros(membership.cortex_union.shape, dtype=np.uint8)
    separator[membership.support & ~semantic_valid] = SEPARATOR_IGNORE
    separator[membership.cortex_union] = SEPARATOR_CORTEX
    separator[
        relation_valid & (distances <= config.separator_mm)
    ] = SEPARATOR_CONTACT

    rim_contact = np.full(
        membership.cortex_union.shape, RIM_IGNORE, dtype=np.uint8
    )
    rim_negative = relation_valid & (
        ~np.isfinite(distances) | (distances >= config.rim_negative_mm)
    )
    rim_positive = relation_valid & (distances <= config.rim_positive_mm)
    rim_contact[rim_negative] = RIM_NEGATIVE
    rim_contact[rim_positive] = RIM_POSITIVE
    rim_valid = rim_negative | rim_positive

    validity = np.zeros(membership.cortex_union.shape, dtype=np.uint8)
    validity[semantic_valid] |= SEMANTIC_VALID
    validity[relation_valid] |= RELATION_VALID
    validity[rim_valid] |= RIM_VALID

    records: list[InstanceRecord] = []
    for fragment in membership.fragments:
        local_id = membership.local_id_by_name[fragment.name]
        cortical = membership.cortical_by_base.get(fragment.name)
        records.append(
            InstanceRecord(
                local_id=local_id,
                fragment_name=fragment.name,
                cortical_name=None if cortical is None else cortical.name,
                fragment_layer=fragment.layer,
                fragment_label_value=fragment.label_value,
                cortical_layer=None if cortical is None else cortical.layer,
                cortical_label_value=None
                if cortical is None
                else cortical.label_value,
                fragment_voxels=int(membership.fragment_counts[local_id]),
                cortical_voxels=int(membership.cortical_counts[local_id]),
            )
        )

    audit = {
        "fragment_count": len(records),
        "cortical_count": sum(record.cortical_name is not None for record in records),
        "fragment_without_cortical_count": sum(
            record.cortical_name is None for record in records
        ),
        "fragment_support_voxels": int(np.count_nonzero(membership.support)),
        "fragment_overlap_voxels": int(
            np.count_nonzero(membership.fragment_overlap)
        ),
        "cortical_union_voxels": int(np.count_nonzero(membership.cortex_union)),
        "cortical_assignment_voxels": int(membership.cortical_counts.sum()),
        "cortical_overlap_voxels": int(
            np.count_nonzero(membership.cortical_overlap)
        ),
        "semantic_valid_voxels": int(np.count_nonzero(semantic_valid)),
        "relation_valid_voxels": int(np.count_nonzero(relation_valid)),
        "separator_voxels": int(
            np.count_nonzero(separator == SEPARATOR_CONTACT)
        ),
        "rim_positive_voxels": int(
            np.count_nonzero(rim_contact == RIM_POSITIVE)
        ),
        "rim_negative_voxels": int(
            np.count_nonzero(rim_contact == RIM_NEGATIVE)
        ),
        "rim_ignore_voxels": int(np.count_nonzero(rim_contact == RIM_IGNORE)),
    }
    return CaseTargets(
        bbox_xyz=membership.bbox_xyz,
        cortex_union=membership.cortex_union.astype(np.uint8),
        cortical_instances=membership.cortical_instances,
        cortical_overlap=membership.cortical_overlap.astype(np.uint8),
        fragment_instances=membership.fragment_instances,
        fragment_overlap=membership.fragment_overlap.astype(np.uint8),
        fragment_support=membership.support.astype(np.uint8),
        validity=validity,
        separator=separator,
        rim_contact=rim_contact,
        instance_records=tuple(records),
        audit=audit,
    )
