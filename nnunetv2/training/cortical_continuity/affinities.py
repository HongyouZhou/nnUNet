from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import AffinityOffset, CorticalContinuityHeadSchema


@dataclass(frozen=True)
class ContinuityTargets:
    """Dense full-resolution targets produced after spatial augmentation."""

    cortex_target: np.ndarray
    cortex_valid: np.ndarray
    affinity_target: np.ndarray
    affinity_valid: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "cortex_target": self.cortex_target,
            "cortex_valid": self.cortex_valid,
            "affinity_target": self.affinity_target,
            "affinity_valid": self.affinity_valid,
        }


def build_continuity_targets(
    augmented_instance_map: np.ndarray,
    schema_or_offsets: CorticalContinuityHeadSchema | Sequence[AffinityOffset],
    *,
    cortex_mask: np.ndarray | None = None,
    cortex_valid_mask: np.ndarray | None = None,
    relation_valid_mask: np.ndarray | None = None,
    overlap_mask: np.ndarray | None = None,
    offset_valid: Sequence[bool] | np.ndarray | None = None,
) -> ContinuityTargets:
    """Derive ``C`` and directional ``A`` targets from an augmented ID map.

    This function must be called *after* synchronized spatial augmentation.
    Directional affinity channels must never be precomputed and then mirrored
    or rotated like semantic labels.

    The affinity value is stored at the source voxel. An edge is valid only
    when source and destination are in bounds, have positive instance IDs, and
    neither endpoint is ignored or multiply owned. Overlap voxels remain valid
    positive cortex supervision but are excluded from identity supervision.
    """

    instance_map = np.asarray(augmented_instance_map)
    if instance_map.ndim != 3:
        raise ValueError(f"augmented_instance_map must be 3D z-y-x, got shape {instance_map.shape}")
    if not np.issubdtype(instance_map.dtype, np.integer):
        raise TypeError("augmented_instance_map must contain integer instance IDs")
    if np.any(instance_map < 0):
        raise ValueError("Instance IDs must be non-negative; zero is background")

    offsets = (
        schema_or_offsets.affinity_offsets
        if isinstance(schema_or_offsets, CorticalContinuityHeadSchema)
        else tuple(schema_or_offsets)
    )
    if len(offsets) == 0:
        raise ValueError("At least one affinity offset is required")
    if offset_valid is None:
        valid_offset_channels = np.ones(len(offsets), dtype=bool)
    else:
        valid_offset_channels = np.asarray(offset_valid, dtype=bool)
        if valid_offset_channels.shape != (len(offsets),):
            raise ValueError(
                "offset_valid must contain one boolean per affinity channel"
            )

    cortex = instance_map > 0 if cortex_mask is None else _as_bool_mask(cortex_mask, instance_map.shape, "cortex_mask")
    cortex_valid = (
        np.ones(instance_map.shape, dtype=bool)
        if cortex_valid_mask is None
        else _as_bool_mask(cortex_valid_mask, instance_map.shape, "cortex_valid_mask")
    )
    relation_valid = (
        cortex_valid.copy()
        if relation_valid_mask is None
        else _as_bool_mask(relation_valid_mask, instance_map.shape, "relation_valid_mask")
    )
    overlap = (
        np.zeros(instance_map.shape, dtype=bool)
        if overlap_mask is None
        else _as_bool_mask(overlap_mask, instance_map.shape, "overlap_mask")
    )
    relation_valid &= ~overlap

    affinity_target = np.zeros((len(offsets), *instance_map.shape), dtype=np.float32)
    affinity_valid = np.zeros((len(offsets), *instance_map.shape), dtype=bool)

    for channel, affinity_offset in enumerate(offsets):
        if not valid_offset_channels[channel]:
            continue
        offset = tuple(int(i) for i in affinity_offset.voxel_offset_zyx)
        source_slices, destination_slices = _edge_slices(instance_map.shape, offset)
        source_ids = instance_map[source_slices]
        destination_ids = instance_map[destination_slices]
        valid = (
            relation_valid[source_slices]
            & relation_valid[destination_slices]
            & (source_ids > 0)
            & (destination_ids > 0)
        )
        target = valid & (source_ids == destination_ids)

        affinity_valid[(channel, *source_slices)] = valid
        affinity_target[(channel, *source_slices)] = target.astype(np.float32, copy=False)

    return ContinuityTargets(
        cortex_target=cortex.astype(np.float32, copy=False)[None],
        cortex_valid=cortex_valid[None],
        affinity_target=affinity_target,
        affinity_valid=affinity_valid,
    )


def _as_bool_mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match instance map shape {shape}")
    return mask.astype(bool, copy=False)


def _edge_slices(
    shape: tuple[int, int, int],
    offset: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    if len(offset) != 3 or offset == (0, 0, 0):
        raise ValueError(f"Invalid non-zero z-y-x affinity offset {offset}")

    source: list[slice] = []
    destination: list[slice] = []
    for size, delta in zip(shape, offset):
        if abs(delta) >= size:
            # Empty but correctly shaped slices make every edge invalid.
            source.append(slice(0, 0))
            destination.append(slice(0, 0))
        elif delta >= 0:
            source.append(slice(0, size - delta))
            destination.append(slice(delta, size))
        else:
            source.append(slice(-delta, size))
            destination.append(slice(0, size + delta))
    return tuple(source), tuple(destination)
