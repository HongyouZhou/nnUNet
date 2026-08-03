"""Dynamic material-pair construction from spatially augmented instance IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .contract import (
    HU_CODE_CHANNEL,
    HU_PADDING_CODE,
    IGNORE_LABEL,
    INSTANCE_CHANNEL,
    NORMAL_SURFACE_CHANNEL,
    PAIR_OFFSETS_ZYX,
    RELATION_VALID_BIT,
    SEMANTIC_CHANNEL,
    SEPARATOR_LABEL,
    TARGET_CHANNELS,
    VALIDITY_CHANNEL,
    DensityCalibration,
    decode_hu_code,
    density_score_from_hu,
    path_offsets_for_pair,
)


@dataclass(frozen=True)
class MaterialPairChannel:
    """One aligned, memory-bounded pair channel."""

    offset_zyx: tuple[int, int, int]
    q_cut: torch.Tensor
    different: torch.Tensor
    valid: torch.Tensor
    density_candidate: torch.Tensor
    density_pair: torch.Tensor | None


def iter_material_pair_channels(
    separator_probability: torch.Tensor,
    target: torch.Tensor,
    *,
    calibration: DensityCalibration | None = None,
    offsets: Sequence[Sequence[int]] = PAIR_OFFSETS_ZYX,
) -> Iterator[MaterialPairChannel]:
    """Yield pairs after augmentation without materializing 16 dense channels.

    ``target`` is the augmented six-channel tensor. Thus mirroring, rotation,
    scaling and patch cropping all happen before instance relations are read.
    """

    _validate_pair_inputs(separator_probability, target)
    probability = separator_probability.float()
    semantic = target[:, SEMANTIC_CHANNEL]
    validity = target[:, VALIDITY_CHANNEL].long()
    surface = target[:, NORMAL_SURFACE_CHANNEL] == 1
    instance = target[:, INSTANCE_CHANNEL].long()
    endpoint_candidate = (semantic == SEPARATOR_LABEL) | surface

    density = None
    hu_valid = None
    if calibration is not None:
        hu_code = target[:, HU_CODE_CHANNEL]
        hu_valid = hu_code != HU_PADDING_CODE
        density = density_score_from_hu(
            decode_hu_code(hu_code.float()),
            calibration.q25_hu,
            calibration.q50_hu,
            calibration.q75_hu,
        )

    spatial_shape = tuple(int(value) for value in target.shape[-3:])
    for raw_offset in offsets:
        offset = tuple(int(value) for value in raw_offset)
        path = path_offsets_for_pair(offset)
        source_spatial, path_spatial = _aligned_path_slices(spatial_shape, path)
        source_index = (slice(None), *source_spatial)
        endpoint_index = (slice(None), *path_spatial[-1])

        q_cut = probability[source_index]
        for spatial_slice in path_spatial[1:]:
            q_cut = torch.maximum(q_cut, probability[(slice(None), *spatial_slice)])

        source_id = instance[source_index]
        endpoint_id = instance[endpoint_index]
        relation_source = (validity[source_index] & RELATION_VALID_BIT) != 0
        relation_endpoint = (validity[endpoint_index] & RELATION_VALID_BIT) != 0
        valid = (
            (source_id > 0)
            & (endpoint_id > 0)
            & relation_source
            & relation_endpoint
            & (semantic[source_index] != IGNORE_LABEL)
            & (semantic[endpoint_index] != IGNORE_LABEL)
        )
        different = source_id != endpoint_id
        density_candidate = (
            valid
            & endpoint_candidate[source_index]
            & endpoint_candidate[endpoint_index]
        )

        density_pair = None
        if density is not None and hu_valid is not None:
            density_candidate = (
                density_candidate
                & hu_valid[source_index]
                & hu_valid[endpoint_index]
            )
            density_pair = (
                density[source_index] + density[endpoint_index]
            ) * 0.5

        yield MaterialPairChannel(
            offset_zyx=offset,
            q_cut=q_cut,
            different=different,
            valid=valid,
            density_candidate=density_candidate,
            density_pair=density_pair,
        )


def _aligned_path_slices(
    shape: tuple[int, int, int],
    path_offsets: Sequence[Sequence[int]],
) -> tuple[
    tuple[slice, slice, slice],
    tuple[tuple[slice, slice, slice], ...],
]:
    """Return slices for source coordinates whose complete path is in bounds."""

    source_starts: list[int] = []
    source_stops: list[int] = []
    for axis, size in enumerate(shape):
        positions = [int(value[axis]) for value in path_offsets]
        start = max(0, -min(positions))
        stop = min(size, size - max(positions))
        if stop < start:
            stop = start
        source_starts.append(start)
        source_stops.append(stop)

    source = tuple(
        slice(start, stop) for start, stop in zip(source_starts, source_stops)
    )
    shifted = tuple(
        tuple(
            slice(source_starts[axis] + int(offset[axis]), source_stops[axis] + int(offset[axis]))
            for axis in range(3)
        )
        for offset in path_offsets
    )
    return source, shifted


def _validate_pair_inputs(probability: torch.Tensor, target: torch.Tensor) -> None:
    if probability.ndim != 4:
        raise RuntimeError(
            "separator_probability must be [B,Z,Y,X], got "
            f"{tuple(probability.shape)}"
        )
    if target.ndim != 5 or int(target.shape[1]) != TARGET_CHANNELS:
        raise RuntimeError(
            f"continuity target must be [B,{TARGET_CHANNELS},Z,Y,X], got "
            f"{tuple(target.shape)}"
        )
    if tuple(probability.shape) != (int(target.shape[0]), *target.shape[-3:]):
        raise RuntimeError("separator probability and target grids differ")
