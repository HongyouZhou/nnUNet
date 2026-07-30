from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .affinities import build_continuity_targets
from .schema import CorticalContinuityHeadSchema


SOURCE_CHANNELS = ("cortex", "instance_id", "overlap", "valid_bits", "support")
NATIVE_MIN_STEP_CHANNELS = (
    "native_min_steps_z",
    "native_min_steps_y",
    "native_min_steps_x",
)
PREPROCESSED_SOURCE_CHANNELS = SOURCE_CHANNELS + NATIVE_MIN_STEP_CHANNELS
SEMANTIC_LABEL_VALUES = (0, 1, 2, 3)
SEMANTIC_CORTEX_LABEL = 1
SEMANTIC_CONTACT_LABEL = 2
SEMANTIC_IGNORE_LABEL = 3
SEMANTIC_VALID_BIT = 1
RELATION_VALID_BIT = 2
RIM_VALID_BIT = 4


@dataclass(frozen=True)
class PackedTargetLayout:
    cortex: slice
    cortex_valid: slice
    affinity: slice
    affinity_valid: slice
    total_channels: int

    @classmethod
    def from_schema(cls, schema: CorticalContinuityHeadSchema) -> "PackedTargetLayout":
        affinity_start = 2
        affinity_stop = affinity_start + schema.num_affinity_channels
        affinity_valid_stop = affinity_stop + schema.num_affinity_channels
        return cls(
            cortex=slice(0, 1),
            cortex_valid=slice(1, 2),
            affinity=slice(affinity_start, affinity_stop),
            affinity_valid=slice(affinity_stop, affinity_valid_stop),
            total_channels=affinity_valid_stop,
        )

    def unpack(self, packed: Any, channel_axis: int) -> dict[str, Any]:
        ndim = int(packed.ndim)
        axis = channel_axis if channel_axis >= 0 else ndim + channel_axis
        if not 0 <= axis < ndim:
            raise ValueError(f"channel_axis {channel_axis} is invalid for a {ndim}D target")
        if int(packed.shape[axis]) != self.total_channels:
            raise ValueError(
                "Expected packed cortical target with "
                f"{self.total_channels} channels [C,C_valid,A({self.affinity.stop - self.affinity.start}),"
                f"A_valid({self.affinity_valid.stop - self.affinity_valid.start})], "
                f"got {packed.shape[axis]}"
            )

        def take(channel_slice: slice) -> Any:
            index = [slice(None)] * ndim
            index[axis] = channel_slice
            return packed[tuple(index)]

        return {
            "cortex_target": take(self.cortex),
            "cortex_valid": take(self.cortex_valid),
            "affinity_target": take(self.affinity),
            "affinity_valid": take(self.affinity_valid),
        }


def build_source_segmentation(
    semantic_label: np.ndarray,
    instance_map: np.ndarray,
    overlap_mask: np.ndarray,
    valid_bits: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Validate raw sidecars and construct the spatially augmented source.

    The semantic label is deliberately converted with equality checks:
    labels 1 and 2 supervise cortex, label 3 is ignore and is never treated as
    foreground. Inputs may be 3-D arrays or nnU-Net-style singleton-channel
    arrays.
    """

    semantic = _as_discrete_channel(semantic_label, "semantic label")
    instances = _as_discrete_channel(instance_map, "cortical instance map")
    overlap = _as_discrete_channel(overlap_mask, "cortical overlap mask")
    validity = _as_discrete_channel(valid_bits, "validity bitmask")
    support = _as_discrete_channel(support_mask, "support mask")
    shapes = {
        semantic.shape,
        instances.shape,
        overlap.shape,
        validity.shape,
        support.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "Cortical continuity sidecars must have identical array shapes; "
            f"got semantic={semantic.shape}, instances={instances.shape}, "
            f"overlap={overlap.shape}, valid={validity.shape}, support={support.shape}"
        )

    semantic_values = set(int(i) for i in np.unique(semantic))
    unsupported_semantic = semantic_values.difference(SEMANTIC_LABEL_VALUES)
    if unsupported_semantic:
        raise ValueError(
            "semantic label contains unsupported values "
            f"{sorted(unsupported_semantic)}; expected only {SEMANTIC_LABEL_VALUES}"
        )
    if np.any(instances < 0):
        raise ValueError("cortical instance IDs must be non-negative")
    if np.any(instances > np.iinfo(np.int16).max):
        raise ValueError("cortical instance IDs exceed the int16 preprocessing contract")
    _validate_binary(overlap, "cortical overlap mask")
    _validate_binary(support, "support mask")
    allowed_valid_bits = SEMANTIC_VALID_BIT | RELATION_VALID_BIT | RIM_VALID_BIT
    if np.any(validity < 0) or np.any(validity & ~allowed_valid_bits):
        raise ValueError("validity bitmask contains unsupported bits; only 1, 2, and 4 are defined")

    cortex = (semantic == SEMANTIC_CORTEX_LABEL) | (semantic == SEMANTIC_CONTACT_LABEL)
    if np.any((instances > 0) & ~cortex):
        raise ValueError("cortical instance ownership lies outside semantic labels 1/2")
    if np.any(cortex & (support == 0)):
        raise ValueError("semantic cortex/contact lies outside support")

    return np.stack(
        (
            cortex.astype(np.int16),
            instances.astype(np.int16, copy=False),
            overlap.astype(np.int16, copy=False),
            validity.astype(np.int16, copy=False),
            support.astype(np.int16, copy=False),
        ),
        axis=0,
    )


def semantic_contact_mask(semantic_label: np.ndarray) -> np.ndarray:
    """Return the frozen semantic label-2 contact mask after validation."""

    semantic = _as_discrete_channel(semantic_label, "semantic label")
    semantic_values = set(int(i) for i in np.unique(semantic))
    unsupported = semantic_values.difference(SEMANTIC_LABEL_VALUES)
    if unsupported:
        raise ValueError(
            f"semantic label contains unsupported values {sorted(unsupported)}; "
            f"expected only {SEMANTIC_LABEL_VALUES}"
        )
    return semantic == SEMANTIC_CONTACT_LABEL


def pack_targets_from_augmented_source(
    augmented_source_seg: np.ndarray,
    schema: CorticalContinuityHeadSchema,
) -> np.ndarray:
    """Pack ``[C,I,O,U,S]`` into ``[C_t,C_valid,A_t,A_valid]``.

    Source-channel contract:

    - ``C``: binary cortex, already derived from semantic labels 1/2 only;
    - ``I``: non-negative cortical instance owner ID;
    - ``O``: binary multiple-ownership/overlap mask;
    - ``U``: validity bitmask (bit 0 semantic, bit 1 relation, bit 2 rim);
    - ``S``: binary annotation/support mask.

    Call this only after all synchronized spatial augmentation.
    """

    source = np.asarray(augmented_source_seg)
    if source.ndim != 4 or source.shape[0] not in {
        len(SOURCE_CHANNELS),
        len(PREPROCESSED_SOURCE_CHANNELS),
    }:
        raise ValueError(
            "Cortical continuity preprocessing must provide [C,I,O,U,S] "
            "with optional native minimum-step channels; "
            f"got shape {source.shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("Packed source segmentation contains non-finite values")

    rounded = np.rint(source)
    if not np.allclose(source, rounded, atol=1e-4):
        raise ValueError("Packed source channels must remain discrete after augmentation")
    source_int = rounded.astype(np.int64, copy=False)

    cortex = source_int[0] == 1
    instance_map = source_int[1]
    overlap = source_int[2] == 1
    valid_bits = source_int[3]
    support = source_int[4] == 1
    native_min_steps = np.ones(3, dtype=np.int64)
    if source.shape[0] == len(PREPROCESSED_SOURCE_CHANNELS):
        for axis, channel in enumerate(range(5, 8)):
            values = source_int[channel]
            positive = values[values > 0]
            if positive.size == 0:
                raise ValueError(
                    "native minimum-step metadata disappeared from the patch"
                )
            unique = np.unique(positive)
            if unique.size != 1:
                raise ValueError(
                    "native minimum-step metadata must be spatially constant"
                )
            native_min_steps[axis] = int(unique[0])

    if np.any(instance_map < 0):
        raise ValueError("Instance IDs must be non-negative after padding removal")
    _validate_binary(source_int[0], "cortex")
    _validate_binary(source_int[2], "overlap")
    _validate_binary(source_int[4], "support")
    if np.any(valid_bits < 0) or np.any(valid_bits & ~(SEMANTIC_VALID_BIT | RELATION_VALID_BIT | RIM_VALID_BIT)):
        raise ValueError("valid_bits contains unsupported bits; only 1, 2, and 4 are defined")

    cortex_valid = (valid_bits & SEMANTIC_VALID_BIT) != 0
    relation_valid = ((valid_bits & RELATION_VALID_BIT) != 0) & support
    offset_valid = np.asarray(
        [
            all(
                component == 0
                or abs(int(component)) >= int(native_min_steps[axis])
                for axis, component in enumerate(offset.voxel_offset_zyx)
            )
            for offset in schema.affinity_offsets
        ],
        dtype=bool,
    )
    targets = build_continuity_targets(
        instance_map,
        schema,
        cortex_mask=cortex,
        cortex_valid_mask=cortex_valid,
        relation_valid_mask=relation_valid,
        overlap_mask=overlap,
        offset_valid=offset_valid,
    )
    return np.concatenate(
        (
            targets.cortex_target,
            targets.cortex_valid.astype(np.float32),
            targets.affinity_target,
            targets.affinity_valid.astype(np.float32),
        ),
        axis=0,
        dtype=np.float32,
    )


def unpack_packed_targets(
    packed: Any,
    schema: CorticalContinuityHeadSchema,
    *,
    channel_axis: int,
) -> dict[str, Any]:
    return PackedTargetLayout.from_schema(schema).unpack(packed, channel_axis)


class PackContinuityTargetsTransform:
    """Apply a normal nnU-Net transform, then derive directional targets."""

    def __init__(self, base_transform: Any, schema: CorticalContinuityHeadSchema) -> None:
        self.base_transform = base_transform
        self.schema = schema

    def __call__(self, **data_dict: Any) -> dict[str, Any]:
        transformed = self.base_transform(**data_dict)
        try:
            segmentation = transformed["segmentation"]
        except KeyError as exc:
            raise KeyError("Base augmentation did not return a 'segmentation' value") from exc

        is_torch = segmentation.__class__.__module__.startswith("torch")
        source_numpy = segmentation.detach().cpu().numpy() if is_torch else np.asarray(segmentation)
        packed_numpy = pack_targets_from_augmented_source(source_numpy, self.schema)
        if is_torch:
            import torch

            transformed["segmentation"] = torch.from_numpy(packed_numpy)
        else:
            transformed["segmentation"] = packed_numpy
        return transformed


def _validate_binary(value: np.ndarray, name: str) -> None:
    if np.any((value != 0) & (value != 1)):
        raise ValueError(f"{name} source channel must contain only 0 and 1")


def _as_discrete_channel(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D image or singleton-channel 4-D image; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-4):
        raise ValueError(f"{name} must contain discrete integer values")
    return rounded.astype(np.int64, copy=False)
