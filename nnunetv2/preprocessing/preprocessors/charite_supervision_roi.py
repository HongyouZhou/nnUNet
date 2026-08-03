"""Memory-bounded ROI cropping shared by the Charite preprocessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CHARITE_ROI_MARGIN_MM = 32.0
CHARITE_ROI_PLANS_KEY = "charite_supervision_roi_crop"


@dataclass(frozen=True)
class SupervisionROICrop:
    original_shape_zyx: tuple[int, int, int]
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    margin_mm: float


def crop_to_supervision_roi(
    data: np.ndarray,
    segmentation: np.ndarray,
    roi_mask: np.ndarray,
    properties: dict[str, Any],
    *,
    margin_mm: float = CHARITE_ROI_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray, SupervisionROICrop]:
    """Crop synchronized CZYX arrays before nnU-Net allocates resampled grids."""

    image = np.asarray(data)
    target = np.asarray(segmentation)
    mask = np.asarray(roi_mask, dtype=bool)
    if image.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"ROI cropping requires CZYX arrays; got data={image.shape}, seg={target.shape}"
        )
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"ROI mask must be ZYX or singleton-CZYX; got {mask.shape}")
    if tuple(image.shape[1:]) != tuple(target.shape[1:]) or tuple(mask.shape) != tuple(
        image.shape[1:]
    ):
        raise ValueError(
            "ROI crop inputs must share a spatial grid; "
            f"data={image.shape}, seg={target.shape}, mask={mask.shape}"
        )
    spacing = np.asarray(properties.get("spacing"), dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid ZYX spacing for ROI crop: {spacing}")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError(f"ROI margin must be finite and non-negative; got {margin_mm}")

    bbox = _expanded_bbox(mask, spacing, float(margin_mm))
    slices = tuple(slice(start, stop) for start, stop in bbox)
    cropped_data = np.ascontiguousarray(image[(slice(None), *slices)])
    cropped_target = np.ascontiguousarray(target[(slice(None), *slices)])
    context = SupervisionROICrop(
        original_shape_zyx=tuple(int(value) for value in image.shape[1:]),
        bbox_zyx=bbox,
        margin_mm=float(margin_mm),
    )
    return cropped_data, cropped_target, context


def restore_full_grid_crop_properties(
    properties: dict[str, Any],
    crop: SupervisionROICrop,
    transpose_forward: list[int] | tuple[int, ...],
) -> None:
    """Compose the outer supervision crop with nnU-Net's inner nonzero crop."""

    permutation = tuple(int(value) for value in transpose_forward)
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError(f"Invalid nnU-Net transpose_forward: {permutation}")
    inner = properties.get("bbox_used_for_cropping")
    if inner is None or len(inner) != 3:
        raise ValueError("nnU-Net preprocessing did not record bbox_used_for_cropping")
    outer_transposed = tuple(crop.bbox_zyx[axis] for axis in permutation)
    combined = []
    for outer_axis, inner_axis in zip(outer_transposed, inner):
        if len(inner_axis) != 2:
            raise ValueError(f"Invalid inner crop bbox: {inner}")
        combined.append(
            [
                int(outer_axis[0]) + int(inner_axis[0]),
                int(outer_axis[0]) + int(inner_axis[1]),
            ]
        )
    properties["shape_before_cropping"] = tuple(
        crop.original_shape_zyx[axis] for axis in permutation
    )
    properties["bbox_used_for_cropping"] = combined
    properties["charite_supervision_roi_bbox_zyx"] = [
        [int(start), int(stop)] for start, stop in crop.bbox_zyx
    ]
    properties["charite_supervision_roi_margin_mm"] = crop.margin_mm
    properties["charite_supervision_roi_restore_contract"] = 1


def roi_plans_contract(*, training_source: str) -> dict[str, Any]:
    return {
        "version": 1,
        "margin_mm": CHARITE_ROI_MARGIN_MM,
        "training_source": training_source,
        "inference_source": "frozen_abbc_provisional_support",
        "identity_evidence": "cortical_only",
    }


def _expanded_bbox(
    mask: np.ndarray,
    spacing_zyx: np.ndarray,
    margin_mm: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if not bool(np.any(mask)):
        raise ValueError("Cannot crop a case whose supervision ROI is empty")
    margin_voxels = np.ceil(margin_mm / spacing_zyx).astype(np.int64)
    result = []
    for axis, size in enumerate(mask.shape):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        active = np.flatnonzero(np.any(mask, axis=reduce_axes))
        start = max(0, int(active[0]) - int(margin_voxels[axis]))
        stop = min(int(size), int(active[-1]) + 1 + int(margin_voxels[axis]))
        result.append((start, stop))
    return tuple(result)  # type: ignore[return-value]
