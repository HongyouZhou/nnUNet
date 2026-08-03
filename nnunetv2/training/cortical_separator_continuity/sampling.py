"""Shared 40/40/20 separator, normal-surface, and random sampler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from .contract import (
    SAMPLING_WEIGHTS,
    SEPARATOR_SAMPLING_KEY,
    SURFACE_SAMPLING_KEY,
)


class SeparatorContinuityDataLoader(nnUNetDataLoader):
    """Use an identical patch distribution for all three pilot arms."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sampling_fallback_counts: dict[str, int] = {}

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: dict | None,
        overwrite_class=None,
        verbose: bool = False,
    ):
        del force_fg, overwrite_class
        category, selected_voxel, requested = sample_separator_continuity_center(
            class_locations
        )
        if category != requested:
            key = f"{requested}->{category}"
            self.sampling_fallback_counts[key] = self.sampling_fallback_counts.get(key, 0) + 1
        need_to_pad = self.need_to_pad.copy()
        if len(data_shape) != 3:
            raise RuntimeError("Separator continuity training supports 3-D full resolution only")
        for axis in range(3):
            if need_to_pad[axis] + data_shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - data_shape[axis]
        lower = [-need_to_pad[axis] // 2 for axis in range(3)]
        upper = [
            data_shape[axis]
            + need_to_pad[axis] // 2
            + need_to_pad[axis] % 2
            - self.patch_size[axis]
            for axis in range(3)
        ]
        if selected_voxel is None:
            bbox_lbs = [
                np.random.randint(lower[axis], upper[axis] + 1) for axis in range(3)
            ]
        else:
            if selected_voxel.shape != (4,):
                raise RuntimeError(
                    f"Sampling centre must be [channel,z,y,x], got {selected_voxel.shape}"
                )
            bbox_lbs = [
                min(
                    upper[axis],
                    max(
                        lower[axis],
                        int(selected_voxel[axis + 1]) - int(self.patch_size[axis]) // 2,
                    ),
                )
                for axis in range(3)
            ]
        if verbose:
            print(f"continuity sampling requested={requested} selected={category}")
        return bbox_lbs, [
            bbox_lbs[axis] + int(self.patch_size[axis]) for axis in range(3)
        ]


def sample_separator_continuity_center(
    class_locations: Mapping[Any, Any] | None,
    *,
    rng: Any = np.random,
) -> tuple[str, np.ndarray | None, str]:
    locations = _validate_locations(class_locations)
    categories = tuple(SAMPLING_WEIGHTS)
    probabilities = np.asarray(
        [SAMPLING_WEIGHTS[name] for name in categories], dtype=np.float64
    )
    requested = str(rng.choice(categories, p=probabilities))
    pools = {
        "separator": _coordinates(locations[SEPARATOR_SAMPLING_KEY], SEPARATOR_SAMPLING_KEY),
        "surface": _coordinates(locations[SURFACE_SAMPLING_KEY], SURFACE_SAMPLING_KEY),
    }
    category = requested
    if category != "random" and len(pools[category]) == 0:
        fallback = "surface" if category == "separator" else "separator"
        category = fallback if len(pools[fallback]) else "random"
    if category == "random":
        return category, None, requested
    pool = pools[category]
    selected = pool[int(rng.choice(len(pool)))]
    return category, np.asarray(selected, dtype=np.int64), requested


def _validate_locations(value: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Separator continuity class_locations are missing")
    missing = [
        key
        for key in (SEPARATOR_SAMPLING_KEY, SURFACE_SAMPLING_KEY)
        if key not in value
    ]
    if missing:
        raise RuntimeError(f"Separator continuity sampling keys are missing: {missing}")
    return value


def _coordinates(value: Any, name: str) -> np.ndarray:
    coordinates = np.asarray(value)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise RuntimeError(f"{name} must have shape [N,4], got {coordinates.shape}")
    if coordinates.size and (
        not np.issubdtype(coordinates.dtype, np.integer)
        or np.any(coordinates[:, 0] != 0)
    ):
        raise RuntimeError(f"{name} must contain integer [0,z,y,x] coordinates")
    return coordinates
