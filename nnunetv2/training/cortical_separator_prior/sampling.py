"""Paired contact/surface samplers for separator control and prior trainers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nnunetv2.training.cortical_separator_prior.contract import (
    CONTROL_SAMPLING_WEIGHTS,
    PRIOR_SAMPLING_WEIGHTS,
    PRIOR_SEPARATOR_KEY,
    PRIOR_SURFACE_HU_KEY,
    PRIOR_SURFACE_KEY,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class SeparatorPriorDataLoader(nnUNetDataLoader):
    """Select paired separator/surface/random centres without changing input."""

    def __init__(
        self,
        *args: Any,
        density_median_hu: float,
        use_density_prior: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.density_median_hu = float(density_median_hu)
        self.use_density_prior = bool(use_density_prior)
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
        category, selected_voxel, requested = sample_separator_prior_center(
            class_locations,
            density_median_hu=self.density_median_hu,
            use_density_prior=self.use_density_prior,
        )
        if category != requested:
            key = f"{requested}->{category}"
            self.sampling_fallback_counts[key] = self.sampling_fallback_counts.get(key, 0) + 1
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        if dim != 3:
            raise RuntimeError("Density-prior separator supports 3-D full-resolution only")
        for axis in range(dim):
            if need_to_pad[axis] + data_shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - data_shape[axis]
        lower_limits = [-need_to_pad[axis] // 2 for axis in range(dim)]
        upper_limits = [
            data_shape[axis]
            + need_to_pad[axis] // 2
            + need_to_pad[axis] % 2
            - self.patch_size[axis]
            for axis in range(dim)
        ]
        if selected_voxel is None:
            bbox_lbs = [
                np.random.randint(lower_limits[axis], upper_limits[axis] + 1)
                for axis in range(dim)
            ]
        else:
            if selected_voxel.shape != (4,):
                raise RuntimeError(
                    f"Separator sampling centre must be [channel,z,y,x], got {selected_voxel.shape}"
                )
            bbox_lbs = [
                min(
                    upper_limits[axis],
                    max(
                        lower_limits[axis],
                        int(selected_voxel[axis + 1]) - int(self.patch_size[axis]) // 2,
                    ),
                )
                for axis in range(dim)
            ]
        if verbose:
            print(
                f"separator sampling requested={requested} selected={category} "
                f"density_prior={self.use_density_prior}"
            )
        return bbox_lbs, [
            bbox_lbs[axis] + int(self.patch_size[axis]) for axis in range(dim)
        ]


def sample_separator_prior_center(
    class_locations: Mapping[Any, Any] | None,
    *,
    density_median_hu: float,
    use_density_prior: bool,
    rng: Any = np.random,
) -> tuple[str, np.ndarray | None, str]:
    locations = _validate_locations(class_locations)
    weights = PRIOR_SAMPLING_WEIGHTS if use_density_prior else CONTROL_SAMPLING_WEIGHTS
    requested_categories = tuple(weights)
    requested_probabilities = np.asarray(
        [weights[name] for name in requested_categories], dtype=np.float64
    )
    requested = str(rng.choice(requested_categories, p=requested_probabilities))
    pools = _candidate_pools(
        locations,
        density_median_hu=float(density_median_hu),
        use_density_prior=use_density_prior,
    )
    category = requested
    if category != "random" and len(pools[category]) == 0:
        fallback_order = {
            "separator": ("surface", "surface_low", "surface_high", "random"),
            "surface": ("separator", "random"),
            "surface_low": ("surface_high", "separator", "random"),
            "surface_high": ("surface_low", "separator", "random"),
        }[category]
        category = next(
            name for name in fallback_order if name == "random" or len(pools.get(name, ()))
        )
    if category == "random":
        return category, None, requested
    pool = np.asarray(pools[category])
    selected = pool[int(rng.choice(len(pool)))]
    return category, np.asarray(selected[:4], dtype=np.int64), requested


def _candidate_pools(
    class_locations: Mapping[Any, Any],
    *,
    density_median_hu: float,
    use_density_prior: bool,
) -> dict[str, np.ndarray]:
    separator = _coordinates(class_locations[PRIOR_SEPARATOR_KEY], PRIOR_SEPARATOR_KEY)
    surface = _coordinates(class_locations[PRIOR_SURFACE_KEY], PRIOR_SURFACE_KEY)
    result = {"separator": separator, "surface": surface}
    if use_density_prior:
        records = np.asarray(class_locations[PRIOR_SURFACE_HU_KEY])
        if records.ndim != 2 or records.shape[1] != 5:
            raise RuntimeError(
                f"{PRIOR_SURFACE_HU_KEY} must have shape [N,5], got {records.shape}"
            )
        if len(records) != len(surface) or (
            len(records) and not np.array_equal(records[:, :4], surface)
        ):
            raise RuntimeError("Normal-surface coordinates and HU records are not aligned")
        result["surface_low"] = records[records[:, 4] < density_median_hu]
        result["surface_high"] = records[records[:, 4] >= density_median_hu]
    return result


def _validate_locations(value: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Density-prior preprocessing class_locations are missing")
    missing = [
        key
        for key in (PRIOR_SEPARATOR_KEY, PRIOR_SURFACE_KEY, PRIOR_SURFACE_HU_KEY)
        if key not in value
    ]
    if missing:
        raise RuntimeError(
            f"Density-prior preprocessing sampling keys are missing: {missing}"
        )
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
