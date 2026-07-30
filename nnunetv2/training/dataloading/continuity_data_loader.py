"""Training-only 40/30/20/10 cortical-continuity patch loader."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nnunetv2.training.cortical_continuity.sampling import (
    DEFAULT_SAMPLING_WEIGHTS,
    sample_patch_center,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class ContinuityDataLoader(nnUNetDataLoader):
    """Select contact/instance/support/random centres with case-local fallback."""

    def __init__(
        self,
        *args: Any,
        continuity_sampling_weights: Mapping[str, float] = DEFAULT_SAMPLING_WEIGHTS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.continuity_sampling_weights = dict(continuity_sampling_weights)

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: dict | None,
        overwrite_class=None,
        verbose: bool = False,
    ):
        del force_fg, overwrite_class
        category, selected_voxel = sample_patch_center(
            class_locations,
            sampling_weights=self.continuity_sampling_weights,
        )
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        if dim != 3:
            raise RuntimeError(
                "Cortical continuity v1 supports 3-D full-resolution training only"
            )
        for axis in range(dim):
            if need_to_pad[axis] + data_shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - data_shape[axis]
        lower_limits = [-need_to_pad[i] // 2 for i in range(dim)]
        upper_limits = [
            data_shape[i]
            + need_to_pad[i] // 2
            + need_to_pad[i] % 2
            - self.patch_size[i]
            for i in range(dim)
        ]
        if selected_voxel is None:
            bbox_lbs = [
                np.random.randint(lower_limits[i], upper_limits[i] + 1)
                for i in range(dim)
            ]
        else:
            if selected_voxel.shape != (4,):
                raise RuntimeError(
                    f"Selected continuity centre must be [channel,z,y,x], got {selected_voxel.shape}"
                )
            bbox_lbs = [
                max(
                    lower_limits[i],
                    int(selected_voxel[i + 1]) - int(self.patch_size[i]) // 2,
                )
                for i in range(dim)
            ]
        if verbose:
            print(f"continuity sampling category={category}")
        bbox_ubs = [
            bbox_lbs[i] + int(self.patch_size[i])
            for i in range(dim)
        ]
        return bbox_lbs, bbox_ubs
