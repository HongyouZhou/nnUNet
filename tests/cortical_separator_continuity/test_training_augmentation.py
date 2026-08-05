from unittest.mock import patch

import numpy as np
import pytest
import torch
from batchgeneratorsv2.transforms.spatial import spatial as spatial_module
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from nnunetv2.training.cortical_separator_continuity.trainer import (
    _assert_joint_nearest_spatial_transform,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _training_transforms(*, mode: str = "bilinear"):
    return nnUNetTrainer.get_training_transforms(
        patch_size=np.asarray((8, 8, 8)),
        rotation_for_DA=(-np.pi / 6, np.pi / 6),
        deep_supervision_scales=None,
        mirror_axes=(),
        do_dummy_2d_data_aug=False,
        segmentation_interpolation_mode=mode,
    )


def _spatial_transform(transforms) -> SpatialTransform:
    matches = [
        transform
        for transform in transforms.transforms
        if isinstance(transform, SpatialTransform)
    ]
    assert len(matches) == 1
    return matches[0]


def test_continuity_can_request_nearest_without_changing_nnunet_default():
    assert _spatial_transform(_training_transforms()).mode_seg == "bilinear"

    continuity_transforms = _training_transforms(mode="nearest")
    assert _spatial_transform(continuity_transforms).mode_seg == "nearest"
    _assert_joint_nearest_spatial_transform(continuity_transforms)


def test_runtime_guard_rejects_categorical_continuity_augmentation():
    with pytest.raises(RuntimeError, match="joint nearest-neighbour"):
        _assert_joint_nearest_spatial_transform(_training_transforms())


def test_nearest_spatial_transform_cost_is_independent_of_target_cardinality():
    transform = SpatialTransform(
        patch_size=(8, 8, 8),
        patch_center_dist_from_border=0,
        random_crop=False,
        p_rotation=1,
        rotation=(0.1, 0.1),
        p_scaling=0,
        bg_style_seg_sampling=False,
        mode_seg="nearest",
        border_mode_seg="constant",
        padding_value_seg=-1,
    )
    image = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    target = torch.zeros((6, 8, 8, 8), dtype=torch.int16)
    target[4] = torch.arange(8**3, dtype=torch.int16).reshape(8, 8, 8)
    target[5] = target[4] % 37

    original_grid_sample = spatial_module.grid_sample
    with patch.object(
        spatial_module, "grid_sample", wraps=original_grid_sample
    ) as grid_sample:
        transformed = transform(image=image, segmentation=target)

    # One call transforms the image and one jointly transforms all six target
    # channels. The 512 unique HU codes must not cause 512 categorical calls.
    assert grid_sample.call_count == 2
    observed = transformed["segmentation"]
    assert observed.dtype == torch.int16
    assert observed.shape == target.shape
