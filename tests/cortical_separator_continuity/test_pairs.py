import torch

from nnunetv2.training.cortical_separator_continuity.contract import (
    HU_CODE_OFFSET,
    INSTANCE_CHANNEL,
    LOCAL_HALF_OFFSETS_ZYX,
    LIFTED_OFFSETS_ZYX,
    PAIR_OFFSETS_ZYX,
    RELATION_VALID_BIT,
    VALIDITY_CHANNEL,
    DensityCalibration,
    path_offsets_for_pair,
)
from nnunetv2.training.cortical_separator_continuity.pairs import (
    iter_material_pair_channels,
)


def _target(ids):
    ids = torch.as_tensor(ids, dtype=torch.int16)
    target = torch.zeros((1, 6, *ids.shape), dtype=torch.int16)
    target[:, 0] = 1
    target[:, 1] = 1
    target[:, VALIDITY_CHANNEL] = RELATION_VALID_BIT | 1
    target[:, INSTANCE_CHANNEL] = ids
    target[:, 4] = 2049
    return target


def test_frozen_offsets_are_13_local_plus_three_lifted_paths():
    assert len(LOCAL_HALF_OFFSETS_ZYX) == 13
    assert len(LIFTED_OFFSETS_ZYX) == 3
    assert len(PAIR_OFFSETS_ZYX) == 16
    assert len(set(PAIR_OFFSETS_ZYX)) == 16
    assert path_offsets_for_pair((2, 0, 0)) == (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    )


def test_same_different_targets_and_lifted_path_max():
    target = _target([[[1, 1, 2]]])
    probability = torch.tensor([[[[0.1, 0.8, 0.2]]]], dtype=torch.float32)
    channels = {
        pair.offset_zyx: pair
        for pair in iter_material_pair_channels(probability, target)
    }
    local = channels[(0, 0, 1)]
    assert local.valid.tolist() == [[[[True, True]]]]
    assert local.different.tolist() == [[[[False, True]]]]
    assert torch.allclose(local.q_cut, torch.tensor([[[[0.8, 0.8]]]]))
    lifted = channels[(0, 0, 2)]
    assert lifted.valid.numel() == 1
    assert lifted.valid.item()
    assert lifted.different.item()
    assert lifted.q_cut.item() == 0.8


def test_lifted_density_uses_endpoints_and_not_middle_gap():
    target = _target([[[1, 1, 2]]])
    target[:, 4] = torch.tensor(
        [[[[HU_CODE_OFFSET - 500, HU_CODE_OFFSET + 900, HU_CODE_OFFSET - 500]]]],
        dtype=torch.int16,
    )
    digest = "0" * 64
    calibration = DensityCalibration(
        fold=0,
        train_cases=("case",),
        q25_hu=0,
        q50_hu=100,
        q75_hu=200,
        splits_sha256=digest,
        plans_sha256=digest,
        source_manifest_sha256=digest,
        voxel_count=1,
    )
    probability = torch.full((1, 1, 1, 3), 0.5)
    lifted = next(
        pair
        for pair in iter_material_pair_channels(
            probability, target, calibration=calibration
        )
        if pair.offset_zyx == (0, 0, 2)
    )
    assert lifted.density_pair is not None
    assert lifted.density_pair.item() < 0.01


def test_boundary_relation_validity_and_zero_ownership_exclude_pairs():
    target = _target([[[1, 0, 2]]])
    target[:, VALIDITY_CHANNEL, 0, 0, 2] = 1
    probability = torch.full((1, 1, 1, 3), 0.5)
    channels = {
        pair.offset_zyx: pair
        for pair in iter_material_pair_channels(probability, target)
    }
    assert not torch.any(channels[(0, 0, 1)].valid)
    assert not torch.any(channels[(0, 0, 2)].valid)
    assert channels[(0, 0, 2)].valid.shape[-1] == 1

    ignored = _target([[[1, 2, 2]]])
    ignored[:, 0, 0, 0, 1] = 3
    local = next(
        pair
        for pair in iter_material_pair_channels(probability, ignored)
        if pair.offset_zyx == (0, 0, 1)
    )
    assert not torch.any(local.valid)


def test_pairs_are_rebuilt_from_mirrored_instance_ids():
    target = _target([[[1, 1, 2]]])
    probability = torch.full((1, 1, 1, 3), 0.5)
    original = next(
        pair
        for pair in iter_material_pair_channels(probability, target)
        if pair.offset_zyx == (0, 0, 1)
    )
    mirrored_target = torch.flip(target, dims=(-1,))
    mirrored_probability = torch.flip(probability, dims=(-1,))
    mirrored = next(
        pair
        for pair in iter_material_pair_channels(mirrored_probability, mirrored_target)
        if pair.offset_zyx == (0, 0, 1)
    )
    assert original.different.tolist() == [[[[False, True]]]]
    assert mirrored.different.tolist() == [[[[True, False]]]]


def test_pairs_are_rebuilt_in_rotated_direction():
    target = _target([[[1, 1, 2]]])
    probability = torch.full((1, 1, 1, 3), 0.5)
    rotated_target = target.transpose(-1, -2)
    rotated_probability = probability.transpose(-1, -2)
    channels = {
        pair.offset_zyx: pair
        for pair in iter_material_pair_channels(rotated_probability, rotated_target)
    }
    assert channels[(0, 1, 0)].different.tolist() == [[[[False], [True]]]]
    assert channels[(0, 0, 1)].valid.numel() == 0
