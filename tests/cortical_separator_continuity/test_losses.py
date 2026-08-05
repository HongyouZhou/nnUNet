import inspect

import torch

from nnunetv2.training.cortical_separator_continuity.contract import (
    HU_CODE_OFFSET,
    INSTANCE_CHANNEL,
    NORMAL_SURFACE_CHANNEL,
    RELATION_VALID_BIT,
    VALIDITY_CHANNEL,
    DensityCalibration,
)
from nnunetv2.training.cortical_separator_continuity.losses import (
    CorticalSeparatorRegularizedLoss,
)


def _calibration():
    digest = "0" * 64
    return DensityCalibration(
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


def _target(ids, *, surface=False, semantic=1, hu=100):
    ids = torch.as_tensor(ids, dtype=torch.int16)
    target = torch.zeros((1, 6, *ids.shape), dtype=torch.int16)
    target[:, 0] = semantic
    target[:, 1] = 1
    target[:, VALIDITY_CHANNEL] = RELATION_VALID_BIT | 1
    target[:, NORMAL_SURFACE_CHANNEL] = int(surface)
    target[:, 4] = int(hu + HU_CODE_OFFSET)
    target[:, INSTANCE_CHANNEL] = ids
    return target


def _loss(*, continuity=0.1, density=0.0):
    return CorticalSeparatorRegularizedLoss(
        batch_dice=False,
        ddp=False,
        deep_supervision_weights=None,
        continuity_weight=continuity,
        density_weight=density,
        density_calibration=_calibration() if density else None,
    )


def _pair_losses(module, separator_logits, target):
    body = torch.zeros_like(separator_logits)
    logits = torch.cat((body, body, separator_logits), dim=1)
    return module._pair_losses(logits, target)


def test_high_separator_probability_hurts_same_and_helps_different():
    module = _loss()
    low = torch.full((1, 1, 1, 1, 3), -4.0)
    high = torch.full((1, 1, 1, 1, 3), 4.0)
    same = _target([[[1, 1, 1]]])
    different = _target([[[1, 2, 3]]])
    same_low = _pair_losses(module, low, same)[0]
    same_high = _pair_losses(module, high, same)[0]
    different_low = _pair_losses(module, low, different)[0]
    different_high = _pair_losses(module, high, different)[0]
    assert same_high > same_low
    assert different_high < different_low


def test_density_is_conditional_on_surface_candidates():
    module = _loss(density=0.1)
    logits = torch.zeros((1, 1, 1, 1, 3))
    noncandidate = _target([[[1, 2, 2]]], surface=False, semantic=1)
    candidate = _target([[[1, 2, 2]]], surface=True, semantic=1)
    continuity_a, density_a, stats_a = _pair_losses(module, logits, noncandidate)
    continuity_b, density_b, stats_b = _pair_losses(module, logits, candidate)
    assert torch.allclose(continuity_a, continuity_b)
    assert density_a == 0
    assert stats_a["valid_density_pairs"] == 0
    assert density_b > 0
    assert stats_b["valid_density_pairs"] > 0


def test_high_density_different_and_low_density_same_receive_more_weight():
    module = _loss(density=0.1)
    logits = torch.zeros((1, 1, 1, 1, 3))
    different_low = _target([[[1, 2, 3]]], surface=True, hu=-500)
    different_high = _target([[[1, 2, 3]]], surface=True, hu=700)
    same_low = _target([[[1, 1, 1]]], surface=True, hu=-500)
    same_high = _target([[[1, 1, 1]]], surface=True, hu=700)
    assert _pair_losses(module, logits, different_high)[1] > _pair_losses(module, logits, different_low)[1]
    assert _pair_losses(module, logits, same_low)[1] > _pair_losses(module, logits, same_high)[1]


def test_empty_single_class_and_mixed_precision_losses_have_finite_gradients():
    module = _loss(density=0.1)
    empty_target = _target([[[0, 0, 0]]], surface=True)
    empty_logits = torch.zeros((1, 3, 1, 1, 3), requires_grad=True)
    continuity, density, _ = module._pair_losses(empty_logits, empty_target)
    (continuity + density).backward()
    assert torch.isfinite(continuity)
    assert torch.isfinite(density)
    assert empty_logits.grad is not None
    assert torch.all(empty_logits.grad == 0)

    same_target = _target([[[1, 1, 1]]], surface=True)
    half_logits = torch.zeros((1, 3, 1, 1, 3), dtype=torch.float16, requires_grad=True)
    continuity, density, _ = module._pair_losses(half_logits, same_target)
    (continuity + density).backward()
    assert torch.isfinite(continuity)
    assert torch.isfinite(density)
    assert torch.isfinite(half_logits.grad).all()


def test_zero_regularizer_weights_are_exact_base_dice_ce():
    module = _loss(continuity=0.0, density=0.0)
    target = _target([[[1, 1, 1]]])
    logits = torch.randn((1, 3, 1, 1, 3), requires_grad=True)
    observed = module(logits, target)
    expected = module.base(logits, target[:, :1])
    assert torch.equal(observed, expected)


def test_pair_reduction_has_no_explicit_device_to_host_control_flow():
    source = inspect.getsource(CorticalSeparatorRegularizedLoss._pair_losses)
    assert ".item(" not in source
    assert "torch.any(" not in source
