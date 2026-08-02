from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nnunetv2.training.cortical_separator_prior.contract import (  # noqa: E402
    HU_CODE_OFFSET,
    DensityCalibration,
)
from nnunetv2.training.cortical_separator_prior.losses import (  # noqa: E402
    PairedSeparatorLoss,
)


def _calibration() -> DensityCalibration:
    return DensityCalibration(
        fold=0,
        train_cases=("case",),
        q25_hu=100,
        q50_hu=500,
        q75_hu=900,
        splits_sha256="0" * 64,
        plans_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        voxel_count=100,
    )


def test_density_loss_retains_low_density_separator_gradient() -> None:
    loss = PairedSeparatorLoss(
        _calibration(),
        use_density_prior=True,
        batch_dice=False,
        ddp=False,
        deep_supervision_weights=None,
    )
    output = torch.zeros((1, 3, 2, 2, 2), requires_grad=True)
    target = torch.zeros((1, 5, 2, 2, 2), dtype=torch.int16)
    target[:, 0] = 2
    target[:, 1] = 1
    target[:, 2] = 1
    target[:, 4] = HU_CODE_OFFSET - 500
    value = loss(output, target)
    value.backward()
    assert torch.isfinite(value)
    assert output.grad is not None
    assert torch.any(output.grad != 0)
