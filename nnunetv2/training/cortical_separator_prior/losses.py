"""Paired standard and density-aware losses for the semantic separator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from nnunetv2.training.cortical_separator_prior.contract import (
    HU_CODE_CHANNEL,
    HU_PADDING_CODE,
    IGNORE_LABEL,
    NORMAL_SURFACE_CHANNEL,
    SEMANTIC_CHANNEL,
    SEPARATOR_LABEL,
    TARGET_CHANNELS,
    DensityCalibration,
    decode_hu_code,
    density_score_from_hu,
)
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class PairedSeparatorLoss(nn.Module):
    """Standard Dice+CE at all scales plus a finest-scale density term."""

    def __init__(
        self,
        calibration: DensityCalibration,
        *,
        use_density_prior: bool,
        batch_dice: bool,
        ddp: bool,
        deep_supervision_weights: Sequence[float] | None,
        density_auxiliary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.calibration = calibration
        self.use_density_prior = bool(use_density_prior)
        self.density_auxiliary_weight = float(density_auxiliary_weight)
        if self.density_auxiliary_weight < 0:
            raise ValueError("density_auxiliary_weight must be non-negative")
        self.base = DC_and_CE_loss(
            {
                "batch_dice": bool(batch_dice),
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": bool(ddp),
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=IGNORE_LABEL,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.deep_supervision_weights = (
            None
            if deep_supervision_weights is None
            else tuple(float(value) for value in deep_supervision_weights)
        )

    def forward(self, output, target):
        outputs = tuple(output) if isinstance(output, (tuple, list)) else (output,)
        targets = tuple(target) if isinstance(target, (tuple, list)) else (target,)
        if len(outputs) != len(targets):
            raise RuntimeError(
                f"Output/target deep-supervision lengths differ: {len(outputs)} != {len(targets)}"
            )
        if self.deep_supervision_weights is None:
            weights = (1.0,) * len(outputs)
        else:
            if len(self.deep_supervision_weights) != len(outputs):
                raise RuntimeError(
                    "Configured deep-supervision weights do not match network outputs"
                )
            weights = self.deep_supervision_weights
        total = outputs[0].sum() * 0.0
        for scale_weight, scale_output, scale_target in zip(
            weights, outputs, targets, strict=True
        ):
            if scale_weight == 0:
                continue
            _validate_target(scale_target)
            semantic = scale_target[:, SEMANTIC_CHANNEL : SEMANTIC_CHANNEL + 1]
            total = total + float(scale_weight) * self.base(scale_output, semantic)
        if self.use_density_prior and self.density_auxiliary_weight:
            total = total + self.density_auxiliary_weight * self._density_loss(
                outputs[0], targets[0]
            )
        return total

    def _density_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_target(target)
        semantic = target[:, SEMANTIC_CHANNEL : SEMANTIC_CHANNEL + 1]
        normal_surface = target[
            :, NORMAL_SURFACE_CHANNEL : NORMAL_SURFACE_CHANNEL + 1
        ] == 1
        hu_code = target[:, HU_CODE_CHANNEL : HU_CODE_CHANNEL + 1]
        separator = semantic == SEPARATOR_LABEL
        candidate = (separator | normal_surface) & (semantic != IGNORE_LABEL)
        candidate = candidate & (hu_code != HU_PADDING_CODE)
        if not torch.any(candidate):
            return output.sum() * 0.0

        hu = decode_hu_code(hu_code.float())
        density = density_score_from_hu(
            hu,
            self.calibration.q25_hu,
            self.calibration.q50_hu,
            self.calibration.q75_hu,
        )
        probability = torch.softmax(output, dim=1)[:, SEPARATOR_LABEL : SEPARATOR_LABEL + 1]
        probability = probability.clamp(min=1e-6, max=1.0 - 1e-6)
        binary_target = separator.float()
        voxel_loss = functional.binary_cross_entropy(
            probability,
            binary_target,
            reduction="none",
        )
        positive_weight = 1.0 + 2.0 * density
        negative_weight = 1.0 + (1.0 - density)
        weight = torch.where(separator, positive_weight, negative_weight)
        weight = weight * candidate.float()
        return (voxel_loss * weight).sum() / weight.sum().clamp_min(1.0)


def _validate_target(target: torch.Tensor) -> None:
    if target.ndim != 5 or int(target.shape[1]) != TARGET_CHANNELS:
        raise RuntimeError(
            "Density-prior separator target must be "
            f"[B,{TARGET_CHANNELS},Z,Y,X], got {tuple(target.shape)}"
        )
