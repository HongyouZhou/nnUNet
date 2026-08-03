"""Matched Dice+CE, continuity, and conditional-density separator losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from .contract import (
    CONTINUITY_WEIGHT,
    DENSITY_WEIGHT,
    IGNORE_LABEL,
    SEMANTIC_CHANNEL,
    SEPARATOR_LABEL,
    TARGET_CHANNELS,
    DensityCalibration,
)
from .pairs import iter_material_pair_channels


class CorticalSeparatorRegularizedLoss(nn.Module):
    """Base loss at all scales and pair regularizers at the finest scale."""

    def __init__(
        self,
        *,
        batch_dice: bool,
        ddp: bool,
        deep_supervision_weights: Sequence[float] | None,
        continuity_weight: float = CONTINUITY_WEIGHT,
        density_weight: float = 0.0,
        density_calibration: DensityCalibration | None = None,
    ) -> None:
        super().__init__()
        self.continuity_weight = float(continuity_weight)
        self.density_weight = float(density_weight)
        if self.continuity_weight < 0 or self.density_weight < 0:
            raise ValueError("regularizer weights must be non-negative")
        if self.density_weight and density_calibration is None:
            raise ValueError("density_weight requires a fold-specific calibration")
        self.density_calibration = density_calibration
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
        self.last_components: dict[str, torch.Tensor] = {}

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
                raise RuntimeError("Deep-supervision weights do not match outputs")
            weights = self.deep_supervision_weights

        base_loss = outputs[0].sum() * 0.0
        for scale_weight, scale_output, scale_target in zip(
            weights, outputs, targets, strict=True
        ):
            if scale_weight == 0:
                continue
            _validate_target(scale_target)
            semantic = scale_target[:, SEMANTIC_CHANNEL : SEMANTIC_CHANNEL + 1]
            base_loss = base_loss + float(scale_weight) * self.base(
                scale_output, semantic
            )

        zero = outputs[0].sum() * 0.0
        continuity_loss = zero
        density_loss = zero
        counts = {
            "valid_same_pairs": zero.detach(),
            "valid_different_pairs": zero.detach(),
            "valid_density_pairs": zero.detach(),
        }
        if self.continuity_weight or self.density_weight:
            continuity_loss, density_loss, counts = self._pair_losses(
                outputs[0], targets[0]
            )
        total = (
            base_loss
            + self.continuity_weight * continuity_loss
            + self.density_weight * density_loss
        )
        self.last_components = {
            "base_loss": base_loss.detach(),
            "continuity_loss": continuity_loss.detach(),
            "density_loss": density_loss.detach(),
            **counts,
        }
        return total

    def _pair_losses(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        _validate_target(target)
        probability = torch.softmax(output.float(), dim=1)[:, SEPARATOR_LABEL]
        eps = torch.finfo(probability.dtype).eps
        zero = output.sum() * 0.0
        same_sum = zero
        different_sum = zero
        density_same_sum = zero
        density_different_sum = zero
        same_count = probability.new_zeros(())
        different_count = probability.new_zeros(())
        density_same_count = probability.new_zeros(())
        density_different_count = probability.new_zeros(())

        calibration = self.density_calibration if self.density_weight else None
        for pair in iter_material_pair_channels(
            probability, target, calibration=calibration
        ):
            q_cut = pair.q_cut.clamp(min=eps, max=1.0 - eps)
            same = pair.valid & ~pair.different
            different = pair.valid & pair.different
            if torch.any(same):
                same_sum = same_sum + (-torch.log1p(-q_cut[same])).sum()
                same_count = same_count + same.sum(dtype=torch.float32)
            if torch.any(different):
                different_sum = different_sum + (-torch.log(q_cut[different])).sum()
                different_count = different_count + different.sum(dtype=torch.float32)

            if self.density_weight:
                if pair.density_pair is None:
                    raise AssertionError("Density calibration did not produce pair density")
                density_same = pair.density_candidate & ~pair.different
                density_different = pair.density_candidate & pair.different
                if torch.any(density_same):
                    density_same_sum = density_same_sum + (
                        (1.0 - pair.density_pair[density_same])
                        * -torch.log1p(-q_cut[density_same])
                    ).sum()
                    density_same_count = density_same_count + density_same.sum(
                        dtype=torch.float32
                    )
                if torch.any(density_different):
                    density_different_sum = density_different_sum + (
                        pair.density_pair[density_different]
                        * -torch.log(q_cut[density_different])
                    ).sum()
                    density_different_count = (
                        density_different_count
                        + density_different.sum(dtype=torch.float32)
                    )

        continuity = zero
        if bool(same_count.detach().item()):
            continuity = continuity + same_sum / same_count
        if bool(different_count.detach().item()):
            continuity = continuity + different_sum / different_count

        density_loss = zero
        if bool(density_same_count.detach().item()):
            density_loss = density_loss + density_same_sum / density_same_count
        if bool(density_different_count.detach().item()):
            density_loss = density_loss + density_different_sum / density_different_count

        return continuity, density_loss, {
            "valid_same_pairs": same_count.detach(),
            "valid_different_pairs": different_count.detach(),
            "valid_density_pairs": (
                density_same_count + density_different_count
            ).detach(),
        }


def _validate_target(target: torch.Tensor) -> None:
    if target.ndim != 5 or int(target.shape[1]) != TARGET_CHANNELS:
        raise RuntimeError(
            f"separator continuity target must be [B,{TARGET_CHANNELS},Z,Y,X], "
            f"got {tuple(target.shape)}"
        )
