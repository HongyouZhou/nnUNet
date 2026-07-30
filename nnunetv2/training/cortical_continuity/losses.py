from __future__ import annotations

from typing import Any, Mapping

from .packing import unpack_packed_targets
from .schema import CorticalContinuityHeadSchema

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # Keep schema/target tooling importable without torch.
    torch = None
    F = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def torch_is_available() -> bool:
    return torch is not None


if torch is not None:

    def masked_binary_dice_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        smooth: float = 1e-5,
    ) -> torch.Tensor:
        _validate_equal_shapes(logits, target, valid_mask)
        target_float = target.to(dtype=logits.dtype)
        valid_float = valid_mask.to(dtype=logits.dtype)
        probability = torch.sigmoid(logits)
        axes = (0, *range(2, logits.ndim))
        intersection = (probability * target_float * valid_float).sum(axes, dtype=torch.float32)
        prediction_sum = (probability * valid_float).sum(axes, dtype=torch.float32)
        target_sum = (target_float * valid_float).sum(axes, dtype=torch.float32)
        dice = (2.0 * intersection + smooth) / (prediction_sum + target_sum + smooth).clamp_min(1e-8)
        return 1.0 - dice.mean()


    def masked_bce_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_equal_shapes(logits, target, valid_mask)
        target_float = target.to(dtype=logits.dtype)
        valid_float = valid_mask.to(dtype=logits.dtype)
        element_loss = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
        return (element_loss * valid_float).sum() / valid_float.sum().clamp_min(1.0)


    def balanced_affinity_bce(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Give valid positive and negative edges equal total weight.

        If a batch contains only one class, that class still receives full
        weight. A batch without any valid edge returns a differentiable zero.
        """

        _validate_equal_shapes(logits, target, valid_mask)
        target_float = target.to(dtype=logits.dtype)
        valid = valid_mask.bool()
        element_loss = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
        positive = valid & (target_float >= 0.5)
        negative = valid & ~positive
        terms = []
        if bool(positive.any()):
            terms.append(element_loss[positive].mean())
        if bool(negative.any()):
            terms.append(element_loss[negative].mean())
        if not terms:
            return logits.sum() * 0.0
        return torch.stack(terms).mean()


    class CorticalContinuityLoss(nn.Module):
        """Masked ``C`` Dice/BCE plus balanced directional-affinity BCE."""

        def __init__(
            self,
            schema: CorticalContinuityHeadSchema,
            *,
            cortex_dice_weight: float = 1.0,
            cortex_bce_weight: float = 1.0,
            cortex_weight: float = 1.0,
            affinity_weight: float = 1.0,
        ) -> None:
            super().__init__()
            self.schema = schema
            self.cortex_dice_weight = float(cortex_dice_weight)
            self.cortex_bce_weight = float(cortex_bce_weight)
            self.cortex_weight = float(cortex_weight)
            self.affinity_weight = float(affinity_weight)
            if min(
                self.cortex_dice_weight,
                self.cortex_bce_weight,
                self.cortex_weight,
                self.affinity_weight,
            ) < 0:
                raise ValueError("Loss weights must be non-negative")

        def components(
            self,
            flat_logits: torch.Tensor,
            targets: Mapping[str, torch.Tensor] | Any,
        ) -> dict[str, torch.Tensor]:
            if isinstance(flat_logits, (tuple, list)):
                raise ValueError("Cortical continuity v1 requires full-resolution logits without deep supervision")
            heads = self.schema.split(flat_logits, channel_axis=1)
            unpacked_targets = (
                unpack_packed_targets(targets, self.schema, channel_axis=1)
                if isinstance(targets, torch.Tensor)
                else targets
            )
            cortex_target = _target(unpacked_targets, "cortex_target").to(heads["cortex"].device)
            cortex_valid = _target(unpacked_targets, "cortex_valid").to(heads["cortex"].device)
            affinity_target = _target(unpacked_targets, "affinity_target").to(heads["affinity"].device)
            affinity_valid = _target(unpacked_targets, "affinity_valid").to(heads["affinity"].device)

            cortex_dice = masked_binary_dice_loss(heads["cortex"], cortex_target, cortex_valid)
            cortex_bce = masked_bce_with_logits(heads["cortex"], cortex_target, cortex_valid)
            cortex = self.cortex_dice_weight * cortex_dice + self.cortex_bce_weight * cortex_bce
            affinity = balanced_affinity_bce(heads["affinity"], affinity_target, affinity_valid)
            total = self.cortex_weight * cortex + self.affinity_weight * affinity
            return {
                "total": total,
                "cortex": cortex,
                "cortex_dice": cortex_dice,
                "cortex_bce": cortex_bce,
                "affinity": affinity,
            }

        def forward(
            self,
            flat_logits: torch.Tensor,
            targets: Mapping[str, torch.Tensor] | Any,
        ) -> torch.Tensor:
            return self.components(flat_logits, targets)["total"]


    def _validate_equal_shapes(*values: torch.Tensor) -> None:
        shapes = [tuple(i.shape) for i in values]
        if len(set(shapes)) != 1:
            raise ValueError(f"Logits, target, and valid mask must have equal shapes, got {shapes}")


    def _target(targets: Mapping[str, torch.Tensor] | Any, name: str) -> torch.Tensor:
        if isinstance(targets, Mapping):
            value = targets[name]
        else:
            value = getattr(targets, name)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value

else:

    class CorticalContinuityLoss:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "CorticalContinuityLoss requires PyTorch; schema and NumPy target generation "
                "remain available without it"
            ) from _TORCH_IMPORT_ERROR


    def masked_binary_dice_loss(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("masked_binary_dice_loss requires PyTorch") from _TORCH_IMPORT_ERROR


    def masked_bce_with_logits(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("masked_bce_with_logits requires PyTorch") from _TORCH_IMPORT_ERROR


    def balanced_affinity_bce(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("balanced_affinity_bce requires PyTorch") from _TORCH_IMPORT_ERROR
