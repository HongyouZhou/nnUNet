from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from nnunetv2.training.cortical_continuity.losses import (
    CorticalContinuityLoss,
    balanced_affinity_bce,
    masked_bce_with_logits,
    masked_binary_dice_loss,
)
from nnunetv2.training.cortical_continuity.packing import PackedTargetLayout
from nnunetv2.training.cortical_continuity.schema import build_cortical_continuity_schema


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TorchLossTests(unittest.TestCase):
    def test_empty_masks_return_differentiable_finite_losses(self) -> None:
        logits = torch.randn((1, 1, 2, 2, 2), requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.zeros_like(logits, dtype=torch.bool)
        loss = (
            masked_binary_dice_loss(logits, target, valid)
            + masked_bce_with_logits(logits, target, valid)
            + balanced_affinity_bce(logits, target, valid)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_balanced_affinity_bce_weights_classes_equally(self) -> None:
        logits = torch.zeros((1, 1, 1, 1, 4), requires_grad=True)
        target = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        self.assertAlmostEqual(
            balanced_affinity_bce(logits, target, valid).item(),
            math.log(2.0),
        )

    def test_composite_loss_accepts_packed_target(self) -> None:
        schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))
        logits = torch.zeros((1, schema.total_channels, 2, 2, 2), requires_grad=True)
        target = torch.zeros(
            (1, PackedTargetLayout.from_schema(schema).total_channels, 2, 2, 2)
        )
        target[:, 1] = 1
        loss = CorticalContinuityLoss(schema)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
