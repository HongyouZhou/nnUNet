"""Thin wrapper exposing the trainer to nnUNetv2_train ``-tr`` discovery."""

from nnunetv2.training.cortical_continuity.trainer import (
    nnUNetTrainerCorticalContinuity,
)

__all__ = ["nnUNetTrainerCorticalContinuity"]
