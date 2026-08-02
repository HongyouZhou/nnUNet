"""Density-aware training support for the Dataset778 separator baseline."""

from .contract import DensityCalibration, density_score_from_hu

__all__ = ["DensityCalibration", "density_score_from_hu"]
