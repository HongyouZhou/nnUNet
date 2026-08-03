"""Voxel-space cortical continuity regularization for separator training."""

from .contract import (
    CONTINUITY_PLANS_KEY,
    CONTINUITY_CONTRACT_VERSION,
    TARGET_CHANNEL_LAYOUT,
    continuity_plans_contract,
    validate_continuity_plans_contract,
)

__all__ = [
    "CONTINUITY_PLANS_KEY",
    "CONTINUITY_CONTRACT_VERSION",
    "TARGET_CHANNEL_LAYOUT",
    "continuity_plans_contract",
    "validate_continuity_plans_contract",
]
