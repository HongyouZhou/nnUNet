"""Cortical-continuity model primitives.

The package root intentionally imports only NumPy/standard-library modules so
that dataset preparation and target QA remain usable on machines without a
PyTorch nnU-Net environment.
"""

from .affinities import ContinuityTargets, build_continuity_targets
from .schema import (
    AXIAL_19_DIRECTION_SET,
    CORTICAL_CONTINUITY_SCHEMA_VERSION,
    DENSE_39_DIRECTION_SET,
    SCHEMA_PLANS_KEY,
    SUPPORTED_DIRECTION_SETS,
    AffinityOffset,
    CorticalContinuityHeadSchema,
    FlatHead,
    build_cortical_continuity_schema,
)
from .packing import (
    RELATION_VALID_BIT,
    RIM_VALID_BIT,
    SEMANTIC_CONTACT_LABEL,
    SEMANTIC_CORTEX_LABEL,
    SEMANTIC_IGNORE_LABEL,
    SEMANTIC_LABEL_VALUES,
    SEMANTIC_VALID_BIT,
    SOURCE_CHANNELS,
    PackContinuityTargetsTransform,
    PackedTargetLayout,
    build_source_segmentation,
    pack_targets_from_augmented_source,
    semantic_contact_mask,
    unpack_packed_targets,
)

__all__ = [
    "AffinityOffset",
    "AXIAL_19_DIRECTION_SET",
    "CORTICAL_CONTINUITY_SCHEMA_VERSION",
    "ContinuityTargets",
    "CorticalContinuityHeadSchema",
    "DENSE_39_DIRECTION_SET",
    "FlatHead",
    "PackContinuityTargetsTransform",
    "PackedTargetLayout",
    "RELATION_VALID_BIT",
    "RIM_VALID_BIT",
    "SCHEMA_PLANS_KEY",
    "SEMANTIC_CONTACT_LABEL",
    "SEMANTIC_CORTEX_LABEL",
    "SEMANTIC_IGNORE_LABEL",
    "SEMANTIC_LABEL_VALUES",
    "SEMANTIC_VALID_BIT",
    "SOURCE_CHANNELS",
    "SUPPORTED_DIRECTION_SETS",
    "build_continuity_targets",
    "build_cortical_continuity_schema",
    "build_source_segmentation",
    "pack_targets_from_augmented_source",
    "semantic_contact_mask",
    "unpack_packed_targets",
]
