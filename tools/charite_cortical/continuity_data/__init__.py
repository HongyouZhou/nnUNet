"""Data preparation primitives for cortical-continuity training.

The package is deliberately independent of nnU-Net's training stack.  It only
depends on the Python standard library and NumPy so dataset audits and target
construction can run before a GPU environment is available.
"""

from .schema import (
    DATASET_ID,
    DATASET_NAME,
    MANIFEST_SCHEMA,
    N_FOLDS,
    SCHEMA_VERSION,
    SPLIT_RANDOM_STATE,
    BuildConfig,
    canonical_json_hash,
    validate_manifest,
)
from .targets import (
    RELATION_VALID,
    RIM_VALID,
    SEPARATOR_IGNORE,
    SEMANTIC_VALID,
    CaseTargets,
    derive_case_targets,
)

__all__ = [
    "BuildConfig",
    "CaseTargets",
    "DATASET_ID",
    "DATASET_NAME",
    "MANIFEST_SCHEMA",
    "N_FOLDS",
    "RELATION_VALID",
    "RIM_VALID",
    "SEPARATOR_IGNORE",
    "SCHEMA_VERSION",
    "SEMANTIC_VALID",
    "SPLIT_RANDOM_STATE",
    "canonical_json_hash",
    "derive_case_targets",
    "validate_manifest",
]
