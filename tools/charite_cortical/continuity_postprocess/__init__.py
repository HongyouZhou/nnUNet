"""Cortical-continuity driven fragment refinement.

This package deliberately keeps fragment identity inference on predicted
cortical evidence.  A provisional volume instance only supplies the support
that is passively partitioned after cortical grouping.
"""

from .config import ContinuityConfig
from .evaluation import (
    CalibrationEvaluation,
    InstanceEvaluation,
    MacroMetricEstimate,
    PartitionEvaluation,
    PatientMacroEvaluation,
    RiskCoveragePoint,
    aggregate_patient_macro,
    evaluate_instance_segmentation,
    evaluate_partition_grouping,
    evaluate_split_calibration,
    hungarian_match_iou_matrix,
)
from .graph import (
    GraphResourceLimit,
    MutexWatershedResult,
    SignedEdge,
    SignedGraph,
    build_signed_graph,
    mutex_watershed,
)
from .io import (
    ContinuityInputs,
    ContinuityResult,
    InstanceDecision,
    load_inputs_npz,
    save_inputs_npz,
    save_result,
)
from .oracle import (
    ControlledMerge,
    ControlledEvent,
    ControlledEventManifest,
    OracleEventResult,
    controlled_merge,
    controlled_event_manifest,
    oracle_affinity_logits,
    oracle_cortical_semantic,
    run_o1_event,
    run_o2_event,
)
from .pipeline import refine, refine_instances
from .propagation import PropagationResult, propagate_uniform
from .scoring import (
    AlwaysAbstainScorer,
    ConservativeSplitScorer,
    LogisticSplitScorer,
    ScorerFitResult,
    SplitFeatures,
    SplitScore,
    fit_logistic_split_scorer,
    load_scorer,
    save_scorer,
)
from .separator import refine_with_separator
from .support import hysteresis_cortex_support

__all__ = [
    "ContinuityConfig",
    "ContinuityInputs",
    "ContinuityResult",
    "ControlledMerge",
    "ControlledEvent",
    "ControlledEventManifest",
    "OracleEventResult",
    "InstanceDecision",
    "InstanceEvaluation",
    "PartitionEvaluation",
    "CalibrationEvaluation",
    "MacroMetricEstimate",
    "PatientMacroEvaluation",
    "RiskCoveragePoint",
    "MutexWatershedResult",
    "GraphResourceLimit",
    "PropagationResult",
    "SignedEdge",
    "SignedGraph",
    "AlwaysAbstainScorer",
    "ConservativeSplitScorer",
    "LogisticSplitScorer",
    "ScorerFitResult",
    "SplitFeatures",
    "SplitScore",
    "build_signed_graph",
    "aggregate_patient_macro",
    "controlled_merge",
    "controlled_event_manifest",
    "evaluate_instance_segmentation",
    "evaluate_partition_grouping",
    "evaluate_split_calibration",
    "hysteresis_cortex_support",
    "hungarian_match_iou_matrix",
    "load_inputs_npz",
    "mutex_watershed",
    "oracle_affinity_logits",
    "oracle_cortical_semantic",
    "run_o1_event",
    "run_o2_event",
    "propagate_uniform",
    "refine",
    "refine_instances",
    "refine_with_separator",
    "fit_logistic_split_scorer",
    "load_scorer",
    "save_inputs_npz",
    "save_result",
    "save_scorer",
]
