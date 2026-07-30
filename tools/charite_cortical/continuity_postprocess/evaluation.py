from __future__ import annotations

import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np


InstanceKind = Literal["instance"]


@dataclasses.dataclass(frozen=True)
class InstanceMatch:
    prediction_id: int
    ground_truth_id: int
    iou: float
    dice: float


@dataclasses.dataclass(frozen=True)
class InstanceEvaluation:
    prediction_count: int
    ground_truth_count: int
    matches: tuple[InstanceMatch, ...]
    unmatched_prediction_ids: tuple[int, ...]
    unmatched_ground_truth_ids: tuple[int, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    segmentation_quality: float
    recognition_quality: float
    panoptic_quality: float
    mean_matched_iou: float
    mean_matched_dice: float
    count_error: int
    vi_merge: float
    vi_split: float
    normalized_vi_merge: float
    normalized_vi_split: float
    all_child_recovery: bool
    intact_false_split: bool


@dataclasses.dataclass(frozen=True)
class PartitionEvaluation:
    prediction_count: int
    ground_truth_count: int
    evaluated_voxels: int
    prediction_coverage: float
    exact_k: bool
    adjusted_rand_index: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    vi_merge: float
    vi_split: float
    identity_grouping_correct: bool


@dataclasses.dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    accepted: int


@dataclasses.dataclass(frozen=True)
class CalibrationEvaluation:
    sample_count: int
    positive_rate: float
    brier_score: float
    expected_calibration_error: float
    risk_coverage: tuple[RiskCoveragePoint, ...]
    operating_threshold: Optional[float]
    intact_false_split_rate: Optional[float]


@dataclasses.dataclass(frozen=True)
class MacroMetricEstimate:
    mean: float
    ci_low: float
    ci_high: float


@dataclasses.dataclass(frozen=True)
class PatientMacroEvaluation:
    patient_count: int
    record_count: int
    bootstrap_iterations: int
    random_state: int
    metrics: Mapping[str, MacroMetricEstimate]


def _validate_instance_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _pairwise_iou_and_dice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    prediction_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    iou = np.zeros((len(prediction_ids), len(ground_truth_ids)), dtype=np.float64)
    dice = np.zeros_like(iou)
    prediction_sizes = {
        int(label): int(np.count_nonzero((prediction == label) & valid_mask))
        for label in prediction_ids
    }
    ground_truth_sizes = {
        int(label): int(np.count_nonzero((ground_truth == label) & valid_mask))
        for label in ground_truth_ids
    }
    for pred_index, prediction_id in enumerate(prediction_ids):
        prediction_mask = (prediction == prediction_id) & valid_mask
        for gt_index, ground_truth_id in enumerate(ground_truth_ids):
            intersection = int(
                np.count_nonzero(prediction_mask & (ground_truth == ground_truth_id))
            )
            pred_size = prediction_sizes[int(prediction_id)]
            gt_size = ground_truth_sizes[int(ground_truth_id)]
            union = pred_size + gt_size - intersection
            iou[pred_index, gt_index] = intersection / union if union else 0.0
            denominator = pred_size + gt_size
            dice[pred_index, gt_index] = (
                2.0 * intersection / denominator if denominator else 0.0
            )
    return iou, dice


def hungarian_match_iou_matrix(iou_matrix: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return row/column assignments maximizing total IoU."""
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("iou_matrix must be a finite 2D array")
    if matrix.size == 0:
        return ()
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for Hungarian instance matching"
        ) from error
    rows, columns = linear_sum_assignment(-matrix)
    return tuple((int(row), int(column)) for row, column in zip(rows, columns))


def _conditional_entropies(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    pred = prediction[evaluation_mask].astype(np.int64, copy=False)
    gt = ground_truth[evaluation_mask].astype(np.int64, copy=False)
    sample_count = int(pred.size)
    if sample_count <= 1:
        return 0.0, 0.0, 0.0, 0.0

    pred_values, pred_inverse = np.unique(pred, return_inverse=True)
    gt_values, gt_inverse = np.unique(gt, return_inverse=True)
    joint_index = pred_inverse * len(gt_values) + gt_inverse
    joint = np.bincount(
        joint_index,
        minlength=len(pred_values) * len(gt_values),
    ).reshape(len(pred_values), len(gt_values))
    joint_probability = joint / sample_count
    pred_probability = joint_probability.sum(axis=1)
    gt_probability = joint_probability.sum(axis=0)

    vi_merge = 0.0  # H(GT | Pred): merge error
    vi_split = 0.0  # H(Pred | GT): split error
    nonzero_rows, nonzero_columns = np.nonzero(joint)
    for row, column in zip(nonzero_rows, nonzero_columns):
        probability = float(joint_probability[row, column])
        vi_merge -= probability * np.log2(probability / pred_probability[row])
        vi_split -= probability * np.log2(probability / gt_probability[column])
    normalizer = np.log2(sample_count)
    return (
        float(vi_merge),
        float(vi_split),
        float(vi_merge / normalizer),
        float(vi_split / normalizer),
    )


def evaluate_instance_segmentation(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    prediction_kind: InstanceKind,
    ground_truth_kind: InstanceKind,
    valid_mask: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
    expected_ground_truth_ids: Optional[set[int]] = None,
) -> InstanceEvaluation:
    """Evaluate explicit instance maps without label-value type heuristics."""
    if prediction_kind != "instance" or ground_truth_kind != "instance":
        raise ValueError(
            "this evaluator requires explicit prediction_kind='instance' and "
            "ground_truth_kind='instance'"
        )
    pred = _validate_instance_array("prediction", prediction)
    gt = _validate_instance_array("ground_truth", ground_truth)
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground_truth shapes do not match")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if valid_mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != pred.shape:
            raise ValueError("valid_mask shape does not match prediction")

    prediction_ids = np.unique(pred[valid])
    prediction_ids = prediction_ids[prediction_ids > 0]
    ground_truth_ids = np.unique(gt[valid])
    ground_truth_ids = ground_truth_ids[ground_truth_ids > 0]
    iou, dice = _pairwise_iou_and_dice(
        pred,
        gt,
        prediction_ids,
        ground_truth_ids,
        valid,
    )
    assignments = hungarian_match_iou_matrix(iou)
    matches: list[InstanceMatch] = []
    matched_prediction: set[int] = set()
    matched_ground_truth: set[int] = set()
    for pred_index, gt_index in assignments:
        pair_iou = float(iou[pred_index, gt_index])
        if pair_iou < iou_threshold:
            continue
        prediction_id = int(prediction_ids[pred_index])
        ground_truth_id = int(ground_truth_ids[gt_index])
        matches.append(
            InstanceMatch(
                prediction_id=prediction_id,
                ground_truth_id=ground_truth_id,
                iou=pair_iou,
                dice=float(dice[pred_index, gt_index]),
            )
        )
        matched_prediction.add(prediction_id)
        matched_ground_truth.add(ground_truth_id)

    unmatched_prediction = tuple(
        int(value)
        for value in prediction_ids
        if int(value) not in matched_prediction
    )
    unmatched_ground_truth = tuple(
        int(value)
        for value in ground_truth_ids
        if int(value) not in matched_ground_truth
    )
    true_positives = len(matches)
    false_positives = len(unmatched_prediction)
    false_negatives = len(unmatched_ground_truth)
    segmentation_quality = (
        float(np.mean([match.iou for match in matches])) if matches else 0.0
    )
    denominator = (
        true_positives + 0.5 * false_positives + 0.5 * false_negatives
    )
    recognition_quality = true_positives / denominator if denominator else 1.0
    panoptic_quality = (
        float(sum(match.iou for match in matches) / denominator)
        if denominator
        else 1.0
    )
    evaluation_mask = valid & ((pred > 0) | (gt > 0))
    vi_merge, vi_split, normalized_merge, normalized_split = (
        _conditional_entropies(pred, gt, evaluation_mask)
    )

    expected = (
        set(int(value) for value in ground_truth_ids)
        if expected_ground_truth_ids is None
        else {int(value) for value in expected_ground_truth_ids}
    )
    if not expected.issubset(set(int(value) for value in ground_truth_ids)):
        raise ValueError("expected_ground_truth_ids contains an absent GT ID")
    matched_expected = {
        match.ground_truth_id for match in matches if match.ground_truth_id in expected
    }
    all_child_recovery = bool(
        expected
        and matched_expected == expected
        and false_positives == 0
    )
    intact_false_split = bool(
        len(ground_truth_ids) == 1 and len(prediction_ids) > 1
    )
    return InstanceEvaluation(
        prediction_count=len(prediction_ids),
        ground_truth_count=len(ground_truth_ids),
        matches=tuple(matches),
        unmatched_prediction_ids=unmatched_prediction,
        unmatched_ground_truth_ids=unmatched_ground_truth,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        segmentation_quality=segmentation_quality,
        recognition_quality=float(recognition_quality),
        panoptic_quality=panoptic_quality,
        mean_matched_iou=segmentation_quality,
        mean_matched_dice=(
            float(np.mean([match.dice for match in matches])) if matches else 0.0
        ),
        count_error=int(len(prediction_ids) - len(ground_truth_ids)),
        vi_merge=vi_merge,
        vi_split=vi_split,
        normalized_vi_merge=normalized_merge,
        normalized_vi_split=normalized_split,
        all_child_recovery=all_child_recovery,
        intact_false_split=intact_false_split,
    )


def evaluate_partition_grouping(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    prediction_kind: InstanceKind,
    ground_truth_kind: InstanceKind,
    valid_mask: Optional[np.ndarray] = None,
) -> PartitionEvaluation:
    """Evaluate label-invariant identity grouping on a declared voxel set."""

    if prediction_kind != "instance" or ground_truth_kind != "instance":
        raise ValueError(
            "partition evaluation requires explicit instance kinds"
        )
    pred = _validate_instance_array("prediction", prediction)
    gt = _validate_instance_array("ground_truth", ground_truth)
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground_truth shapes do not match")
    if valid_mask is None:
        valid = gt > 0
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != gt.shape:
            raise ValueError("valid_mask shape does not match partitions")
        valid &= gt > 0
    pred_values = pred[valid].astype(np.int64, copy=False)
    gt_values = gt[valid].astype(np.int64, copy=False)
    sample_count = int(gt_values.size)
    if sample_count == 0:
        raise ValueError("partition evaluation has no valid GT voxels")

    pred_unique, pred_inverse = np.unique(pred_values, return_inverse=True)
    gt_unique, gt_inverse = np.unique(gt_values, return_inverse=True)
    joint = np.bincount(
        pred_inverse * len(gt_unique) + gt_inverse,
        minlength=len(pred_unique) * len(gt_unique),
    ).reshape(len(pred_unique), len(gt_unique))
    row_counts = joint.sum(axis=1)
    column_counts = joint.sum(axis=0)
    same_both = float(np.sum(_combination_two(joint)))
    predicted_pairs = float(np.sum(_combination_two(row_counts)))
    ground_truth_pairs = float(np.sum(_combination_two(column_counts)))
    all_pairs = float(_combination_two(np.asarray(sample_count)))

    if all_pairs == 0:
        adjusted_rand = 1.0
    else:
        expected = predicted_pairs * ground_truth_pairs / all_pairs
        maximum = 0.5 * (predicted_pairs + ground_truth_pairs)
        denominator = maximum - expected
        adjusted_rand = (
            1.0
            if abs(denominator) <= 1e-12
            and abs(same_both - maximum) <= 1e-12
            else (
                0.0
                if abs(denominator) <= 1e-12
                else float((same_both - expected) / denominator)
            )
        )
    pairwise_precision = (
        same_both / predicted_pairs if predicted_pairs > 0 else 1.0
    )
    pairwise_recall = (
        same_both / ground_truth_pairs if ground_truth_pairs > 0 else 1.0
    )
    pairwise_f1 = (
        2.0
        * pairwise_precision
        * pairwise_recall
        / (pairwise_precision + pairwise_recall)
        if pairwise_precision + pairwise_recall > 0
        else 0.0
    )
    positive_prediction_ids = {
        int(value) for value in pred_unique if int(value) > 0
    }
    positive_ground_truth_ids = {
        int(value) for value in gt_unique if int(value) > 0
    }
    coverage = float(np.mean(pred_values > 0))
    exact_k = len(positive_prediction_ids) == len(positive_ground_truth_ids)
    vi_merge, vi_split, _, _ = _conditional_entropies(
        pred,
        gt,
        valid,
    )
    identity_correct = bool(
        exact_k
        and coverage == 1.0
        and abs(adjusted_rand - 1.0) <= 1e-12
    )
    return PartitionEvaluation(
        prediction_count=len(positive_prediction_ids),
        ground_truth_count=len(positive_ground_truth_ids),
        evaluated_voxels=sample_count,
        prediction_coverage=coverage,
        exact_k=exact_k,
        adjusted_rand_index=adjusted_rand,
        pairwise_precision=float(pairwise_precision),
        pairwise_recall=float(pairwise_recall),
        pairwise_f1=float(pairwise_f1),
        vi_merge=vi_merge,
        vi_split=vi_split,
        identity_grouping_correct=identity_correct,
    )


def evaluate_split_calibration(
    probabilities: Sequence[float] | np.ndarray,
    proposal_correct: Sequence[bool] | np.ndarray,
    *,
    bins: int = 10,
    operating_threshold: Optional[float] = None,
    intact_negative: Optional[Sequence[bool] | np.ndarray] = None,
) -> CalibrationEvaluation:
    """Report Brier/ECE and tied-threshold risk-coverage on OOF proposals."""

    probability = np.asarray(probabilities, dtype=np.float64)
    correct = np.asarray(proposal_correct, dtype=bool)
    if probability.ndim != 1 or correct.shape != probability.shape:
        raise ValueError("probabilities and proposal_correct must have shape (N,)")
    if probability.size == 0:
        raise ValueError("calibration evaluation requires at least one sample")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.minimum(
        np.searchsorted(edges, probability, side="right") - 1,
        bins - 1,
    )
    expected_calibration_error = 0.0
    for index in range(bins):
        selected = bin_index == index
        if not np.any(selected):
            continue
        confidence = float(np.mean(probability[selected]))
        accuracy = float(np.mean(correct[selected]))
        expected_calibration_error += (
            float(np.mean(selected)) * abs(accuracy - confidence)
        )

    order = np.argsort(-probability, kind="stable")
    ordered_probability = probability[order]
    ordered_correct = correct[order]
    cumulative_correct = np.cumsum(ordered_correct, dtype=np.int64)
    points: list[RiskCoveragePoint] = []
    for index in range(len(order)):
        is_group_end = (
            index == len(order) - 1
            or ordered_probability[index + 1] != ordered_probability[index]
        )
        if not is_group_end:
            continue
        accepted = index + 1
        points.append(
            RiskCoveragePoint(
                threshold=float(ordered_probability[index]),
                coverage=float(accepted / len(order)),
                risk=float(1.0 - cumulative_correct[index] / accepted),
                accepted=accepted,
            )
        )

    false_split_rate: Optional[float] = None
    if operating_threshold is not None:
        if not 0 <= operating_threshold <= np.nextafter(1.0, 2.0):
            raise ValueError(
                "operating_threshold must be in [0, nextafter(1,+inf)]"
            )
        if intact_negative is None:
            raise ValueError(
                "intact_negative is required with an operating_threshold"
            )
        intact = np.asarray(intact_negative, dtype=bool)
        if intact.shape != probability.shape:
            raise ValueError("intact_negative must have shape (N,)")
        false_split_rate = (
            float(np.mean(probability[intact] >= operating_threshold))
            if np.any(intact)
            else None
        )
    return CalibrationEvaluation(
        sample_count=int(probability.size),
        positive_rate=float(np.mean(correct)),
        brier_score=float(np.mean((probability - correct.astype(float)) ** 2)),
        expected_calibration_error=float(expected_calibration_error),
        risk_coverage=tuple(points),
        operating_threshold=operating_threshold,
        intact_false_split_rate=false_split_rate,
    )


def aggregate_patient_macro(
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    patient_key: str = "patient_id",
    metrics_key: str = "metrics",
    bootstrap_iterations: int = 2000,
    random_state: int = 20260729,
) -> PatientMacroEvaluation:
    """Aggregate record metrics by patient with a deterministic bootstrap CI."""

    names = tuple(str(name) for name in metric_names)
    if not records:
        raise ValueError("patient-macro aggregation requires records")
    if not names or len(names) != len(set(names)):
        raise ValueError("metric_names must be a non-empty unique sequence")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    grouped: dict[str, dict[str, list[float]]] = {}
    for index, record in enumerate(records):
        if patient_key not in record:
            raise ValueError(f"records[{index}] is missing {patient_key}")
        patient_id = str(record[patient_key])
        if not patient_id:
            raise ValueError(f"records[{index}] has an empty patient ID")
        metrics = record.get(metrics_key)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"records[{index}].{metrics_key} must be an object")
        destination = grouped.setdefault(
            patient_id,
            {name: [] for name in names},
        )
        for name in names:
            if name not in metrics:
                raise ValueError(
                    f"records[{index}].{metrics_key} is missing {name}"
                )
            value = float(metrics[name])
            if not np.isfinite(value):
                raise ValueError(
                    f"records[{index}].{metrics_key}.{name} is not finite"
                )
            destination[name].append(value)

    patient_ids = sorted(grouped)
    patient_matrix = np.asarray(
        [
            [
                float(np.mean(grouped[patient_id][name]))
                for name in names
            ]
            for patient_id in patient_ids
        ],
        dtype=np.float64,
    )
    random_generator = np.random.RandomState(random_state)
    indices = random_generator.randint(
        0,
        len(patient_ids),
        size=(bootstrap_iterations, len(patient_ids)),
    )
    bootstrap_means = patient_matrix[indices].mean(axis=1)
    estimates = {
        name: MacroMetricEstimate(
            mean=float(patient_matrix[:, column].mean()),
            ci_low=float(np.quantile(bootstrap_means[:, column], 0.025)),
            ci_high=float(np.quantile(bootstrap_means[:, column], 0.975)),
        )
        for column, name in enumerate(names)
    }
    return PatientMacroEvaluation(
        patient_count=len(patient_ids),
        record_count=len(records),
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
        metrics=estimates,
    )


def _combination_two(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array * (array - 1.0) / 2.0
