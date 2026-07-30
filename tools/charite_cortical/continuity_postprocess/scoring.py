from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .config import ContinuityConfig


FEATURE_NAMES = (
    "normalized_energy_gain",
    "cluster_count",
    "min_high_confidence_volume_mm3",
    "min_repulsive_edges",
    "min_piece_fraction",
    "rim_support",
    "unseeded_volume_components",
)


@dataclasses.dataclass(frozen=True)
class SplitFeatures:
    normalized_energy_gain: float
    cluster_count: int
    min_high_confidence_volume_mm3: float
    min_repulsive_edges: int
    min_piece_fraction: float
    rim_support: float
    unseeded_volume_components: int

    def as_array(self) -> np.ndarray:
        values = np.asarray(
            [float(getattr(self, name)) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("split features must be finite")
        return values

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SplitFeatures":
        missing = set(FEATURE_NAMES).difference(values)
        unknown = set(values).difference(FEATURE_NAMES)
        if missing or unknown:
            raise ValueError(
                f"split feature schema mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**{name: values[name] for name in FEATURE_NAMES})


@dataclasses.dataclass(frozen=True)
class SplitScore:
    accepted: bool
    probability: float
    reason: str


class SplitScorer(Protocol):
    def evaluate(self, features: SplitFeatures) -> SplitScore:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...


@dataclasses.dataclass(frozen=True)
class ConservativeSplitScorer:
    """Fixed-threshold safe default used before learned calibration exists."""

    min_normalized_energy_gain: float
    min_high_confidence_volume_mm3: float
    min_repulsive_edges_per_cluster: int

    @classmethod
    def from_config(cls, config: ContinuityConfig) -> "ConservativeSplitScorer":
        return cls(
            min_normalized_energy_gain=config.min_normalized_energy_gain,
            min_high_confidence_volume_mm3=(
                config.min_high_confidence_volume_mm3
            ),
            min_repulsive_edges_per_cluster=(
                config.min_repulsive_edges_per_cluster
            ),
        )

    def evaluate(self, features: SplitFeatures) -> SplitScore:
        if (
            features.min_high_confidence_volume_mm3
            < self.min_high_confidence_volume_mm3
        ):
            return SplitScore(False, 0.0, "weak_cortical_cluster")
        if features.min_repulsive_edges < self.min_repulsive_edges_per_cluster:
            return SplitScore(False, 0.0, "insufficient_repulsive_evidence")
        if features.normalized_energy_gain < self.min_normalized_energy_gain:
            return SplitScore(False, 0.0, "insufficient_energy_gain")
        # A zero-gain split is unsupported even when the configured floor is
        # zero. This makes disconnected false-positive cortex abstain safely.
        if features.normalized_energy_gain <= 0:
            return SplitScore(False, 0.0, "nonpositive_energy_gain")
        return SplitScore(True, 1.0, "fixed_thresholds_passed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "conservative",
            "min_normalized_energy_gain": self.min_normalized_energy_gain,
            "min_high_confidence_volume_mm3": (
                self.min_high_confidence_volume_mm3
            ),
            "min_repulsive_edges_per_cluster": (
                self.min_repulsive_edges_per_cluster
            ),
        }


@dataclasses.dataclass(frozen=True)
class AlwaysAbstainScorer:
    reason: str = "calibration_unavailable"

    def evaluate(self, features: SplitFeatures) -> SplitScore:
        features.as_array()
        return SplitScore(False, 0.0, self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "always_abstain", "reason": self.reason}


def _sigmoid_scalar(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


@dataclasses.dataclass(frozen=True)
class LogisticSplitScorer:
    """Standardized L2 logistic scorer with a frozen safe operating threshold."""

    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float
    l2_penalty: float
    max_intact_false_split_rate: float

    def __post_init__(self) -> None:
        for name in ("weights", "feature_mean", "feature_scale"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (len(FEATURE_NAMES),):
                raise ValueError(
                    f"{name} must have shape ({len(FEATURE_NAMES)},)"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
        if np.any(np.asarray(self.feature_scale) <= 0):
            raise ValueError("feature_scale must be positive")
        if not np.isfinite(self.bias):
            raise ValueError("bias must be finite")
        if not 0 <= self.threshold <= np.nextafter(1.0, 2.0):
            raise ValueError("threshold must be in [0, nextafter(1, +inf)]")

    def probability(self, features: SplitFeatures) -> float:
        values = features.as_array()
        standardized = (values - self.feature_mean) / self.feature_scale
        logit = float(np.dot(self.weights, standardized) + self.bias)
        return _sigmoid_scalar(logit)

    def evaluate(self, features: SplitFeatures) -> SplitScore:
        probability = self.probability(features)
        accepted = probability >= self.threshold
        return SplitScore(
            accepted=accepted,
            probability=probability,
            reason=(
                "logistic_threshold_passed"
                if accepted
                else "logistic_threshold_failed"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "l2_logistic",
            "feature_names": list(FEATURE_NAMES),
            "weights": np.asarray(self.weights).tolist(),
            "bias": float(self.bias),
            "feature_mean": np.asarray(self.feature_mean).tolist(),
            "feature_scale": np.asarray(self.feature_scale).tolist(),
            "threshold": float(self.threshold),
            "l2_penalty": float(self.l2_penalty),
            "max_intact_false_split_rate": float(
                self.max_intact_false_split_rate
            ),
        }


@dataclasses.dataclass(frozen=True)
class ScorerFitResult:
    scorer: SplitScorer
    status: str
    sample_count: int
    intact_count: int
    accepted_true_positives: int
    intact_false_split_rate: float


def _feature_matrix(
    features: Sequence[SplitFeatures] | np.ndarray,
) -> np.ndarray:
    if isinstance(features, np.ndarray):
        matrix = np.asarray(features, dtype=np.float64)
    else:
        matrix = np.stack([feature.as_array() for feature in features], axis=0)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"feature matrix must have shape (N, {len(FEATURE_NAMES)})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix contains NaN or infinite values")
    return matrix


def _fit_l2_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    sample_count, feature_count = standardized.shape
    augmented = np.concatenate(
        (standardized, np.ones((sample_count, 1), dtype=np.float64)),
        axis=1,
    )
    parameters = np.zeros(feature_count + 1, dtype=np.float64)
    prevalence = float(np.clip(labels.mean(), 1e-4, 1 - 1e-4))
    parameters[-1] = np.log(prevalence / (1.0 - prevalence))
    regularizer = np.eye(feature_count + 1, dtype=np.float64)
    regularizer[-1, -1] = 0.0

    for _ in range(max_iterations):
        logits = augmented @ parameters
        probabilities = np.empty_like(logits)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        negative_exp = np.exp(logits[~positive])
        probabilities[~positive] = negative_exp / (1.0 + negative_exp)
        residual = probabilities - labels
        gradient = (
            augmented.T @ residual / sample_count
            + l2_penalty * (regularizer @ parameters)
        )
        curvature = probabilities * (1.0 - probabilities)
        hessian = (
            (augmented.T * curvature) @ augmented / sample_count
            + l2_penalty * regularizer
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        parameters -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return parameters[:-1], float(parameters[-1]), mean, scale


def fit_logistic_split_scorer(
    features: Sequence[SplitFeatures] | np.ndarray,
    proposal_correct: Sequence[bool] | np.ndarray,
    intact_negative: Sequence[bool] | np.ndarray,
    *,
    l2_penalty: float = 1.0,
    max_intact_false_split_rate: float = 0.05,
    min_samples: int = 20,
    min_intact_samples: int = 10,
    max_iterations: int = 100,
) -> ScorerFitResult:
    """Fit and freeze a scorer under an intact false-split constraint.

    Insufficient data, one-class labels, or an operating set that cannot
    accept at least one correct proposal safely returns ``AlwaysAbstainScorer``.
    """
    matrix = _feature_matrix(features)
    labels = np.asarray(proposal_correct, dtype=bool)
    intact = np.asarray(intact_negative, dtype=bool)
    if labels.shape != (matrix.shape[0],) or intact.shape != labels.shape:
        raise ValueError("labels and intact_negative must have shape (N,)")
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")
    if not 0 <= max_intact_false_split_rate <= 1:
        raise ValueError("max_intact_false_split_rate must be in [0, 1]")

    def abstain(reason: str) -> ScorerFitResult:
        return ScorerFitResult(
            scorer=AlwaysAbstainScorer(reason),
            status=reason,
            sample_count=int(matrix.shape[0]),
            intact_count=int(intact.sum()),
            accepted_true_positives=0,
            intact_false_split_rate=0.0,
        )

    if matrix.shape[0] < min_samples:
        return abstain("insufficient_calibration_samples")
    if int(intact.sum()) < min_intact_samples:
        return abstain("insufficient_intact_negatives")
    if np.unique(labels).size < 2:
        return abstain("calibration_requires_two_classes")

    weights, bias, mean, scale = _fit_l2_logistic(
        matrix,
        labels.astype(np.float64),
        l2_penalty,
        max_iterations,
    )
    standardized = (matrix - mean) / scale
    logits = standardized @ weights + bias
    probabilities = np.asarray(
        [_sigmoid_scalar(float(value)) for value in logits],
        dtype=np.float64,
    )
    candidates = sorted(
        {float(value) for value in probabilities},
        reverse=True,
    )
    candidates.append(float(np.nextafter(1.0, 2.0)))

    best: tuple[int, float, float] | None = None
    for threshold in candidates:
        accepted = probabilities >= threshold
        false_rate = float(np.mean(accepted[intact]))
        if false_rate > max_intact_false_split_rate + 1e-12:
            continue
        true_positives = int(np.count_nonzero(accepted & labels))
        # Prefer coverage of correct proposals, then lower intact risk, then a
        # higher (more conservative) threshold.
        candidate = (true_positives, -false_rate, threshold)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] == 0:
        return abstain("no_safe_positive_operating_point")

    threshold = float(best[2])
    accepted = probabilities >= threshold
    false_rate = float(np.mean(accepted[intact]))
    scorer = LogisticSplitScorer(
        weights=weights,
        bias=bias,
        feature_mean=mean,
        feature_scale=scale,
        threshold=threshold,
        l2_penalty=l2_penalty,
        max_intact_false_split_rate=max_intact_false_split_rate,
    )
    return ScorerFitResult(
        scorer=scorer,
        status="ok",
        sample_count=int(matrix.shape[0]),
        intact_count=int(intact.sum()),
        accepted_true_positives=int(np.count_nonzero(accepted & labels)),
        intact_false_split_rate=false_rate,
    )


def scorer_from_dict(values: Mapping[str, Any]) -> SplitScorer:
    scorer_type = values.get("type")
    if scorer_type == "always_abstain":
        return AlwaysAbstainScorer(str(values.get("reason", "calibration_unavailable")))
    if scorer_type == "conservative":
        return ConservativeSplitScorer(
            min_normalized_energy_gain=float(
                values["min_normalized_energy_gain"]
            ),
            min_high_confidence_volume_mm3=float(
                values["min_high_confidence_volume_mm3"]
            ),
            min_repulsive_edges_per_cluster=int(
                values["min_repulsive_edges_per_cluster"]
            ),
        )
    if scorer_type == "l2_logistic":
        if tuple(values.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("serialized scorer feature schema does not match")
        return LogisticSplitScorer(
            weights=np.asarray(values["weights"], dtype=np.float64),
            bias=float(values["bias"]),
            feature_mean=np.asarray(values["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(values["feature_scale"], dtype=np.float64),
            threshold=float(values["threshold"]),
            l2_penalty=float(values["l2_penalty"]),
            max_intact_false_split_rate=float(
                values["max_intact_false_split_rate"]
            ),
        )
    raise ValueError(f"unsupported split scorer type: {scorer_type!r}")


def save_scorer(path: str | Path, scorer: SplitScorer) -> None:
    Path(path).write_text(
        json.dumps(scorer.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_scorer(path: str | Path) -> SplitScorer:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("split scorer JSON must contain an object")
    return scorer_from_dict(values)
