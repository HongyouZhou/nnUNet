"""Dependency-light target-aware patch-sampling policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


CONTINUITY_CONTACT_KEY = "continuity_contact"
CONTINUITY_INSTANCE_KEY_PREFIX = "continuity_instance"
CONTINUITY_INSTANCES_PROPERTY = "continuity_instance_cortex"
CONTINUITY_SUPPORT_KEY = "continuity_support"

CONTACT_CATEGORY = "contact"
INSTANCE_CATEGORY = "instance"
SUPPORT_CATEGORY = "support"
RANDOM_CATEGORY = "random"

# Frozen formal schedule: contact / per-instance cortex / support / random.
DEFAULT_SAMPLING_WEIGHTS = {
    CONTACT_CATEGORY: 0.40,
    INSTANCE_CATEGORY: 0.30,
    SUPPORT_CATEGORY: 0.20,
    RANDOM_CATEGORY: 0.10,
}


def available_sampling_distribution(
    class_locations: Mapping[Any, Any],
    sampling_weights: Mapping[str, float] = DEFAULT_SAMPLING_WEIGHTS,
) -> dict[str, float]:
    """Renormalize the 40/30/20/10 policy over categories present in a case."""

    _validate_contract_keys(class_locations)
    weights = _validate_weights(sampling_weights)
    available = {
        CONTACT_CATEGORY: _has_coordinates(class_locations[CONTINUITY_CONTACT_KEY]),
        INSTANCE_CATEGORY: bool(_eligible_instance_keys(class_locations)),
        SUPPORT_CATEGORY: _has_coordinates(class_locations[CONTINUITY_SUPPORT_KEY]),
        RANDOM_CATEGORY: True,
    }
    active = {
        category: weight
        for category, weight in weights.items()
        if available[category] and weight > 0
    }
    total = float(sum(active.values()))
    if total <= 0:
        raise ValueError("At least one available continuity sampling category must have positive weight")
    if abs(total - 1.0) <= 1e-12:
        return active
    return {category: weight / total for category, weight in active.items()}


def sample_patch_center(
    class_locations: Mapping[Any, Any],
    *,
    sampling_weights: Mapping[str, float] = DEFAULT_SAMPLING_WEIGHTS,
    rng: Any = np.random,
) -> tuple[str, np.ndarray | None]:
    """Choose a category and a centre; instances are selected uniformly first."""

    distribution = available_sampling_distribution(class_locations, sampling_weights)
    categories = tuple(distribution)
    probabilities = np.asarray([distribution[i] for i in categories], dtype=np.float64)
    category = str(rng.choice(categories, p=probabilities))
    if category == RANDOM_CATEGORY:
        return category, None
    if category == CONTACT_CATEGORY:
        coordinates = _validated_coordinates(
            class_locations[CONTINUITY_CONTACT_KEY],
            CONTINUITY_CONTACT_KEY,
        )
    elif category == SUPPORT_CATEGORY:
        coordinates = _validated_coordinates(
            class_locations[CONTINUITY_SUPPORT_KEY],
            CONTINUITY_SUPPORT_KEY,
        )
    elif category == INSTANCE_CATEGORY:
        instance_keys = _eligible_instance_keys(class_locations)
        key = instance_keys[int(rng.choice(len(instance_keys)))]
        coordinates = _validated_coordinates(class_locations[key], str(key))
    else:  # pragma: no cover - distribution construction makes this unreachable.
        raise AssertionError(category)
    return category, np.asarray(coordinates[int(rng.choice(len(coordinates)))], dtype=np.int64)


def _validate_contract_keys(class_locations: Mapping[Any, Any]) -> None:
    if not isinstance(class_locations, Mapping):
        raise RuntimeError(
            "Cortical continuity training requires preprocessed class_locations; "
            f"got {type(class_locations).__name__}"
        )
    missing = [
        key
        for key in (CONTINUITY_CONTACT_KEY, CONTINUITY_SUPPORT_KEY)
        if key not in class_locations
    ]
    if missing:
        raise RuntimeError(
            "Preprocessed case is missing cortical-continuity sampling metadata "
            f"{missing}. Re-run preprocessing with ChariteCorticalPreprocessor."
        )
    _validated_coordinates(class_locations[CONTINUITY_CONTACT_KEY], CONTINUITY_CONTACT_KEY)
    _validated_coordinates(class_locations[CONTINUITY_SUPPORT_KEY], CONTINUITY_SUPPORT_KEY)
    for key in _instance_keys(class_locations):
        _validated_coordinates(class_locations[key], str(key))


def _validate_weights(sampling_weights: Mapping[str, float]) -> dict[str, float]:
    expected = {
        CONTACT_CATEGORY,
        INSTANCE_CATEGORY,
        SUPPORT_CATEGORY,
        RANDOM_CATEGORY,
    }
    if set(sampling_weights) != expected:
        raise ValueError(f"sampling_weights must define exactly {sorted(expected)}")
    result = {key: float(value) for key, value in sampling_weights.items()}
    if any(not np.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("sampling weights must be finite and non-negative")
    return result


def _instance_keys(class_locations: Mapping[Any, Any]) -> list[tuple[str, int]]:
    result = []
    for key in class_locations:
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == CONTINUITY_INSTANCE_KEY_PREFIX
        ):
            try:
                instance_id = int(key[1])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid continuity instance sampling key {key!r}") from exc
            if instance_id <= 0:
                raise RuntimeError(f"Continuity instance sampling ID must be positive, got {key!r}")
            result.append((CONTINUITY_INSTANCE_KEY_PREFIX, instance_id))
    return sorted(result, key=lambda item: item[1])


def _eligible_instance_keys(class_locations: Mapping[Any, Any]) -> list[tuple[str, int]]:
    return [
        key
        for key in _instance_keys(class_locations)
        if _has_coordinates(class_locations[key])
    ]


def _has_coordinates(value: Any) -> bool:
    coordinates = np.asarray(value)
    return coordinates.ndim == 2 and coordinates.shape[0] > 0


def _validated_coordinates(value: Any, name: str) -> np.ndarray:
    coordinates = np.asarray(value)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise RuntimeError(
            f"{name} sampling coordinates must have shape [N,4] (channel,z,y,x), "
            f"got {coordinates.shape}"
        )
    if coordinates.size and (
        not np.issubdtype(coordinates.dtype, np.integer)
        or np.any(coordinates[:, 0] != 0)
    ):
        raise RuntimeError(f"{name} sampling coordinates must be integer with channel column zero")
    return coordinates
