from __future__ import annotations

import dataclasses
import heapq
from collections import deque
from typing import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class PropagationResult:
    labels: np.ndarray
    distance_mm: np.ndarray
    unseeded_components: int


def _validate_spacing(spacing_zyx: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
        raise ValueError("spacing_zyx must contain three finite values")
    if np.any(spacing <= 0):
        raise ValueError("spacing_zyx must be positive")
    return spacing


def _component_coordinates(mask: np.ndarray) -> list[np.ndarray]:
    """Return six-connected component coordinates without requiring SciPy."""
    remaining = np.asarray(mask, dtype=bool).copy()
    shape = remaining.shape
    components: list[np.ndarray] = []
    while np.any(remaining):
        start = tuple(int(value) for value in np.argwhere(remaining)[0])
        remaining[start] = False
        queue = deque([start])
        points: list[tuple[int, int, int]] = []
        while queue:
            point = queue.popleft()
            points.append(point)
            for axis in range(3):
                for delta in (-1, 1):
                    neighbour = list(point)
                    neighbour[axis] += delta
                    if neighbour[axis] < 0 or neighbour[axis] >= shape[axis]:
                        continue
                    neighbour_tuple = tuple(neighbour)
                    if remaining[neighbour_tuple]:
                        remaining[neighbour_tuple] = False
                        queue.append(neighbour_tuple)
        components.append(np.asarray(points, dtype=np.int64))
    return components


def _nearest_seed_label(
    coordinates: np.ndarray,
    seed_labels: np.ndarray,
    spacing_zyx: np.ndarray,
) -> int:
    """Find a deterministic physically-nearest seed label for one component."""
    component_physical = coordinates.astype(np.float64) * spacing_zyx
    best_distance = np.inf
    best_label = 0
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    for label_id in np.unique(seed_labels):
        if label_id <= 0:
            continue
        seed_coordinates = np.argwhere(seed_labels == label_id)
        seed_physical = seed_coordinates.astype(np.float64) * spacing_zyx
        if cKDTree is not None:
            tree = cKDTree(seed_physical)
            label_best = float(np.min(tree.query(component_physical, k=1)[0]) ** 2)
        else:
            label_best = np.inf
            # Bound both axes in the no-SciPy fallback.
            for component_start in range(0, len(component_physical), 256):
                component_chunk = component_physical[
                    component_start : component_start + 256
                ]
                for seed_start in range(0, len(seed_physical), 4096):
                    seed_chunk = seed_physical[seed_start : seed_start + 4096]
                    squared = np.sum(
                        (
                            component_chunk[:, None, :]
                            - seed_chunk[None, :, :]
                        )
                        ** 2,
                        axis=2,
                    )
                    label_best = min(label_best, float(np.min(squared)))
        if label_best < best_distance or (
            np.isclose(label_best, best_distance) and int(label_id) < best_label
        ):
            best_distance = label_best
            best_label = int(label_id)
    if best_label <= 0:
        raise RuntimeError("could not assign an unseeded mask component")
    return best_label


def propagate_uniform(
    instance_mask: np.ndarray,
    seed_labels: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    tie_tolerance: float = 1e-9,
) -> PropagationResult:
    """Spacing-aware multi-source geodesic Voronoi inside fixed support."""
    mask = np.asarray(instance_mask, dtype=bool)
    seeds = np.asarray(seed_labels)
    spacing = _validate_spacing(spacing_zyx)
    if mask.ndim != 3 or seeds.shape != mask.shape:
        raise ValueError("instance_mask and seed_labels must share one 3D shape")
    if not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("seed_labels must have an integer dtype")
    if np.any(seeds < 0):
        raise ValueError("seed_labels must be non-negative")
    if np.any((seeds > 0) & ~mask):
        raise ValueError("all seeds must lie inside instance_mask")
    unique_seeds = np.unique(seeds)
    unique_seeds = unique_seeds[unique_seeds > 0]
    if unique_seeds.size == 0:
        raise ValueError("at least one positive seed label is required")
    expected = np.arange(1, int(unique_seeds.max()) + 1)
    if not np.array_equal(unique_seeds, expected):
        raise ValueError("seed labels must be consecutive positive integers")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")

    distance = np.full(mask.shape, np.inf, dtype=np.float64)
    labels = np.zeros(mask.shape, dtype=np.int32)
    queue: list[tuple[float, int, int, int, int]] = []
    for coordinate in np.argwhere(seeds > 0):
        point = tuple(int(value) for value in coordinate)
        label_id = int(seeds[point])
        distance[point] = 0.0
        labels[point] = label_id
        heapq.heappush(queue, (0.0, label_id, *point))

    shape = mask.shape
    while queue:
        current_distance, label_id, z, y, x = heapq.heappop(queue)
        point = (z, y, x)
        if current_distance > distance[point] + tie_tolerance:
            continue
        if (
            abs(current_distance - distance[point]) <= tie_tolerance
            and label_id != int(labels[point])
        ):
            continue
        for axis, step_cost in enumerate(spacing):
            for delta in (-1, 1):
                neighbour = [z, y, x]
                neighbour[axis] += delta
                if neighbour[axis] < 0 or neighbour[axis] >= shape[axis]:
                    continue
                neighbour_tuple = tuple(neighbour)
                if not mask[neighbour_tuple]:
                    continue
                proposed = current_distance + float(step_cost)
                old_distance = float(distance[neighbour_tuple])
                old_label = int(labels[neighbour_tuple])
                improve = proposed < old_distance - tie_tolerance
                tie_wins = (
                    abs(proposed - old_distance) <= tie_tolerance
                    and (old_label == 0 or label_id < old_label)
                )
                if improve or tie_wins:
                    distance[neighbour_tuple] = proposed
                    labels[neighbour_tuple] = label_id
                    heapq.heappush(
                        queue,
                        (proposed, label_id, *neighbour_tuple),
                    )

    uncovered = mask & (labels == 0)
    components = _component_coordinates(uncovered)
    for coordinates in components:
        label_id = _nearest_seed_label(coordinates, seeds, spacing)
        labels[tuple(coordinates.T)] = label_id
        # This fallback crosses an absent support connection; the distance is
        # diagnostic only and remains infinite.

    if not np.array_equal(labels > 0, mask):
        raise RuntimeError("propagation did not cover the exact input support")
    if np.any(labels[seeds > 0] != seeds[seeds > 0]):
        raise RuntimeError("propagation changed cortical seed ownership")
    if set(np.unique(labels[mask])) != set(int(value) for value in unique_seeds):
        raise RuntimeError("propagation lost a seed label")
    return PropagationResult(
        labels=labels,
        distance_mm=distance,
        unseeded_components=len(components),
    )
