from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

from .config import ContinuityConfig
from .evaluation import InstanceEvaluation, evaluate_instance_segmentation
from .io import InstanceDecision
from .pipeline import refine
from .propagation import propagate_uniform


@dataclasses.dataclass(frozen=True)
class ControlledMerge:
    provisional_instances: np.ndarray
    child_ids: tuple[int, ...]
    merged_id: int
    merged_support: np.ndarray


@dataclasses.dataclass(frozen=True)
class ControlledEvent:
    event_id: str
    kind: Literal["binary", "multiway", "intact"]
    child_ids: tuple[int, ...]
    adjacency_edges: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "child_ids": list(self.child_ids),
            "adjacency_edges": [
                {"first": first, "second": second, "distance_mm": distance}
                for first, second, distance in self.adjacency_edges
            ],
        }


@dataclasses.dataclass(frozen=True)
class ControlledEventManifest:
    spacing_zyx_mm: tuple[float, float, float]
    maximum_surface_distance_mm: float
    eligible_fragment_ids: tuple[int, ...]
    cortex_missing_fragment_ids: tuple[int, ...]
    events: tuple[ControlledEvent, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "spacing_zyx_mm": list(self.spacing_zyx_mm),
            "maximum_surface_distance_mm": self.maximum_surface_distance_mm,
            "eligible_fragment_ids": list(self.eligible_fragment_ids),
            "cortex_missing_fragment_ids": list(
                self.cortex_missing_fragment_ids
            ),
            "events": [event.to_dict() for event in self.events],
        }


@dataclasses.dataclass(frozen=True)
class OracleEventResult:
    mode: Literal["O1", "O2"]
    event: ControlledEvent
    prediction: np.ndarray
    ground_truth: np.ndarray
    evaluation: InstanceEvaluation
    cortical_grouping_evaluation: InstanceEvaluation
    decisions: tuple[InstanceDecision, ...] = ()


def controlled_merge(
    fragment_instances: np.ndarray,
    child_ids: Iterable[int],
    *,
    merged_id: int | None = None,
) -> ControlledMerge:
    """Create a deterministic controlled merge without changing image support."""
    fragments = np.asarray(fragment_instances)
    if fragments.ndim != 3 or not np.issubdtype(fragments.dtype, np.integer):
        raise ValueError("fragment_instances must be a 3D integer array")
    if np.any(fragments < 0):
        raise ValueError("fragment_instances must be non-negative")
    children = tuple(sorted({int(value) for value in child_ids}))
    if len(children) < 2 or any(value <= 0 for value in children):
        raise ValueError("controlled merge requires at least two positive child IDs")
    present = set(int(value) for value in np.unique(fragments))
    missing = set(children).difference(present)
    if missing:
        raise ValueError(f"child IDs are absent from fragment_instances: {sorted(missing)}")
    target = children[0] if merged_id is None else int(merged_id)
    if target <= 0:
        raise ValueError("merged_id must be positive")
    if target in present and target not in children:
        raise ValueError("merged_id collides with an unrelated instance")

    support = np.isin(fragments, children)
    provisional = fragments.copy()
    provisional[support] = target
    if not np.array_equal(provisional > 0, fragments > 0):
        raise RuntimeError("controlled merge changed foreground support")
    return ControlledMerge(
        provisional_instances=provisional,
        child_ids=children,
        merged_id=target,
        merged_support=support,
    )


def oracle_cortical_semantic(cortical_instances: np.ndarray) -> np.ndarray:
    cortical = np.asarray(cortical_instances)
    if cortical.ndim != 3 or not np.issubdtype(cortical.dtype, np.integer):
        raise ValueError("cortical_instances must be a 3D integer array")
    if np.any(cortical < 0):
        raise ValueError("cortical_instances must be non-negative")
    return (cortical > 0).astype(np.float32)


def _paired_slices(
    shape: tuple[int, int, int],
    offset: np.ndarray,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source: list[slice] = []
    target: list[slice] = []
    for size, raw_delta in zip(shape, offset):
        delta = int(raw_delta)
        if abs(delta) >= size:
            return (
                (slice(0, 0), slice(0, 0), slice(0, 0)),
                (slice(0, 0), slice(0, 0), slice(0, 0)),
            )
        if delta >= 0:
            source.append(slice(0, size - delta))
            target.append(slice(delta, size))
        else:
            source.append(slice(-delta, size))
            target.append(slice(0, size + delta))
    return tuple(source), tuple(target)


def oracle_affinity_logits(
    cortical_instances: np.ndarray,
    affinity_offsets_zyx: np.ndarray,
    *,
    same_logit: float = 20.0,
    different_logit: float = -20.0,
) -> np.ndarray:
    """Generate perfect relation logits on valid cortical endpoint pairs."""
    cortical = np.asarray(cortical_instances)
    offsets = np.asarray(affinity_offsets_zyx)
    if cortical.ndim != 3 or not np.issubdtype(cortical.dtype, np.integer):
        raise ValueError("cortical_instances must be a 3D integer array")
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("affinity_offsets_zyx must have shape (E, 3)")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("affinity offsets must be integers")
    if not np.isfinite(same_logit) or not np.isfinite(different_logit):
        raise ValueError("oracle logits must be finite")
    if same_logit <= 0 or different_logit >= 0:
        raise ValueError("same_logit must be positive and different_logit negative")

    logits = np.zeros((len(offsets),) + cortical.shape, dtype=np.float32)
    for edge_index, offset in enumerate(offsets):
        source_slice, target_slice = _paired_slices(cortical.shape, offset)
        source = cortical[source_slice]
        target = cortical[target_slice]
        valid = (source > 0) & (target > 0)
        same = valid & (source == target)
        different = valid & (source != target)
        channel = logits[edge_index]
        local = channel[source_slice]
        local[same] = same_logit
        local[different] = different_logit
    return logits


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    surface = np.asarray(mask, dtype=bool).copy()
    for axis in range(3):
        for delta in (-1, 1):
            neighbour = np.roll(mask, shift=delta, axis=axis)
            boundary = [slice(None)] * 3
            boundary[axis] = 0 if delta == 1 else -1
            neighbour[tuple(boundary)] = False
            surface &= neighbour
    return np.asarray(mask, dtype=bool) & ~surface


def _minimum_physical_distance(
    first_coordinates: np.ndarray,
    second_coordinates: np.ndarray,
    spacing_zyx: np.ndarray,
) -> float:
    first = first_coordinates.astype(np.float64) * spacing_zyx
    second = second_coordinates.astype(np.float64) * spacing_zyx
    if first.size == 0 or second.size == 0:
        return float("inf")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        tree = cKDTree(second)
        distance, _ = tree.query(first, k=1)
        return float(np.min(distance))
    minimum_squared = np.inf
    for start in range(0, len(first), 1024):
        chunk = first[start : start + 1024]
        squared = np.sum(
            (chunk[:, None, :] - second[None, :, :]) ** 2,
            axis=2,
        )
        minimum_squared = min(minimum_squared, float(np.min(squared)))
    return float(np.sqrt(minimum_squared))


def _connected_subset(
    subset: tuple[int, ...],
    adjacency: set[tuple[int, int]],
) -> bool:
    visited = {subset[0]}
    frontier = [subset[0]]
    subset_set = set(subset)
    while frontier:
        current = frontier.pop()
        for first, second in adjacency:
            if first == current:
                neighbour = second
            elif second == current:
                neighbour = first
            else:
                continue
            if neighbour in subset_set and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == subset_set


def controlled_event_manifest(
    full_fragment_instances: np.ndarray,
    cortical_instances: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    maximum_surface_distance_mm: float = 2.0,
    overlap_mask: Optional[np.ndarray] = None,
    maximum_multiway_events: Optional[int] = 100,
) -> ControlledEventManifest:
    """Build deterministic binary, connected 3--5 way, and intact events."""
    fragments = np.asarray(full_fragment_instances)
    cortical = np.asarray(cortical_instances)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if (
        fragments.ndim != 3
        or cortical.shape != fragments.shape
        or not np.issubdtype(fragments.dtype, np.integer)
        or not np.issubdtype(cortical.dtype, np.integer)
    ):
        raise ValueError(
            "full_fragment_instances and cortical_instances must be matching "
            "3D integer arrays"
        )
    if np.any(fragments < 0) or np.any(cortical < 0):
        raise ValueError("instance arrays must be non-negative")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(
        spacing <= 0
    ):
        raise ValueError("spacing_zyx must contain three finite positive values")
    if maximum_surface_distance_mm < 0:
        raise ValueError("maximum_surface_distance_mm must be non-negative")
    if maximum_multiway_events is not None and maximum_multiway_events < 0:
        raise ValueError("maximum_multiway_events must be non-negative or None")
    if overlap_mask is None:
        valid = np.ones(fragments.shape, dtype=bool)
    else:
        overlap = np.asarray(overlap_mask, dtype=bool)
        if overlap.shape != fragments.shape:
            raise ValueError("overlap_mask shape does not match instances")
        valid = ~overlap

    fragment_ids = np.unique(fragments[valid])
    fragment_ids = fragment_ids[fragment_ids > 0]
    cortical_ids = set(int(value) for value in np.unique(cortical[valid]) if value > 0)
    eligible = tuple(
        int(value) for value in fragment_ids if int(value) in cortical_ids
    )
    missing = tuple(
        int(value) for value in fragment_ids if int(value) not in cortical_ids
    )
    surfaces = {
        fragment_id: np.argwhere(
            _surface_mask((fragments == fragment_id) & valid)
        )
        for fragment_id in eligible
    }
    distances: dict[tuple[int, int], float] = {}
    for first, second in itertools.combinations(eligible, 2):
        distance = _minimum_physical_distance(
            surfaces[first],
            surfaces[second],
            spacing,
        )
        if distance <= maximum_surface_distance_mm + 1e-9:
            distances[(first, second)] = distance

    events: list[ControlledEvent] = []
    for first, second in sorted(distances):
        events.append(
            ControlledEvent(
                event_id=f"binary_{first}_{second}",
                kind="binary",
                child_ids=(first, second),
                adjacency_edges=((first, second, distances[(first, second)]),),
            )
        )
    adjacency = set(distances)
    multiway_count = 0
    for size in range(3, 6):
        for subset in itertools.combinations(eligible, size):
            if (
                maximum_multiway_events is not None
                and multiway_count >= maximum_multiway_events
            ):
                break
            if not _connected_subset(subset, adjacency):
                continue
            subset_edges = tuple(
                (first, second, distance)
                for (first, second), distance in sorted(distances.items())
                if first in subset and second in subset
            )
            events.append(
                ControlledEvent(
                    event_id="multiway_" + "_".join(str(value) for value in subset),
                    kind="multiway",
                    child_ids=subset,
                    adjacency_edges=subset_edges,
                )
            )
            multiway_count += 1
            if (
                maximum_multiway_events is not None
                and multiway_count >= maximum_multiway_events
            ):
                break
        if (
            maximum_multiway_events is not None
            and multiway_count >= maximum_multiway_events
        ):
            break
    for fragment_id in eligible:
        events.append(
            ControlledEvent(
                event_id=f"intact_{fragment_id}",
                kind="intact",
                child_ids=(fragment_id,),
                adjacency_edges=(),
            )
        )
    return ControlledEventManifest(
        spacing_zyx_mm=tuple(float(value) for value in spacing),
        maximum_surface_distance_mm=float(maximum_surface_distance_mm),
        eligible_fragment_ids=eligible,
        cortex_missing_fragment_ids=missing,
        events=tuple(events),
    )


def _event_arrays(
    event: ControlledEvent,
    full_fragment_instances: np.ndarray,
    cortical_instances: np.ndarray,
    overlap_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fragments = np.asarray(full_fragment_instances)
    cortical = np.asarray(cortical_instances)
    if fragments.shape != cortical.shape or fragments.ndim != 3:
        raise ValueError("fragment and cortical arrays must share one 3D shape")
    valid = np.ones(fragments.shape, dtype=bool)
    if overlap_mask is not None:
        overlap = np.asarray(overlap_mask, dtype=bool)
        if overlap.shape != fragments.shape:
            raise ValueError("overlap_mask shape does not match instances")
        valid &= ~overlap
    support = np.isin(fragments, event.child_ids) & valid
    ground_truth = np.zeros(fragments.shape, dtype=np.uint32)
    cortical_local = np.zeros(fragments.shape, dtype=np.uint32)
    for local_id, source_id in enumerate(event.child_ids, start=1):
        ground_truth[(fragments == source_id) & valid] = local_id
        cortical_local[(cortical == source_id) & support] = local_id
    if any(not np.any(cortical_local == local_id) for local_id in range(1, len(event.child_ids) + 1)):
        raise ValueError("every oracle event child must contain valid cortex")
    return support, ground_truth, cortical_local


def run_o1_event(
    event: ControlledEvent,
    full_fragment_instances: np.ndarray,
    cortical_instances: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    overlap_mask: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
) -> OracleEventResult:
    """O1: propagate perfect cortical identities into a controlled union."""
    support, ground_truth, cortical_local = _event_arrays(
        event,
        full_fragment_instances,
        cortical_instances,
        overlap_mask,
    )
    propagation = propagate_uniform(support, cortical_local, spacing_zyx)
    evaluation = evaluate_instance_segmentation(
        propagation.labels,
        ground_truth,
        prediction_kind="instance",
        ground_truth_kind="instance",
        valid_mask=support,
        iou_threshold=iou_threshold,
    )
    cortical_evaluation = evaluate_instance_segmentation(
        cortical_local,
        cortical_local,
        prediction_kind="instance",
        ground_truth_kind="instance",
        valid_mask=cortical_local > 0,
        iou_threshold=iou_threshold,
    )
    return OracleEventResult(
        mode="O1",
        event=event,
        prediction=propagation.labels,
        ground_truth=ground_truth,
        evaluation=evaluation,
        cortical_grouping_evaluation=cortical_evaluation,
        decisions=(),
    )


def run_o2_event(
    event: ControlledEvent,
    full_fragment_instances: np.ndarray,
    cortical_instances: np.ndarray,
    affinity_offsets_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    overlap_mask: Optional[np.ndarray] = None,
    config: Optional[ContinuityConfig] = None,
    iou_threshold: float = 0.5,
) -> OracleEventResult:
    """O2: infer unknown K from GT semantic cortex and oracle affinities."""
    support, ground_truth, cortical_local = _event_arrays(
        event,
        full_fragment_instances,
        cortical_instances,
        overlap_mask,
    )
    provisional = support.astype(np.uint32)
    cortex_probability = oracle_cortical_semantic(cortical_local)
    affinity_logits = oracle_affinity_logits(
        cortical_local,
        affinity_offsets_zyx,
    )
    result = refine(
        provisional,
        cortex_probability,
        affinity_logits,
        affinity_offsets_zyx,
        spacing_zyx,
        affinity_kind="logits",
        config=config,
    )
    evaluation = evaluate_instance_segmentation(
        result.full_instances,
        ground_truth,
        prediction_kind="instance",
        ground_truth_kind="instance",
        valid_mask=support,
        iou_threshold=iou_threshold,
    )
    cortical_evaluation = evaluate_instance_segmentation(
        result.raw_cortical_clusters,
        cortical_local,
        prediction_kind="instance",
        ground_truth_kind="instance",
        valid_mask=cortical_local > 0,
        iou_threshold=iou_threshold,
    )
    return OracleEventResult(
        mode="O2",
        event=event,
        prediction=result.full_instances,
        ground_truth=ground_truth,
        evaluation=evaluation,
        cortical_grouping_evaluation=cortical_evaluation,
        decisions=result.decisions,
    )
