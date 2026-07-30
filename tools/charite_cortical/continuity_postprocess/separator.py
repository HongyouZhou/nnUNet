from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from .config import ContinuityConfig
from .io import ContinuityResult, InstanceDecision
from .pipeline import _instance_bboxes, _stable_global_mapping
from .propagation import propagate_uniform
from .support import hysteresis_cortex_support


def _connected_components_6(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import label
    except ImportError:
        label = None
    if label is not None:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
        labels, count = label(binary, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)

    output = np.zeros(binary.shape, dtype=np.int32)
    remaining = binary.copy()
    component_id = 0
    shape = binary.shape
    while np.any(remaining):
        component_id += 1
        start = tuple(int(value) for value in np.argwhere(remaining)[0])
        remaining[start] = False
        output[start] = component_id
        queue = deque([start])
        while queue:
            point = queue.popleft()
            for axis in range(3):
                for delta in (-1, 1):
                    neighbour = list(point)
                    neighbour[axis] += delta
                    if neighbour[axis] < 0 or neighbour[axis] >= shape[axis]:
                        continue
                    neighbour_tuple = tuple(neighbour)
                    if remaining[neighbour_tuple]:
                        remaining[neighbour_tuple] = False
                        output[neighbour_tuple] = component_id
                        queue.append(neighbour_tuple)
    return output, component_id


def _separator_incidence(
    component_labels: np.ndarray,
    separator_mask: np.ndarray,
    component_count: int,
) -> np.ndarray:
    incidence = np.zeros(component_count + 1, dtype=np.int64)
    shape = component_labels.shape
    for point_array in np.argwhere(separator_mask):
        point = tuple(int(value) for value in point_array)
        neighbours: set[int] = set()
        for axis in range(3):
            for delta in (-1, 1):
                neighbour = list(point)
                neighbour[axis] += delta
                if neighbour[axis] < 0 or neighbour[axis] >= shape[axis]:
                    continue
                label_id = int(component_labels[tuple(neighbour)])
                if label_id > 0:
                    neighbours.add(label_id)
        for label_id in neighbours:
            incidence[label_id] += 1
    return incidence


def _validate_softmax(
    provisional_instances: np.ndarray,
    three_class_softmax: np.ndarray,
    spacing_zyx: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    provisional = np.asarray(provisional_instances)
    probability = np.asarray(three_class_softmax, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if provisional.ndim != 3 or not np.issubdtype(provisional.dtype, np.integer):
        raise ValueError("provisional_instances must be a 3D integer array")
    if np.any(provisional < 0):
        raise ValueError("provisional_instances must be non-negative")
    if probability.shape != (3,) + provisional.shape:
        raise ValueError(
            "three_class_softmax must have shape "
            "(3, *provisional_instances.shape)"
        )
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("three_class_softmax must contain finite probabilities")
    if not np.allclose(probability.sum(axis=0), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("three_class_softmax channels must sum to one")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(
        spacing <= 0
    ):
        raise ValueError("spacing_zyx must contain three finite positive values")
    return provisional, probability, spacing


def refine_with_separator(
    provisional_instances: np.ndarray,
    three_class_softmax: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    config: Optional[ContinuityConfig] = None,
) -> ContinuityResult:
    """MVP0 semantic-separator baseline with automatic unknown K.

    Channel semantics are ``background, cortex, separator``. Cortical support
    is ``p(cortex) + p(separator)``. High-confidence separator voxels are
    removed only while generating six-connected identity seeds; the complete
    provisional volume, including the separator band, is passively assigned
    by the same uniform propagation used by the affinity method.
    """
    provisional, softmax, spacing = _validate_softmax(
        provisional_instances,
        three_class_softmax,
        spacing_zyx,
    )
    cfg = ContinuityConfig() if config is None else config
    full_output = provisional.astype(np.uint32, copy=True)
    cortical_output = np.zeros(provisional.shape, dtype=np.uint32)
    raw_output = np.zeros(provisional.shape, dtype=np.uint32)
    next_id = int(np.max(provisional)) + 1 if provisional.size else 1
    next_raw_id = 1
    voxel_volume_mm3 = float(np.prod(spacing))
    decisions: list[InstanceDecision] = []

    for instance_id, crop in sorted(_instance_bboxes(provisional).items()):
        local_provisional = provisional[crop]
        instance_mask = local_provisional == instance_id
        local_softmax = softmax[(slice(None),) + crop]
        cortex_probability = local_softmax[1] + local_softmax[2]
        cortex_support = hysteresis_cortex_support(
            cortex_probability,
            instance_mask,
            cfg.cortex_low_threshold,
            cfg.cortex_high_threshold,
        )
        cortex_voxels = int(cortex_support.sum())
        local_cortical_output = cortical_output[crop]
        local_raw_output = raw_output[crop]
        local_full_output = full_output[crop]
        if cortex_voxels == 0:
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="unchanged",
                    reason="no_reliable_cortex",
                    cortical_voxels=0,
                    raw_clusters=0,
                    output_ids=(instance_id,),
                )
            )
            continue
        if cfg.max_graph_nodes is not None and cortex_voxels > cfg.max_graph_nodes:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="abstain_resource_limit",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=0,
                    output_ids=(instance_id,),
                    diagnostic_detail=(
                        f"separator cortex node count {cortex_voxels} exceeds "
                        f"max_graph_nodes={cfg.max_graph_nodes}"
                    ),
                )
            )
            continue

        separator = (
            cortex_support
            & (local_softmax[2] >= cfg.separator_probability_threshold)
        )
        identity_support = cortex_support & ~separator
        component_labels, component_count = _connected_components_6(identity_support)
        if component_count > cfg.max_clusters_per_instance:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="too_many_cortical_clusters",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=component_count,
                    output_ids=(instance_id,),
                )
            )
            continue
        for component_id in range(1, component_count + 1):
            local_raw_output[component_labels == component_id] = next_raw_id
            next_raw_id += 1
        if component_count <= 1:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="unchanged",
                    reason=(
                        "single_cortical_cluster"
                        if component_count == 1
                        else "no_separator_free_cortex"
                    ),
                    cortical_voxels=cortex_voxels,
                    raw_clusters=component_count,
                    output_ids=(instance_id,),
                )
            )
            continue

        high_confidence = cortex_probability >= cfg.cortex_high_threshold
        high_volumes = np.asarray(
            [
                np.count_nonzero(
                    (component_labels == component_id) & high_confidence
                )
                * voxel_volume_mm3
                for component_id in range(1, component_count + 1)
            ],
            dtype=np.float64,
        )
        if np.any(high_volumes <= 0) or np.any(
            high_volumes < cfg.min_high_confidence_volume_mm3
        ):
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="weak_cortical_cluster",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=component_count,
                    output_ids=(instance_id,),
                )
            )
            continue
        incidence = _separator_incidence(
            component_labels,
            separator,
            component_count,
        )[1:]
        if np.any(incidence < 1):
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="separator_not_incident_to_all_clusters",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=component_count,
                    output_ids=(instance_id,),
                )
            )
            continue

        propagation = propagate_uniform(
            instance_mask,
            component_labels,
            spacing,
            tie_tolerance=cfg.distance_tie_tolerance,
        )
        piece_voxels = np.asarray(
            [
                np.count_nonzero(propagation.labels == component_id)
                for component_id in range(1, component_count + 1)
            ],
            dtype=np.int64,
        )
        if np.any(piece_voxels * voxel_volume_mm3 < cfg.min_piece_volume_mm3):
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="piece_below_minimum_physical_volume",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=component_count,
                    output_ids=(instance_id,),
                    unseeded_volume_components=propagation.unseeded_components,
                )
            )
            continue

        mapping, next_id = _stable_global_mapping(
            propagation.labels,
            component_labels,
            instance_id,
            next_id,
        )
        for component_id, global_id in mapping.items():
            local_full_output[propagation.labels == component_id] = global_id
            local_cortical_output[
                cortex_support & (propagation.labels == component_id)
            ] = global_id
        decisions.append(
            InstanceDecision(
                instance_id=instance_id,
                status="accepted",
                reason="separator_components_accepted",
                cortical_voxels=cortex_voxels,
                raw_clusters=component_count,
                output_ids=tuple(sorted(mapping.values())),
                unseeded_volume_components=propagation.unseeded_components,
                diagnostic_detail=(
                    f"separator_voxels={int(separator.sum())};"
                    f"min_separator_incidence={int(incidence.min())}"
                ),
            )
        )

    result = ContinuityResult(
        full_instances=full_output,
        cortical_instances=cortical_output,
        raw_cortical_clusters=raw_output,
        decisions=tuple(decisions),
    )
    result.validate_against(provisional)
    return result
