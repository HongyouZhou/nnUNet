from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import ContinuityConfig
from .graph import (
    GraphResourceLimit,
    build_signed_graph,
    mutex_watershed,
    node_labels_to_volume,
    repulsive_incidence,
)
from .io import ContinuityInputs, ContinuityResult, InstanceDecision
from .propagation import propagate_uniform
from .scoring import (
    ConservativeSplitScorer,
    SplitFeatures,
    SplitScorer,
)
from .support import hysteresis_cortex_support


def _instance_bboxes(
    provisional_instances: np.ndarray,
) -> dict[int, tuple[slice, slice, slice]]:
    """Find exact nonzero-label boxes, with a NumPy fallback for minimal envs."""
    instance_ids = np.unique(provisional_instances)
    instance_ids = instance_ids[instance_ids > 0]
    try:
        from scipy.ndimage import find_objects
    except ImportError:
        find_objects = None

    boxes: dict[int, tuple[slice, slice, slice]] = {}
    maximum_id = int(instance_ids[-1]) if instance_ids.size else 0
    # scipy.find_objects returns a list indexed up to max(label); avoid that
    # representation for sparse externally supplied IDs.
    use_find_objects = (
        find_objects is not None
        and maximum_id <= max(10_000, 4 * int(instance_ids.size))
    )
    if use_find_objects:
        objects = find_objects(provisional_instances)
        for raw_instance_id in instance_ids:
            instance_id = int(raw_instance_id)
            if instance_id <= len(objects) and objects[instance_id - 1] is not None:
                boxes[instance_id] = objects[instance_id - 1]
        if len(boxes) == len(instance_ids):
            return boxes

    for raw_instance_id in instance_ids:
        instance_id = int(raw_instance_id)
        if instance_id in boxes:
            continue
        coordinates = np.argwhere(provisional_instances == instance_id)
        if coordinates.size == 0:
            continue
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        boxes[instance_id] = tuple(
            slice(int(start), int(stop)) for start, stop in zip(lower, upper)
        )
    return boxes


def _cluster_min_flat(seed_labels: np.ndarray, cluster_id: int) -> int:
    locations = np.flatnonzero(seed_labels.ravel() == cluster_id)
    if locations.size == 0:
        raise RuntimeError(f"cluster {cluster_id} has no cortical seed")
    return int(locations.min())


def _stable_global_mapping(
    local_partition: np.ndarray,
    seed_labels: np.ndarray,
    original_id: int,
    next_id: int,
) -> tuple[dict[int, int], int]:
    local_ids = np.unique(local_partition)
    local_ids = local_ids[local_ids > 0]
    sizes = {
        int(label): int(np.count_nonzero(local_partition == label))
        for label in local_ids
    }
    min_flat = {
        int(label): _cluster_min_flat(seed_labels, int(label))
        for label in local_ids
    }
    keeper = min(
        (int(label) for label in local_ids),
        key=lambda label: (-sizes[label], min_flat[label]),
    )
    mapping = {keeper: int(original_id)}
    for label in sorted(
        (int(value) for value in local_ids if int(value) != keeper),
        key=lambda value: min_flat[value],
    ):
        mapping[label] = next_id
        next_id += 1
    return mapping, next_id


def refine_instances(
    inputs: ContinuityInputs,
    config: Optional[ContinuityConfig] = None,
    scorer: Optional[SplitScorer] = None,
) -> ContinuityResult:
    """Refine each provisional instance independently from cortical identity."""
    data = inputs.validated()
    cfg = ContinuityConfig() if config is None else config
    split_scorer: SplitScorer = (
        ConservativeSplitScorer.from_config(cfg) if scorer is None else scorer
    )

    provisional = data.provisional_instances
    full_output = provisional.astype(np.uint32, copy=True)
    cortical_output = np.zeros(provisional.shape, dtype=np.uint32)
    raw_clusters_output = np.zeros(provisional.shape, dtype=np.uint32)
    next_id = int(provisional.max(initial=0)) + 1
    next_raw_id = 1
    decisions: list[InstanceDecision] = []
    voxel_volume_mm3 = float(np.prod(data.spacing_zyx))

    instance_boxes = _instance_bboxes(provisional)
    for instance_id in sorted(instance_boxes):
        crop = instance_boxes[instance_id]
        local_provisional = provisional[crop]
        instance_mask = local_provisional == instance_id
        local_cortex_probability = data.cortex_probability[crop]
        local_affinity = data.affinity[(slice(None),) + crop]
        local_rim_probability = (
            None if data.rim_probability is None else data.rim_probability[crop]
        )
        local_full_output = full_output[crop]
        local_cortical_output = cortical_output[crop]
        local_raw_clusters_output = raw_clusters_output[crop]
        cortex_support = hysteresis_cortex_support(
            local_cortex_probability,
            instance_mask,
            cfg.cortex_low_threshold,
            cfg.cortex_high_threshold,
        )
        cortex_voxels = int(cortex_support.sum())
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

        try:
            graph = build_signed_graph(
                cortex_support,
                local_affinity,
                data.affinity_offsets_zyx,
                cfg,
                affinity_kind=data.affinity_kind,
                node_confidence=local_cortex_probability,
                max_nodes=cfg.max_graph_nodes,
                max_edges=cfg.max_graph_edges,
            )
        except GraphResourceLimit as error:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="abstain_resource_limit",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=0,
                    output_ids=(instance_id,),
                    diagnostic_detail=str(error),
                )
            )
            continue
        grouping = mutex_watershed(graph)
        local_cortex_clusters = node_labels_to_volume(
            graph,
            grouping.node_labels,
            dtype=np.int32,
        )
        if grouping.cluster_count > cfg.max_clusters_per_instance:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="too_many_cortical_clusters",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=grouping.cluster_count,
                    output_ids=(instance_id,),
                    normalized_energy_gain=grouping.normalized_energy_gain,
                    graph_edges=len(graph.edges),
                )
            )
            continue
        for cluster_id in range(1, grouping.cluster_count + 1):
            local_raw_clusters_output[
                local_cortex_clusters == cluster_id
            ] = next_raw_id
            next_raw_id += 1

        if grouping.cluster_count <= 1:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="unchanged",
                    reason="single_cortical_cluster",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=grouping.cluster_count,
                    output_ids=(instance_id,),
                    normalized_energy_gain=grouping.normalized_energy_gain,
                    graph_edges=len(graph.edges),
                )
            )
            continue

        high_confidence = (
            cortex_support
            & (local_cortex_probability >= cfg.cortex_high_threshold)
        )
        high_counts = np.asarray(
            [
                int(
                    np.count_nonzero(
                        high_confidence & (local_cortex_clusters == cluster_id)
                    )
                )
                for cluster_id in range(1, grouping.cluster_count + 1)
            ],
            dtype=np.int64,
        )
        # Each MWS cluster needs at least one actual high-threshold seed for
        # algorithmic validity. Quantitative acceptance uses physical volume.
        high_confidence_volumes = high_counts * voxel_volume_mm3
        if np.any(high_counts < 1) or np.any(
            high_confidence_volumes < cfg.min_high_confidence_volume_mm3
        ):
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="weak_cortical_cluster",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=grouping.cluster_count,
                    output_ids=(instance_id,),
                    normalized_energy_gain=grouping.normalized_energy_gain,
                    graph_edges=len(graph.edges),
                )
            )
            continue

        propagation = propagate_uniform(
            instance_mask,
            local_cortex_clusters,
            data.spacing_zyx,
            tie_tolerance=cfg.distance_tie_tolerance,
        )
        piece_voxels = np.asarray(
            [
                int(np.count_nonzero(propagation.labels == cluster_id))
                for cluster_id in range(1, grouping.cluster_count + 1)
            ],
            dtype=np.int64,
        )
        piece_volumes = piece_voxels * voxel_volume_mm3
        if np.any(piece_volumes < cfg.min_piece_volume_mm3):
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason="piece_below_minimum_physical_volume",
                    cortical_voxels=cortex_voxels,
                    raw_clusters=grouping.cluster_count,
                    output_ids=(instance_id,),
                    normalized_energy_gain=grouping.normalized_energy_gain,
                    graph_edges=len(graph.edges),
                    unseeded_volume_components=propagation.unseeded_components,
                )
            )
            continue

        incidence = repulsive_incidence(
            graph,
            grouping.node_labels,
            grouping.cluster_count,
        )[1:]
        rim_support = (
            0.0
            if local_rim_probability is None
            else float(np.mean(local_rim_probability[cortex_support]))
        )
        features = SplitFeatures(
            normalized_energy_gain=grouping.normalized_energy_gain,
            cluster_count=grouping.cluster_count,
            min_high_confidence_volume_mm3=float(
                high_confidence_volumes.min()
            ),
            min_repulsive_edges=int(incidence.min()),
            min_piece_fraction=float(piece_voxels.min() / instance_mask.sum()),
            rim_support=rim_support,
            unseeded_volume_components=propagation.unseeded_components,
        )
        score = split_scorer.evaluate(features)
        if not score.accepted:
            local_cortical_output[cortex_support] = instance_id
            decisions.append(
                InstanceDecision(
                    instance_id=instance_id,
                    status="abstained",
                    reason=score.reason,
                    cortical_voxels=cortex_voxels,
                    raw_clusters=grouping.cluster_count,
                    output_ids=(instance_id,),
                    normalized_energy_gain=grouping.normalized_energy_gain,
                    graph_edges=len(graph.edges),
                    unseeded_volume_components=propagation.unseeded_components,
                    split_features=features.to_dict(),
                    split_probability=score.probability,
                )
            )
            continue

        mapping, next_id = _stable_global_mapping(
            propagation.labels,
            local_cortex_clusters,
            instance_id,
            next_id,
        )
        for local_id, global_id in mapping.items():
            local_full_output[propagation.labels == local_id] = global_id
            local_cortical_output[local_cortex_clusters == local_id] = global_id
        decisions.append(
            InstanceDecision(
                instance_id=instance_id,
                status="accepted",
                reason=score.reason,
                cortical_voxels=cortex_voxels,
                raw_clusters=grouping.cluster_count,
                output_ids=tuple(sorted(mapping.values())),
                normalized_energy_gain=grouping.normalized_energy_gain,
                graph_edges=len(graph.edges),
                unseeded_volume_components=propagation.unseeded_components,
                split_features=features.to_dict(),
                split_probability=score.probability,
            )
        )

    result = ContinuityResult(
        full_instances=full_output,
        cortical_instances=cortical_output,
        raw_cortical_clusters=raw_clusters_output,
        decisions=tuple(decisions),
    )
    result.validate_against(provisional)
    return result


def refine(
    provisional_instances: np.ndarray,
    cortex_probability: np.ndarray,
    affinity: np.ndarray,
    affinity_offsets_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    rim_probability: Optional[np.ndarray] = None,
    volume_probability: Optional[np.ndarray] = None,
    affinity_kind: str = "logits",
    config: Optional[ContinuityConfig] = None,
    scorer: Optional[SplitScorer] = None,
) -> ContinuityResult:
    """Array-level end-to-end API intended for CLI and predictor integration."""
    return refine_instances(
        ContinuityInputs(
            provisional_instances=provisional_instances,
            cortex_probability=cortex_probability,
            affinity=affinity,
            affinity_offsets_zyx=affinity_offsets_zyx,
            spacing_zyx=np.asarray(spacing_zyx, dtype=np.float64),
            affinity_kind=affinity_kind,  # type: ignore[arg-type]
            rim_probability=rim_probability,
            volume_probability=volume_probability,
        ),
        config=config,
        scorer=scorer,
    )
