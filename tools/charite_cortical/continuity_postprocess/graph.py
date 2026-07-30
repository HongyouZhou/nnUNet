from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np

from .config import ContinuityConfig


class GraphResourceLimit(RuntimeError):
    """Raised before allocating an edge list that exceeds the frozen limit."""


@dataclasses.dataclass(frozen=True, order=True)
class SignedEdge:
    """One undirected signed cortical relation.

    ``repulsive=False`` means the endpoints should share an instance.
    ``repulsive=True`` means they must remain in different instances when the
    mutex edge is processed before a competing attractive path.
    """

    u: int
    v: int
    weight: float
    repulsive: bool


@dataclasses.dataclass(frozen=True)
class SignedGraph:
    node_flat_indices: np.ndarray
    edges: tuple[SignedEdge, ...]
    volume_shape: tuple[int, int, int]

    @property
    def node_count(self) -> int:
        return int(self.node_flat_indices.size)


@dataclasses.dataclass(frozen=True)
class MutexWatershedResult:
    node_labels: np.ndarray
    cluster_count: int
    unsplit_energy: float
    partition_energy: float
    normalized_energy_gain: float
    processed_attractive_edges: int
    processed_repulsive_edges: int
    skipped_mutex_attractive_edges: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def _paired_slices(
    shape: tuple[int, int, int],
    offset: np.ndarray,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
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
    return tuple(source), tuple(target)  # type: ignore[return-value]


def build_signed_graph(
    cortex_support: np.ndarray,
    affinity: np.ndarray,
    affinity_offsets_zyx: np.ndarray,
    config: ContinuityConfig,
    *,
    affinity_kind: str = "logits",
    node_confidence: np.ndarray | None = None,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> SignedGraph:
    """Convert dense directional affinities into a canonical signed edge list."""
    support = np.asarray(cortex_support, dtype=bool)
    affinities = np.asarray(affinity)
    offsets = np.asarray(affinity_offsets_zyx)
    if support.ndim != 3:
        raise ValueError("cortex_support must be 3D")
    if affinities.ndim != 4 or affinities.shape[1:] != support.shape:
        raise ValueError("affinity must have shape (E, *cortex_support.shape)")
    if offsets.shape != (affinities.shape[0], 3):
        raise ValueError("affinity_offsets_zyx must have shape (E, 3)")
    if affinity_kind not in ("logits", "probabilities"):
        raise ValueError("affinity_kind must be 'logits' or 'probabilities'")
    if not np.all(np.isfinite(affinities)):
        raise ValueError("affinity contains NaN or infinite values")
    if node_confidence is None:
        confidence = np.ones(support.shape, dtype=np.float32)
    else:
        confidence = np.asarray(node_confidence, dtype=np.float32)
        if confidence.shape != support.shape:
            raise ValueError("node_confidence must match cortex_support shape")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("node_confidence contains NaN or infinite values")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("node_confidence must be in [0, 1]")

    node_count = int(np.count_nonzero(support))
    if max_nodes is not None and node_count > max_nodes:
        raise GraphResourceLimit(
            f"graph node count {node_count} exceeds max_graph_nodes={max_nodes}"
        )

    # Preflight edge count before constructing node IDs or Python edge objects.
    edge_count = 0
    for edge_index, offset in enumerate(offsets):
        source_slice, target_slice = _paired_slices(support.shape, offset)
        valid = support[source_slice] & support[target_slice]
        if not np.any(valid):
            continue
        values = affinities[(edge_index,) + source_slice][valid]
        probabilities = _sigmoid(values) if affinity_kind == "logits" else values
        edge_count += int(
            np.count_nonzero(
                (probabilities >= config.attractive_probability_threshold)
                | (probabilities <= config.repulsive_probability_threshold)
            )
        )
        if max_edges is not None and edge_count > max_edges:
            raise GraphResourceLimit(
                f"graph edge count {edge_count} exceeds max_graph_edges={max_edges}"
            )

    flat_indices = np.flatnonzero(support.ravel()).astype(np.int64, copy=False)
    node_map = np.full(support.size, -1, dtype=np.int64)
    node_map[flat_indices] = np.arange(flat_indices.size, dtype=np.int64)
    node_map = node_map.reshape(support.shape)

    edges: list[SignedEdge] = []
    for edge_index, offset in enumerate(offsets):
        source_slice, target_slice = _paired_slices(support.shape, offset)
        source_support = support[source_slice]
        target_support = support[target_slice]
        valid = source_support & target_support
        if not np.any(valid):
            continue
        values = affinities[(edge_index,) + source_slice][valid]
        probabilities = _sigmoid(values) if affinity_kind == "logits" else values
        source_nodes = node_map[source_slice][valid]
        target_nodes = node_map[target_slice][valid]
        confidence_product = (
            confidence[source_slice][valid] * confidence[target_slice][valid]
        )
        if np.any(source_nodes < 0) or np.any(target_nodes < 0):
            raise RuntimeError("internal node-map construction failure")

        for source_node, target_node, probability, endpoint_confidence in zip(
            source_nodes,
            target_nodes,
            probabilities,
            confidence_product,
        ):
            u, v = sorted((int(source_node), int(target_node)))
            if u == v:
                continue
            probability = float(probability)
            if probability >= config.attractive_probability_threshold:
                edges.append(
                    SignedEdge(
                        u=u,
                        v=v,
                        weight=(probability - 0.5) * float(endpoint_confidence),
                        repulsive=False,
                    )
                )
            elif probability <= config.repulsive_probability_threshold:
                edges.append(
                    SignedEdge(
                        u=u,
                        v=v,
                        weight=(0.5 - probability) * float(endpoint_confidence),
                        repulsive=True,
                    )
                )

    # Canonicalization makes channel ordering and input traversal irrelevant.
    # Repulsive edges win exact-confidence ties, preventing an irreversible
    # attractive merge before the corresponding mutex is registered.
    edges.sort(
        key=lambda edge: (
            -edge.weight,
            0 if edge.repulsive else 1,
            edge.u,
            edge.v,
        )
    )
    return SignedGraph(
        node_flat_indices=flat_indices,
        edges=tuple(edges),
        volume_shape=tuple(int(value) for value in support.shape),
    )


class _MutexUnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.mutex: list[set[int]] = [set() for _ in range(size)]

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        while parent != int(self.parent[parent]):
            parent = int(self.parent[parent])
        while item != parent:
            following = int(self.parent[item])
            self.parent[item] = parent
            item = following
        return parent

    def _normalized_mutex(self, root: int) -> set[int]:
        root = self.find(root)
        normalized = {
            self.find(neighbour)
            for neighbour in self.mutex[root]
            if self.find(neighbour) != root
        }
        self.mutex[root] = normalized
        return normalized

    def are_mutex(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        return (
            second_root in self._normalized_mutex(first_root)
            or first_root in self._normalized_mutex(second_root)
        )

    def add_mutex(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        self.mutex[first_root].add(second_root)
        self.mutex[second_root].add(first_root)

    def union(self, first: int, second: int) -> int:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        if self.are_mutex(first_root, second_root):
            raise ValueError("cannot union components carrying a mutex relation")

        # Always preserve the smaller canonical root. This avoids rank/history
        # dependent output under node or edge permutations.
        new_root, old_root = sorted((first_root, second_root))
        neighbours = (
            self._normalized_mutex(new_root)
            | self._normalized_mutex(old_root)
        )
        neighbours.discard(new_root)
        neighbours.discard(old_root)
        self.parent[old_root] = new_root
        self.mutex[old_root].clear()

        normalized_neighbours = {self.find(value) for value in neighbours}
        normalized_neighbours.discard(new_root)
        self.mutex[new_root] = normalized_neighbours
        for neighbour in tuple(normalized_neighbours):
            neighbour_root = self.find(neighbour)
            updated = {
                self.find(value)
                for value in self.mutex[neighbour_root]
                if self.find(value) != neighbour_root
            }
            updated.discard(old_root)
            updated.add(new_root)
            self.mutex[neighbour_root] = updated
        return new_root


def _partition_energy(
    edges: Iterable[SignedEdge],
    node_labels: np.ndarray,
) -> tuple[float, float]:
    unsplit = 0.0
    partition = 0.0
    for edge in edges:
        separated = int(node_labels[edge.u]) != int(node_labels[edge.v])
        if edge.repulsive:
            unsplit += edge.weight
            if not separated:
                partition += edge.weight
        elif separated:
            partition += edge.weight
    return float(unsplit), float(partition)


def mutex_watershed(graph: SignedGraph) -> MutexWatershedResult:
    """Deterministic signed Mutex Watershed with unknown output K."""
    if graph.node_count == 0:
        return MutexWatershedResult(
            node_labels=np.zeros(0, dtype=np.int32),
            cluster_count=0,
            unsplit_energy=0.0,
            partition_energy=0.0,
            normalized_energy_gain=0.0,
            processed_attractive_edges=0,
            processed_repulsive_edges=0,
            skipped_mutex_attractive_edges=0,
        )

    union_find = _MutexUnionFind(graph.node_count)
    attractive_count = 0
    repulsive_count = 0
    skipped_mutex = 0
    ordered_edges = sorted(
        graph.edges,
        key=lambda edge: (
            -edge.weight,
            0 if edge.repulsive else 1,
            min(edge.u, edge.v),
            max(edge.u, edge.v),
        ),
    )
    for edge in ordered_edges:
        first_root = union_find.find(edge.u)
        second_root = union_find.find(edge.v)
        if first_root == second_root:
            continue
        if edge.repulsive:
            union_find.add_mutex(first_root, second_root)
            repulsive_count += 1
        else:
            attractive_count += 1
            if union_find.are_mutex(first_root, second_root):
                skipped_mutex += 1
                continue
            union_find.union(first_root, second_root)

    roots = np.asarray(
        [union_find.find(node) for node in range(graph.node_count)],
        dtype=np.int64,
    )
    unique_roots = np.unique(roots)
    # np.unique orders the canonical (minimum-node) roots.
    root_to_label = {
        int(root): label for label, root in enumerate(unique_roots, start=1)
    }
    node_labels = np.asarray(
        [root_to_label[int(root)] for root in roots],
        dtype=np.int32,
    )
    unsplit, partition = _partition_energy(graph.edges, node_labels)
    total_weight = float(sum(edge.weight for edge in graph.edges))
    gain = (unsplit - partition) / total_weight if total_weight > 0 else 0.0
    return MutexWatershedResult(
        node_labels=node_labels,
        cluster_count=int(unique_roots.size),
        unsplit_energy=unsplit,
        partition_energy=partition,
        normalized_energy_gain=float(gain),
        processed_attractive_edges=attractive_count,
        processed_repulsive_edges=repulsive_count,
        skipped_mutex_attractive_edges=skipped_mutex,
    )


def node_labels_to_volume(
    graph: SignedGraph,
    node_labels: np.ndarray,
    *,
    dtype: np.dtype | type = np.int32,
) -> np.ndarray:
    labels = np.asarray(node_labels)
    if labels.shape != (graph.node_count,):
        raise ValueError("node_labels shape does not match graph.node_count")
    output = np.zeros(int(np.prod(graph.volume_shape)), dtype=dtype)
    output[graph.node_flat_indices] = labels
    return output.reshape(graph.volume_shape)


def repulsive_incidence(
    graph: SignedGraph,
    node_labels: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    counts = np.zeros(cluster_count + 1, dtype=np.int64)
    for edge in graph.edges:
        if not edge.repulsive:
            continue
        first = int(node_labels[edge.u])
        second = int(node_labels[edge.v])
        if first != second:
            counts[first] += 1
            counts[second] += 1
    return counts
