"""Exploratory CT cortical-sheet ownership, independent of GT at inference.

The frozen separator instances supply a coarse partition. Inside each one,
physical cortical patches form a signed graph: abrupt changes of reliable CT
sheet direction supply repulsive edges. Constrained agglomeration enforces those
edges globally, even if a smooth residual bridge offers an alternative path.
This is a mechanism prototype, not a validated fracture detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.segmentation import watershed

from tools.charite_cortical.diagnose_continuity import CONNECTIVITY, separator_markers


@dataclass(frozen=True)
class GeometryConfig:
    patch_mm: float = 1.25
    derivative_mm: float = 0.5
    fine_tensor_mm: float = 0.9
    context_tensor_mm: float = 2.7
    minimum_coherence: float = 0.3
    minimum_angle_deg: float = 20.0
    minimum_excess_deg: float = 8.0
    minimum_repulsive_edges: int = 4
    separator_support_weight: float = 0.15

    def validate(self):
        scales = (self.patch_mm, self.derivative_mm, self.fine_tensor_mm, self.context_tensor_mm)
        if any(not np.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("Physical scales must be finite and positive")
        if self.context_tensor_mm <= self.fine_tensor_mm:
            raise ValueError("Context scale must exceed the fine scale")
        if not 0 <= self.minimum_coherence <= 1:
            raise ValueError("Coherence must lie in [0,1]")
        if not 0 <= self.separator_support_weight <= 0.25:
            raise ValueError("Separator evidence is restricted to a supporting role")
        if not 0 < self.minimum_angle_deg < 90 or not 0 <= self.minimum_excess_deg < 90:
            raise ValueError("Invalid angular thresholds")
        if self.minimum_repulsive_edges < 1:
            raise ValueError("Repulsive support must be positive")


def patch_graph(base, spacing, patch_mm):
    """Connected parts of physical grid cells; never alias nearby sheets.

    26-neighbour voxel edges preserve the frozen connectivity convention.
    A cell is subdivided if its voxels do not actually connect inside the cell.
    Different frozen ownership labels can never share a patch or graph edge.
    """
    positions = np.flatnonzero(base)
    coordinates = np.column_stack(np.unravel_index(positions, base.shape))
    count = len(positions)
    if count == 0:
        return positions, np.empty(0, np.int32), np.empty((0, 2), np.int32), np.empty(0), np.empty((0, 3))
    cells = np.floor(coordinates * spacing / patch_mm).astype(np.int32)
    cell_keys = np.column_stack((base.ravel()[positions], cells))
    _, cell_ids = np.unique(cell_keys, axis=0, return_inverse=True)
    lookup = np.full(base.size, -1, np.int32)
    lookup[positions] = np.arange(count, dtype=np.int32)
    lookup = lookup.reshape(base.shape)
    internal_a, internal_b, external_a, external_b, contact = [], [], [], [], []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        first = tuple(slice(max(0, -d), min(n, n - d)) for n, d in zip(base.shape, offset))
        second = tuple(slice(max(0, d), min(n, n + d)) for n, d in zip(base.shape, offset))
        a, b = lookup[first], lookup[second]
        valid = (a >= 0) & (b >= 0) & (base[first] == base[second])
        a, b = a[valid], b[valid]
        same = cell_ids[a] == cell_ids[b]
        internal_a.append(a[same]); internal_b.append(b[same])
        external_a.append(a[~same]); external_b.append(b[~same])
        # Direction-normalized contact evidence, in physical units.
        weight = np.prod(spacing) / np.linalg.norm(np.asarray(offset) * spacing)
        contact.append(np.full(np.count_nonzero(~same), weight, np.float32))
    a, b = np.concatenate(internal_a), np.concatenate(internal_b)
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(count, count)).tocsr()
    patch_count, voxel_patch = connected_components(graph, directed=False)
    a, b = voxel_patch[np.concatenate(external_a)], voxel_patch[np.concatenate(external_b)]
    pairs = np.sort(np.column_stack((a, b)), axis=1)
    edges, inverse = np.unique(pairs, axis=0, return_inverse=True)
    area = np.bincount(inverse, weights=np.concatenate(contact), minlength=len(edges))
    sizes = np.bincount(voxel_patch, minlength=patch_count)
    centers = np.column_stack([
        np.bincount(voxel_patch, weights=coordinates[:, axis] * spacing[axis], minlength=patch_count) / sizes
        for axis in range(3)
    ])
    return positions, voxel_patch.astype(np.int32), edges.astype(np.int32), area, centers


def _normals_from_covariance(covariance):
    tensor = np.zeros((len(covariance), 3, 3), np.float32)
    for index, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        tensor[:, a, b] = covariance[:, index]
        tensor[:, b, a] = covariance[:, index]
    values, vectors = np.linalg.eigh(tensor)
    # A sheet has one dominant gradient direction, unlike a corner or junction.
    coherence = (values[:, 2] - values[:, 1]) / np.maximum(values[:, 2], 1e-12)
    coherence[values[:, 2] <= 1e-8] = 0
    return vectors[:, :, 2], np.clip(coherence, 0, 1)


def ct_sheet_features(ct, spacing, positions, voxel_patch, config):
    """Sign-invariant CT gradient tensors; both sides of a shell agree.

    Gaussian derivatives and tensor averaging use millimetres, including on
    anisotropic native grids. Patch averages integrate gradients through the
    cortical thickness. Absolute HU never imposes a must-link relation.
    """
    sigma = config.derivative_mm / spacing
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]; order[axis] = 1
        gradients.append(gaussian_filter(ct, sigma, order=order, mode="nearest") / spacing[axis])
    count = int(voxel_patch.max()) + 1
    sizes = np.bincount(voxel_patch, minlength=count)
    covariances = [np.empty((count, 6), np.float32), np.empty((count, 6), np.float32)]
    for index, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        product = gradients[a] * gradients[b]
        for scale_index, scale in enumerate((config.fine_tensor_mm, config.context_tensor_mm)):
            field = gaussian_filter(product, scale / spacing, mode="nearest")
            covariances[scale_index][:, index] = np.bincount(
                voxel_patch, weights=field.ravel()[positions], minlength=count,
            ) / sizes
    return tuple(_normals_from_covariance(value) for value in covariances)


def sheet_relations(edges, fine, context, probability, area, config):
    """Abrupt local direction changes beyond the smoother context scale.

    Equal orientation changes at both scales (ordinary smooth curvature) do
    not supply repulsion. Separator probability only changes merge ordering;
    it cannot erase or create a geometric repulsive constraint.
    """
    a, b = edges.T
    angle = lambda normal: np.degrees(np.arccos(np.clip(np.abs(np.sum(normal[a] * normal[b], axis=1)), 0, 1)))
    fine_angle, context_angle = angle(fine[0]), angle(context[0])
    reliability = np.minimum(fine[1][a], fine[1][b])
    excess = fine_angle - context_angle
    repulsive = (reliability >= config.minimum_coherence) & (fine_angle >= config.minimum_angle_deg) & (excess >= config.minimum_excess_deg)
    strength = area * (0.1 + 0.9 * reliability) * np.exp(-0.5 * (fine_angle / 20) ** 2)
    strength *= 1 - config.separator_support_weight * np.maximum(probability[a], probability[b])
    return strength, repulsive, fine_angle, excess, reliability


def supported_repulsion(edges, candidate, count, minimum_edges):
    """Reject isolated direction outliers using connected seam-edge support."""
    if not np.any(candidate):
        return candidate.copy()
    selected = edges[candidate]
    graph = coo_matrix((np.ones(len(selected)), selected.T), shape=(count, count)).tocsr()
    _, components = connected_components(graph, directed=False)
    support = np.bincount(components[selected[:, 0]], minlength=count)
    keep = np.zeros(len(edges), bool)
    keep[candidate] = support[components[selected[:, 0]]] >= minimum_edges
    return keep


def constrained_partition(count, edges, strength, repulsive):
    """Greedy signed-graph partition with global cannot-link constraints.

    Every repulsive endpoint pair remains in separate components, including
    when an arbitrarily strong positive bridge or alternate route exists.
    Smooth edges are revisable merge preferences, not irrevocable must-links.
    """
    parent = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, np.int32)
    forbidden = {}
    for a, b in edges[repulsive]:
        forbidden.setdefault(int(a), set()).add(int(b))
        forbidden.setdefault(int(b), set()).add(int(a))

    def root(value):
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    rejected = 0
    # Stable order makes tied, physically identical edges reproducible.
    for index in np.argsort(-strength, kind="stable"):
        if repulsive[index]:
            continue
        a, b = (root(value) for value in edges[index])
        if a == b:
            continue
        if b in forbidden.get(a, ()):
            rejected += 1
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]
        # Neighbour sets always refer to current roots; update both directions.
        neighbours = forbidden.pop(b, set())
        for other in neighbours:
            forbidden[other].discard(b)
            forbidden[other].add(a)
        if neighbours:
            forbidden.setdefault(a, set()).update(neighbours)
    roots = np.asarray([root(value) for value in range(count)], np.int32)
    _, partition = np.unique(roots, return_inverse=True)
    if np.any(partition[edges[repulsive, 0]] == partition[edges[repulsive, 1]]):
        raise AssertionError("A repulsive cortical relation was merged")
    return partition.astype(np.int32) + 1, rejected


def geometry_ownership(semantic, separator_probability, ct, spacing, config=GeometryConfig(), *, graph_output=None):
    """Automatic inference. No annotations, GT crop, or fragment count input."""
    config.validate()
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Need three positive physical spacings")
    if semantic.ndim != 3 or len({semantic.shape, separator_probability.shape, ct.shape}) != 1:
        raise ValueError("CT, semantic prediction and probabilities must share a 3D grid")
    if not np.all(np.isfinite(ct)) or not np.all(np.isfinite(separator_probability)):
        raise ValueError("Inputs must be finite")
    if np.any(separator_probability < 0) or np.any(separator_probability > 1):
        raise ValueError("Separator probabilities must lie in [0,1]")
    union = np.isin(semantic, (1, 2))
    markers, _, _ = separator_markers(union, separator_probability)
    base = watershed(separator_probability, markers=markers, mask=union, connectivity=CONNECTIVITY, watershed_line=False).astype(np.int32)
    record = {"config": asdict(config), "algorithm": "ct_sheet_signed_graph_v1", "GT_used_for_inference": False}
    if not union.any():
        return base, base.copy(), record
    positions, voxel_patch, edges, area, centers = patch_graph(base, spacing, config.patch_mm)
    count = len(centers)
    print(f"[GEOMETRY] voxels={len(positions)} patches={count} edges={len(edges)}", flush=True)
    fine, context = ct_sheet_features(np.asarray(ct, np.float32), spacing, positions, voxel_patch, config)
    sizes = np.bincount(voxel_patch, minlength=count)
    probability = np.bincount(voxel_patch, weights=separator_probability.ravel()[positions], minlength=count) / sizes
    strength, candidate, angles, excess, reliability = sheet_relations(edges, fine, context, probability, area, config)
    repulsive = supported_repulsion(edges, candidate, count, config.minimum_repulsive_edges)
    partition, rejected = constrained_partition(count, edges, strength, repulsive)
    output = np.zeros_like(base)
    output.ravel()[positions] = partition[voxel_patch]
    if not np.array_equal(output > 0, union):
        raise AssertionError("Geometry partition changed cortical coverage")
    if graph_output is not None:
        # Diagnostic cache contains prediction-derived features only. Saving it
        # permits an annotation audit without re-running or changing inference.
        np.savez_compressed(Path(graph_output), positions=positions, voxel_patch=voxel_patch,
                            edges=edges, area=area, centers_mm=centers, partition=partition,
                            fine_normals=fine[0], context_normals=context[0],
                            fine_coherence=fine[1], context_coherence=context[1],
                            strength=strength, candidate=candidate, repulsive=repulsive,
                            fine_angles_deg=angles, excess_deg=excess)
    record.update(
        patch_count=count, edge_count=len(edges), candidate_repulsive_edges=int(candidate.sum()),
        supported_repulsive_edges=int(repulsive.sum()), rejected_bridge_merges=rejected,
        baseline_instance_count=int(base.max()), geometry_instance_count=int(partition.max()),
        fine_angle_quantiles=np.quantile(angles, [0.5, 0.9, 0.99]).tolist() if len(edges) else [],
        excess_quantiles=np.quantile(excess, [0.5, 0.9, 0.99]).tolist() if len(edges) else [],
        reliable_edge_fraction=float(np.mean(reliability >= config.minimum_coherence)) if len(edges) else 0,
    )
    return output, base, record
