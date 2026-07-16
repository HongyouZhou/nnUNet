"""
Cortical-anchored fracture surface inference (v2).

Premise (anatomical prior, not learned):
    A bone fragment is a closed cortical shell that has been cut. A fracture
    is therefore a 2D surface whose 1D boundary lies on the cortical
    (label 1). The cortical itself has very strong local image cues (high HU,
    sharp ridge), so label 1 is the easy prediction; label 3 (the inter-
    fragment surface) is the hard one. We exploit (a) the easy label-1 signal
    and (b) the topological constraint that label 3's boundary anchors on
    label 1, to recover label 3 by post-processing without retraining.

Three failure-case taxonomy (Hongyou, 2026-04-25):
    Case 1: cortical shows two visible truncations / concave indentations,
            and the fracture surface between them is roughly planar.
    Case 2: only one cortical truncation is visible; the opposing rim was
            either eroded by trauma or invisible due to partial volume.
    Case 3: two cortical truncations visible, but the fracture surface is
            curved/convex (typical for fragments dominated by trabecular,
            low-density bone — collapsed regions). The geometric shortest
            path through the bone interior is wrong; the true cut follows
            the structural disruption, which lives in the network's soft
            ``prob_label_3`` even when argmax misses it.

Unified algorithm:
    For each candidate over-merged instance, generate "seed clusters" from
    multiple sources (cortical truncations, isolated cores, prob_label_3
    peaks). Pair seeds into source/sink, then run a 3D **graph max-flow /
    min-cut** with edge capacities derived from the network's softmax. The
    min-cut returns a true 2D cut surface that disconnects the volume.
    Per-case strategy lives entirely in seed selection and cost-field weights.

Design notes:
    - Min-cut is implemented via PyMaxflow (Boykov-Kolmogorov), which is the
      standard for graph-cut image segmentation and gives an actual 2D
      separating surface (unlike the v1 path-based prototype, which returned
      a 1D curve and could not disconnect a 3D blob).
    - The pipeline takes the network softmax (4 channels: bg/boundary/core/
      border) when available and falls back to argmax otherwise. Case 3
      essentially requires softmax to work.
    - Composes with the existing inference_abbc.py pipeline as a step
      between the initial abbc2instance call and ``_heal_by_splitting_quick``.

Validation checklist (do BEFORE wiring into inference_abbc.py):
    - Run on a real failing case via ``run_cortical_split.py``
    - Inspect ``label3_added.nii.gz`` next to the original prediction in
      ITK-Snap/napari. Cuts should land on real fracture surfaces.
    - Inspect ``diagnostics.json``: per-instance endpoint counts, cut
      success / disconnection counts, per-case classification.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_propagation,
    distance_transform_edt,
    gaussian_filter,
    label as nd_label,
)
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.morphology import ball, skeletonize
from skimage.segmentation import slic, watershed


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class SplitConfig:
    # ---- Component pre-filter
    min_cortical_voxels: int = 200
    min_instance_voxels: int = 500

    # ---- Seed source: cortical truncations
    cortical_smoothing_sigma: float = 1.0
    concavity_min_relative: float = 0.20      # threshold = this * max(score)
    seed_cluster_dilation: int = 2            # voxels to dilate around each picked seed
    endpoint_min_distance: int = 4            # NMS radius

    # ---- Seed source: prob_label_3 peaks (Case 3 backbone)
    prob3_seed_threshold: float = 0.20        # softmax cell value
    prob3_seed_min_size: int = 10

    # ---- Seed source: isolated cores
    use_isolated_cores: bool = True
    core_seed_min_size: int = 80

    # ---- Interactive core anchors
    core_anchor_erosion_mm: float = 1.0
    core_anchor_min_voxels: int = 20
    core_anchor_distinct_max_distance_mm: float = 15.0
    core_first_max_graph_voxels: int = 5_000_000

    # ---- Interactive geodesic bottleneck partition
    use_geodesic_bottleneck: bool = True
    bottleneck_band_width_mm: float = 1.0
    bottleneck_coarse_to_fine: bool = False
    bottleneck_coarse_band_width_mm: float = 12.0
    bottleneck_coarse_max_band_width_mm: float = 24.0
    bottleneck_coarse_expansion_factor: float = 2.0
    bottleneck_max_graph_nodes: int = 1_500_000
    bottleneck_voxel_graph_max_nodes: int = 100_000
    bottleneck_supervoxel_target_voxels: int = 400
    bottleneck_supervoxel_compactness: float = 0.08
    bottleneck_supervoxel_iterations: int = 3
    bottleneck_core_terminal_margin_mm: float = 5.0
    bottleneck_use_core_growth_seeds: bool = True
    bottleneck_max_full_feature_voxels: int = 10_000_000
    bottleneck_ct_smoothing_mm: float = 0.8
    bottleneck_hu_sigma: float = 120.0
    bottleneck_high_hu_cut_weight: float = 0.0
    bottleneck_high_hu_center_hu: float = 500.0
    bottleneck_high_hu_scale_hu: float = 200.0
    bottleneck_high_hu_protect_cortical: bool = True
    bottleneck_geodesic_gradient_weight: float = 2.0
    bottleneck_geodesic_label3_weight: float = 4.0
    bottleneck_surface_weight: float = 1.0
    bottleneck_unary_weight: float = 0.03
    bottleneck_label1_continuity: float = 0.75
    bottleneck_label2_continuity: float = 0.35
    bottleneck_label3_discount: float = 3.0
    bottleneck_edge_floor: float = 0.05

    # ---- Pairing
    pair_max_distance: int = 60               # voxels between two seed clusters
    pair_max_normal_dot: float = 0.0          # cortical pairs must roughly face

    # ---- Cost field for max-flow
    w_prob3: float = 4.0      # softmax label-3 prefers cuts here
    w_prob1: float = 6.0      # cortical (label 1) repels cut (don't slice cortical)
    w_prob2: float = 1.0      # core (label 2) is mid-cost
    w_bg_outside: float = 1e6 # outside the instance: infinite cost
    w_min: float = 0.05       # floor so max-flow doesn't explode on very low cost

    # ---- Validation
    min_split_piece_size: int = 50            # discard tiny crumbs after a cut


@dataclasses.dataclass
class Seed:
    """A connected source/sink region inside an instance, with provenance."""

    mask: np.ndarray            # bool, full-volume; only non-zero inside the instance
    centroid: np.ndarray        # (3,) float
    source: str                 # cortical_truncation | prob3_peak | isolated_core | manual | core_anchor
    normal: Optional[np.ndarray] = None   # cortical seeds carry an outward normal
    score: float = 0.0


@dataclasses.dataclass
class SeedPair:
    src: Seed
    snk: Seed
    case: int                   # 1, 2, or 3 — best-guess case label
    distance: float
    score: float                # higher = more likely a true fracture pair


@dataclasses.dataclass
class MinCutResult:
    """A complete binary partition and its diagnostic boundary."""

    source_mask: np.ndarray     # bool, full-volume; contains src_seed
    sink_mask: np.ndarray       # bool, full-volume; contains snk_seed
    cut_mask: np.ndarray        # bool, both sides of source/sink boundary
    flow: float


@dataclasses.dataclass
class CoreAnchorResult:
    """Core regions that must remain in the two interactive output instances."""

    source_mask: np.ndarray
    sink_mask: np.ndarray
    raw_core_mask: np.ndarray
    robust_core_mask: np.ndarray
    diagnostics: dict


# ----------------------------------------------------------------------------
# Cortical surface signature (cleaned up from v1)
# ----------------------------------------------------------------------------
def _signed_distance_field(mask: np.ndarray, sigma: float) -> np.ndarray:
    dist_out = distance_transform_edt(~mask)
    dist_in = distance_transform_edt(mask)
    sdf = (dist_out - dist_in).astype(np.float32)
    return gaussian_filter(sdf, sigma=sigma)


def _laplacian(field: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field)
    for ax in range(field.ndim):
        if field.shape[ax] > 1:
            out += np.gradient(np.gradient(field, axis=ax), axis=ax)
    return out


def _normals_from_sdf(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    sdf = _signed_distance_field(mask, sigma)
    grads = np.stack(np.gradient(sdf), axis=0)
    n = np.linalg.norm(grads, axis=0) + 1e-8
    return grads / n


def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    k = np.ones((3, 3, 3), dtype=np.uint8)
    k[1, 1, 1] = 0
    return convolve(skel.astype(np.uint8), k, mode="constant", cval=0) * skel.astype(np.uint8)


def cortical_truncation_seeds(
    cortical_mask: np.ndarray,
    instance_mask: np.ndarray,
    cfg: SplitConfig,
) -> list[Seed]:
    """Detect cortical truncations / concave kinks as seed regions."""
    if int(cortical_mask.sum()) < cfg.min_cortical_voxels:
        return []

    # Source A: skeleton endpoints / branch points (truly topological)
    skel = skeletonize(cortical_mask)
    deg = _skeleton_degree(skel)
    skel_pts = ((deg == 1) | (deg >= 3)) & skel

    # Source B: SDF Laplacian peaks restricted to skeleton (concave indentations)
    sdf = _signed_distance_field(cortical_mask, cfg.cortical_smoothing_sigma)
    lap = _laplacian(sdf)
    score_field = np.where(skel, lap, 0.0)
    if float(score_field.max()) > 0:
        thr = cfg.concavity_min_relative * float(score_field.max())
        conc_pts = (score_field > thr) & skel
    else:
        conc_pts = np.zeros_like(skel, dtype=bool)

    candidates = skel_pts | conc_pts
    if not np.any(candidates):
        return []

    # NMS
    coords = np.argwhere(candidates)
    ranks = np.zeros(len(coords), dtype=np.float32)
    for i, c in enumerate(coords):
        d = int(deg[c[0], c[1], c[2]])
        s = float(score_field[c[0], c[1], c[2]])
        ranks[i] = (100.0 if (d == 1 or d >= 3) else 0.0) + s

    order = np.argsort(-ranks)
    picked: list[int] = []
    taken = np.zeros(len(coords), dtype=bool)
    r = cfg.endpoint_min_distance
    for idx in order:
        if taken[idx]:
            continue
        picked.append(idx)
        d_cheb = np.abs(coords - coords[idx]).max(axis=1)
        taken |= d_cheb <= r

    normals = _normals_from_sdf(cortical_mask)
    seeds: list[Seed] = []
    for i in picked:
        c = coords[i]
        seed_mask = np.zeros(cortical_mask.shape, dtype=bool)
        seed_mask[c[0], c[1], c[2]] = True
        seed_mask = binary_dilation(seed_mask, structure=ball(cfg.seed_cluster_dilation))
        seed_mask &= instance_mask
        if not np.any(seed_mask):
            continue
        seeds.append(Seed(
            mask=seed_mask,
            centroid=c.astype(np.float32),
            source="cortical_truncation",
            normal=normals[:, c[0], c[1], c[2]],
            score=float(ranks[i]),
        ))
    return seeds


def prob3_peak_seeds(
    prob_label_3: np.ndarray,
    instance_mask: np.ndarray,
    cfg: SplitConfig,
) -> list[Seed]:
    """Connected components of prob_label_3 above threshold inside the instance."""
    if prob_label_3 is None:
        return []
    cand = (prob_label_3 > cfg.prob3_seed_threshold) & instance_mask
    if not np.any(cand):
        return []
    labelled, n = nd_label(cand)
    seeds: list[Seed] = []
    for cc_id in range(1, n + 1):
        m = labelled == cc_id
        if int(m.sum()) < cfg.prob3_seed_min_size:
            continue
        coords = np.argwhere(m)
        centroid = coords.mean(axis=0)
        seeds.append(Seed(
            mask=m, centroid=centroid, source="prob3_peak",
            normal=None,
            score=float(prob_label_3[m].mean()),
        ))
    return seeds


def isolated_core_seeds(
    abbc_pred: np.ndarray,
    instance_mask: np.ndarray,
    cfg: SplitConfig,
) -> list[Seed]:
    """If multiple disconnected cores exist within the instance, each is a seed."""
    cores = (abbc_pred == 2) & instance_mask
    if not np.any(cores):
        return []
    labelled, n = nd_label(cores)
    if n < 2:
        return []
    seeds: list[Seed] = []
    for cc_id in range(1, n + 1):
        m = labelled == cc_id
        if int(m.sum()) < cfg.core_seed_min_size:
            continue
        coords = np.argwhere(m)
        centroid = coords.mean(axis=0)
        seeds.append(Seed(
            mask=m, centroid=centroid, source="isolated_core",
            normal=None, score=float(m.sum()),
        ))
    return seeds


# ----------------------------------------------------------------------------
# Seed pairing (per-case)
# ----------------------------------------------------------------------------
def pair_seeds(
    seeds: list[Seed],
    instance_mask: np.ndarray,
    cfg: SplitConfig,
) -> list[SeedPair]:
    """Greedy face-to-face pairing across heterogeneous seed sources."""
    if len(seeds) < 2:
        return []
    candidates: list[SeedPair] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            si, sj = seeds[i], seeds[j]
            d = float(np.linalg.norm(si.centroid - sj.centroid))
            if d > cfg.pair_max_distance:
                continue

            # Midpoint sanity: must be inside instance
            mid = ((si.centroid + sj.centroid) / 2).astype(int)
            if not _safe_index(instance_mask, mid):
                continue

            # Case classification + scoring
            case, score = _classify_pair(si, sj, d, cfg)
            if case == 0:
                continue
            candidates.append(SeedPair(src=si, snk=sj, case=case,
                                       distance=d, score=score))

    candidates.sort(key=lambda p: -p.score)
    # NOTE: do NOT enforce "each seed used once". An over-merged N-fragment
    # instance needs N-1 cuts; if seeds were single-use, only 1 of 3 cuts
    # would fire. The orchestrator refreshes inst_mask after each cut, and
    # min_cut_surface returns None when both seeds aren't in the same
    # current piece — so reusing seeds across pairs is safe and correct.
    return candidates


def _classify_pair(si: Seed, sj: Seed, d: float, cfg: SplitConfig) -> tuple[int, float]:
    """Return (case, score). case=0 means reject.

    Score scale is calibrated so that *more reliable* pair types win first.
    Empirically, isolated_core ↔ isolated_core is the most reliable single
    signal because two distinct cores virtually guarantee two real fragments.
    Cortical truncations are noisier (skeleton endpoints fire on convex
    corners too), so they only win when no core pair exists.
    """
    # Two isolated cores — strongest signal: the network already saw two
    # distinct cores, just lacked the boundary to split them.
    if si.source == "isolated_core" and sj.source == "isolated_core":
        return 3, 50.0 - 0.05 * d

    # Mixed prob3 + core — Case 3 (curved fracture aligned with prob3 ridge)
    if {si.source, sj.source} == {"prob3_peak", "isolated_core"}:
        return 3, 30.0 - 0.05 * d + (si.score + sj.score)

    # Two prob3 peaks — Case 3 (network soft signal on both ends)
    if si.source == "prob3_peak" and sj.source == "prob3_peak":
        return 3, 20.0 - 0.04 * d + (si.score + sj.score)

    # One cortical + one prob3 → Case 2 (single visible cortical end)
    if {si.source, sj.source} == {"cortical_truncation", "prob3_peak"}:
        return 2, 15.0 - 0.04 * d + (si.score + sj.score) * 0.1

    # One cortical + one isolated core → Case 2 fallback
    if {si.source, sj.source} == {"cortical_truncation", "isolated_core"}:
        return 2, 12.0 - 0.03 * d

    # Both cortical — weakest source (skeleton endpoints have many false positives
    # at convex hull corners). Require facing-each-other.
    if si.source == "cortical_truncation" and sj.source == "cortical_truncation":
        if si.normal is not None and sj.normal is not None:
            dot = float(np.dot(si.normal, sj.normal))
            if dot > cfg.pair_max_normal_dot:
                return 0, 0.0
            return 1, 5.0 - 0.05 * d - dot * 2.0
        return 1, 1.0 - 0.02 * d

    return 0, 0.0


def _safe_index(arr: np.ndarray, idx: np.ndarray) -> bool:
    if any(idx[k] < 0 or idx[k] >= arr.shape[k] for k in range(arr.ndim)):
        return False
    return bool(arr[idx[0], idx[1], idx[2]])


# ----------------------------------------------------------------------------
# Min-cut (PyMaxflow, Boykov-Kolmogorov)
# ----------------------------------------------------------------------------
def min_cut_partition(
    instance_mask: np.ndarray,
    src_seed: Seed,
    snk_seed: Seed,
    cost_field: np.ndarray,
    bbox_pad: int = 4,
) -> Optional[MinCutResult]:
    """Partition one 3D instance between source and sink seed regions.

    The cut surface contains the instance voxels on both sides of every cut
    edge. The partition itself covers every input instance voxel, so applying
    it to an instance map does not leave an artificial gap.

    Returns None if PyMaxflow isn't installed or the seeds aren't separable.
    """
    if instance_mask.ndim != 3:
        raise ValueError(f"instance_mask must be 3D, got shape {instance_mask.shape}")
    if src_seed.mask.shape != instance_mask.shape or snk_seed.mask.shape != instance_mask.shape:
        raise ValueError("seed masks must have the same shape as instance_mask")
    if cost_field.shape != instance_mask.shape:
        raise ValueError("cost_field must have the same shape as instance_mask")
    if np.any(src_seed.mask & snk_seed.mask):
        return None
    if not np.all(np.isfinite(cost_field)) or np.any(cost_field < 0):
        raise ValueError("cost_field must contain finite non-negative values")
    if bbox_pad < 0:
        raise ValueError("bbox_pad must be non-negative")

    try:
        import maxflow
    except ImportError:
        return None

    # Restrict to a padded bbox of the instance for memory
    inst_coords = np.argwhere(instance_mask)
    if len(inst_coords) == 0:
        return None
    z0, y0, x0 = inst_coords.min(axis=0)
    z1, y1, x1 = inst_coords.max(axis=0) + 1
    z0 = max(0, z0 - bbox_pad)
    y0 = max(0, y0 - bbox_pad)
    x0 = max(0, x0 - bbox_pad)
    z1 = min(instance_mask.shape[0], z1 + bbox_pad)
    y1 = min(instance_mask.shape[1], y1 + bbox_pad)
    x1 = min(instance_mask.shape[2], x1 + bbox_pad)
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    inside = instance_mask[sl]
    cost = cost_field[sl].copy()
    src_local = src_seed.mask[sl] & inside
    snk_local = snk_seed.mask[sl] & inside
    if not src_local.any() or not snk_local.any():
        return None

    # Build graph. Nodes outside the irregular instance are left disconnected
    # from it; connecting them to sink would bias the cut toward enclosing the
    # source seed along the object's external surface.
    g = maxflow.Graph[float]()
    node_ids = g.add_grid_nodes(inside.shape)

    finite_capacity_sum = 0.0
    for axis in range(3):
        struct = np.zeros((3, 3, 3), dtype=int)
        nbr = [1, 1, 1]
        nbr[axis] = 2
        struct[tuple(nbr)] = 1

        next_cost = np.roll(cost, shift=-1, axis=axis)
        next_inside = np.roll(inside, shift=-1, axis=axis)
        valid_edge = inside & next_inside
        boundary = [slice(None)] * 3
        boundary[axis] = -1
        valid_edge[tuple(boundary)] = False
        cap = np.where(valid_edge, 0.5 * (cost + next_cost), 0.0)
        finite_capacity_sum += float(cap.sum())
        g.add_grid_edges(node_ids, weights=cap, structure=struct, symmetric=True)

    # A terminal capacity larger than all finite pairwise capacities makes the
    # two seed constraints effectively hard without relying on a magic value.
    INF = max(1.0, 2.0 * finite_capacity_sum + 1.0)
    src_caps = np.zeros(inside.shape, dtype=np.float32)
    snk_caps = np.zeros(inside.shape, dtype=np.float32)
    src_caps[src_local] = INF
    snk_caps[snk_local] = INF
    g.add_grid_tedges(node_ids, src_caps, snk_caps)

    flow = float(g.maxflow())
    sgm = g.get_grid_segments(node_ids)

    # PyMaxflow's boolean convention is an implementation detail. Infer the
    # two sides from the hard seeds and reject a violated terminal constraint.
    src_values = np.unique(sgm[src_local])
    snk_values = np.unique(sgm[snk_local])
    if len(src_values) != 1 or len(snk_values) != 1:
        return None
    src_value = bool(src_values[0])
    snk_value = bool(snk_values[0])
    if src_value == snk_value:
        return None

    source_local = inside & (sgm == src_value)
    sink_local = inside & (sgm == snk_value)

    # Cut voxels = those whose 6-neighbour has different segment label, AND inside instance.
    # We mark BOTH sides of the cut to get a 2-vx-thick surface that reliably disconnects.
    cut_mask = np.zeros(inside.shape, dtype=bool)
    for axis in range(3):
        sl_self = [slice(None)] * 3
        sl_next = [slice(None)] * 3
        sl_self[axis] = slice(0, -1)
        sl_next[axis] = slice(1, None)
        sl_self = tuple(sl_self)
        sl_next = tuple(sl_next)
        diff = (
            (sgm[sl_self] != sgm[sl_next])
            & inside[sl_self]
            & inside[sl_next]
        )
        cut_mask[sl_self] |= diff
        cut_mask[sl_next] |= diff
    cut_mask &= inside

    source_full = np.zeros(instance_mask.shape, dtype=bool)
    sink_full = np.zeros(instance_mask.shape, dtype=bool)
    cut_full = np.zeros(instance_mask.shape, dtype=bool)
    source_full[sl] = source_local
    sink_full[sl] = sink_local
    cut_full[sl] = cut_mask
    return MinCutResult(source_full, sink_full, cut_full, flow)


def min_cut_surface(
    instance_mask: np.ndarray,
    src_seed: Seed,
    snk_seed: Seed,
    cost_field: np.ndarray,
    bbox_pad: int = 4,
) -> Optional[np.ndarray]:
    """Backward-compatible wrapper returning only the cut boundary."""
    result = min_cut_partition(instance_mask, src_seed, snk_seed, cost_field, bbox_pad)
    return None if result is None else result.cut_mask


# ----------------------------------------------------------------------------
# Cost field
# ----------------------------------------------------------------------------
def build_cost_field(
    abbc_pred: np.ndarray,
    softmax: Optional[np.ndarray],
    instance_mask: np.ndarray,
    cfg: SplitConfig,
    prob_label_3: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Voxel cost: low where we'd like the cut to go, high where it shouldn't.

    cost = w_min
         + w_prob1 * prob_label_1     (cortical: high → cut should avoid)
         + w_prob2 * prob_label_2     (core: medium repel)
         - w_prob3 * prob_label_3     (border: low cost — cut prefers)
    Outside the instance: w_bg_outside (effectively infinite).
    """
    if instance_mask.shape != abbc_pred.shape:
        raise ValueError(
            f"instance_mask shape {instance_mask.shape} != abbc_pred shape {abbc_pred.shape}"
        )
    p1, p2, p3 = _probability_maps(abbc_pred, softmax, prob_label_3)

    cost = (
        cfg.w_min
        + cfg.w_prob1 * p1
        + cfg.w_prob2 * p2
        - cfg.w_prob3 * p3
    )
    cost = np.maximum(cost, cfg.w_min)
    cost[~instance_mask] = cfg.w_bg_outside
    return cost.astype(np.float32)


def _probability_maps(
    abbc_pred: np.ndarray,
    softmax: Optional[np.ndarray],
    prob_label_3: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve validated probability maps from full softmax or hard labels."""
    if abbc_pred.ndim != 3:
        raise ValueError(f"abbc_pred must be 3D, got shape {abbc_pred.shape}")

    if softmax is not None:
        if softmax.ndim != 4 or softmax.shape[0] < 4 or softmax.shape[1:] != abbc_pred.shape:
            raise ValueError(
                "softmax must have shape (>=4, *abbc_pred.shape), "
                f"got {softmax.shape} for {abbc_pred.shape}"
            )
        p1 = np.asarray(softmax[1], dtype=np.float32)
        p2 = np.asarray(softmax[2], dtype=np.float32)
        p3 = np.asarray(softmax[3], dtype=np.float32)
    else:
        p1 = (abbc_pred == 1).astype(np.float32)
        p2 = (abbc_pred == 2).astype(np.float32)
        p3 = (abbc_pred == 3).astype(np.float32)

    if prob_label_3 is not None:
        if prob_label_3.shape != abbc_pred.shape:
            raise ValueError(
                f"prob_label_3 shape {prob_label_3.shape} != abbc_pred shape {abbc_pred.shape}"
            )
        p3 = np.asarray(prob_label_3, dtype=np.float32)

    for name, prob in (("prob_label_1", p1), ("prob_label_2", p2), ("prob_label_3", p3)):
        if not np.all(np.isfinite(prob)):
            raise ValueError(f"{name} contains NaN or infinite values")

    return tuple(np.clip(p, 0.0, 1.0) for p in (p1, p2, p3))


def _mask_bbox_slices(mask: np.ndarray, pad: int = 4) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("cannot compute a bounding box for an empty mask")
    lower = np.maximum(coords.min(axis=0) - pad, 0)
    upper = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _validated_spacing_zyx(spacing_zyx: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three finite positive values")
    return spacing


def _assign_unseeded_instance_components(
    partition: np.ndarray,
    instance_mask: np.ndarray,
    source_terminal: np.ndarray,
    sink_terminal: np.ndarray,
    spacing_zyx: Sequence[float],
) -> dict[str, int]:
    """Assign marker-free instance components without changing instance support.

    Marker watershed leaves a disconnected mask component at zero when it has
    neither terminal. Such a component is assigned as a whole to the physically
    nearest terminal. This decision uses only the instance geometry and prompts;
    ABBC labels never define or expand the output support.
    """
    uncovered = np.asarray(instance_mask, dtype=bool) & (partition == 0)
    labelled, component_count = nd_label(uncovered)
    diagnostics = {
        "unseeded_component_count": int(component_count),
        "unseeded_components_assigned_source": 0,
        "unseeded_components_assigned_sink": 0,
        "unseeded_voxels_assigned_source": 0,
        "unseeded_voxels_assigned_sink": 0,
    }
    if component_count == 0:
        return diagnostics

    spacing = _validated_spacing_zyx(spacing_zyx)
    source_coordinates = np.argwhere(source_terminal)
    sink_coordinates = np.argwhere(sink_terminal)
    if source_coordinates.size == 0 or sink_coordinates.size == 0:
        raise ValueError("both source and sink terminals must contain a voxel")
    source_tree = cKDTree(source_coordinates * spacing)
    sink_tree = cKDTree(sink_coordinates * spacing)

    for component_id in range(1, component_count + 1):
        component = labelled == component_id
        coordinates = np.argwhere(component) * spacing
        source_distance = float(np.min(source_tree.query(coordinates, k=1)[0]))
        sink_distance = float(np.min(sink_tree.query(coordinates, k=1)[0]))
        voxel_count = int(component.sum())
        if source_distance <= sink_distance:
            partition[component] = 1
            diagnostics["unseeded_components_assigned_source"] += 1
            diagnostics["unseeded_voxels_assigned_source"] += voxel_count
        else:
            partition[component] = 2
            diagnostics["unseeded_components_assigned_sink"] += 1
            diagnostics["unseeded_voxels_assigned_sink"] += voxel_count
    return diagnostics


def _partition_boundary_mask(
    source_mask: np.ndarray,
    sink_mask: np.ndarray,
    instance_mask: np.ndarray,
) -> np.ndarray:
    """Return both voxel sides of every 6-neighbour partition boundary."""
    cut_mask = np.zeros(instance_mask.shape, dtype=bool)
    for axis in range(3):
        current = [slice(None)] * 3
        following = [slice(None)] * 3
        current[axis] = slice(0, -1)
        following[axis] = slice(1, None)
        current_slice = tuple(current)
        following_slice = tuple(following)
        boundary = (
            (source_mask[current_slice] & sink_mask[following_slice])
            | (sink_mask[current_slice] & source_mask[following_slice])
        ) & instance_mask[current_slice] & instance_mask[following_slice]
        cut_mask[current_slice] |= boundary
        cut_mask[following_slice] |= boundary
    return cut_mask


def _repair_terminal_connectivity(
    result: MinCutResult,
    instance_mask: np.ndarray,
    source_terminal: np.ndarray,
    sink_terminal: np.ndarray,
    max_iterations: int = 4,
) -> tuple[MinCutResult, dict[str, int]]:
    """Reassign floating cut islands to the terminal-connected opposite side.

    Disconnected components of the input instance are left intact unless they
    touch the opposite terminal-connected partition. This preserves exact input
    support while removing islands introduced by the graph cut itself.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    arrays = (result.source_mask, result.sink_mask, source_terminal, sink_terminal)
    if any(array.shape != instance_mask.shape for array in arrays):
        raise ValueError("partition and terminal masks must match instance_mask")
    if np.any(source_terminal & sink_terminal):
        raise ValueError("source and sink connectivity terminals must not overlap")
    if np.any(source_terminal & ~instance_mask) or np.any(
        sink_terminal & ~instance_mask
    ):
        raise ValueError("connectivity terminals must lie inside instance_mask")

    source_mask = np.asarray(result.source_mask, dtype=bool).copy()
    sink_mask = np.asarray(result.sink_mask, dtype=bool).copy()
    source_mask[source_terminal] = True
    sink_mask[source_terminal] = False
    sink_mask[sink_terminal] = True
    source_mask[sink_terminal] = False
    if not np.array_equal(source_mask | sink_mask, instance_mask):
        raise RuntimeError("connectivity repair requires a complete binary partition")

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = structure[2, 1, 1] = True
    structure[1, 0, 1] = structure[1, 2, 1] = True
    structure[1, 1, 0] = structure[1, 1, 2] = True
    source_reassigned_voxels = 0
    sink_reassigned_voxels = 0
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        source_reachable = binary_propagation(
            source_terminal,
            structure=structure,
            mask=source_mask,
        )
        sink_reachable = binary_propagation(
            sink_terminal,
            structure=structure,
            mask=sink_mask,
        )
        source_floating = source_mask & ~source_reachable
        sink_floating = sink_mask & ~sink_reachable
        source_touching_sink = source_floating & binary_dilation(
            sink_reachable,
            structure=structure,
        )
        sink_touching_source = sink_floating & binary_dilation(
            source_reachable,
            structure=structure,
        )
        source_to_sink = binary_propagation(
            source_touching_sink,
            structure=structure,
            mask=source_floating,
        )
        sink_to_source = binary_propagation(
            sink_touching_source,
            structure=structure,
            mask=sink_floating,
        )
        source_count = int(source_to_sink.sum())
        sink_count = int(sink_to_source.sum())
        if source_count == 0 and sink_count == 0:
            break

        source_mask[source_to_sink] = False
        sink_mask[source_to_sink] = True
        sink_mask[sink_to_source] = False
        source_mask[sink_to_source] = True
        source_reassigned_voxels += source_count
        sink_reassigned_voxels += sink_count
        iterations = iteration

    source_reachable = binary_propagation(
        source_terminal,
        structure=structure,
        mask=source_mask,
    )
    sink_reachable = binary_propagation(
        sink_terminal,
        structure=structure,
        mask=sink_mask,
    )
    repaired = MinCutResult(
        source_mask=source_mask,
        sink_mask=sink_mask,
        cut_mask=_partition_boundary_mask(source_mask, sink_mask, instance_mask),
        flow=result.flow,
    )
    diagnostics = {
        "topology_cleanup_iterations": iterations,
        "source_floating_voxels_reassigned_to_sink": source_reassigned_voxels,
        "sink_floating_voxels_reassigned_to_source": sink_reassigned_voxels,
        "source_unanchored_voxels_remaining": int(
            np.count_nonzero(source_mask & ~source_reachable)
        ),
        "sink_unanchored_voxels_remaining": int(
            np.count_nonzero(sink_mask & ~sink_reachable)
        ),
    }
    return repaired, diagnostics


def _bottleneck_image_features(
    image: Optional[np.ndarray],
    instance_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    cfg: SplitConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create a smoothed CT and a robust physical gradient feature."""
    if image is None:
        return None, None
    if image.shape != instance_mask.shape:
        raise ValueError("image and instance_mask must have the same shape")
    image = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(image[instance_mask])):
        raise ValueError("image contains non-finite values inside the selected instance")
    if cfg.bottleneck_ct_smoothing_mm < 0:
        raise ValueError("bottleneck_ct_smoothing_mm must be non-negative")

    sigma = cfg.bottleneck_ct_smoothing_mm / spacing_zyx
    smoothed = gaussian_filter(image, sigma=sigma, output=np.float32)
    gradient_squared = np.zeros(image.shape, dtype=np.float32)
    for component in np.gradient(smoothed, *spacing_zyx):
        gradient_squared += np.asarray(component, dtype=np.float32) ** 2
    np.sqrt(gradient_squared, out=gradient_squared)
    values = gradient_squared[instance_mask]
    scale = float(np.percentile(values, 99.0)) if values.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-6:
        gradient_squared.fill(0.0)
    else:
        np.clip(gradient_squared / scale, 0.0, 1.0, out=gradient_squared)
    return smoothed, gradient_squared


def _high_hu_cut_discount(
    current_hu: np.ndarray,
    following_hu: np.ndarray,
    edge_p1: np.ndarray,
    cfg: SplitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a low-capacity preference for high-HU non-cortical edges."""
    if cfg.bottleneck_high_hu_scale_hu <= 0:
        raise ValueError("bottleneck_high_hu_scale_hu must be positive")
    shape = np.broadcast_shapes(current_hu.shape, following_hu.shape, edge_p1.shape)
    if cfg.bottleneck_high_hu_cut_weight <= 0:
        return (
            np.ones(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )

    edge_hu = 0.5 * (
        np.asarray(current_hu, dtype=np.float32)
        + np.asarray(following_hu, dtype=np.float32)
    )
    centered_hu = np.clip(
        (edge_hu - cfg.bottleneck_high_hu_center_hu)
        / cfg.bottleneck_high_hu_scale_hu,
        -20.0,
        20.0,
    )
    high_hu_score = 1.0 / (1.0 + np.exp(-centered_hu))
    if cfg.bottleneck_high_hu_protect_cortical:
        high_hu_score *= np.clip(
            1.0 - np.asarray(edge_p1, dtype=np.float32),
            0.0,
            1.0,
        )
    discount = np.exp(
        -cfg.bottleneck_high_hu_cut_weight * high_hu_score
    ).astype(np.float32)
    return discount, high_hu_score.astype(np.float32, copy=False)


def _geodesic_seed_ownership(
    instance_mask: np.ndarray,
    source_seed: np.ndarray,
    sink_seed: np.ndarray,
    propagation_cost: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Grow two fronts through the instance and return their full ownership."""
    sampling = tuple(float(value) for value in spacing_zyx)
    source_mcp = MCP_Geometric(
        propagation_cost,
        fully_connected=False,
        sampling=sampling,
    )
    source_distance, _ = source_mcp.find_costs(np.argwhere(source_seed))
    del source_mcp
    sink_mcp = MCP_Geometric(
        propagation_cost,
        fully_connected=False,
        sampling=sampling,
    )
    sink_distance, _ = sink_mcp.find_costs(np.argwhere(sink_seed))
    del sink_mcp

    source_finite = np.isfinite(source_distance) & instance_mask
    sink_finite = np.isfinite(sink_distance) & instance_mask
    partition = np.zeros(instance_mask.shape, dtype=np.uint8)
    both = source_finite & sink_finite
    partition[both & (source_distance <= sink_distance)] = 1
    partition[both & (source_distance > sink_distance)] = 2
    partition[source_finite & ~sink_finite] = 1
    partition[sink_finite & ~source_finite] = 2
    disconnected_diagnostics = _assign_unseeded_instance_components(
        partition,
        instance_mask,
        source_seed,
        sink_seed,
        spacing_zyx,
    )
    return partition, source_distance, sink_distance, disconnected_diagnostics


def _bottleneck_graph_node_map(
    active_band: np.ndarray,
    smoothed_image: Optional[np.ndarray],
    p1: np.ndarray,
    p3: np.ndarray,
    spacing_zyx: np.ndarray,
    cfg: SplitConfig,
) -> tuple[np.ndarray, int, str]:
    """Map active voxels to either voxel nodes or CT-aligned supervoxels."""
    active_count = int(active_band.sum())
    voxel_graph_limit = int(cfg.bottleneck_voxel_graph_max_nodes)
    if voxel_graph_limit < 0:
        raise ValueError("bottleneck_voxel_graph_max_nodes must be non-negative")
    target = int(cfg.bottleneck_supervoxel_target_voxels)
    if target < 1:
        raise ValueError("bottleneck_supervoxel_target_voxels must be positive")
    if cfg.bottleneck_supervoxel_compactness < 0:
        raise ValueError("bottleneck_supervoxel_compactness must be non-negative")
    if cfg.bottleneck_supervoxel_iterations < 1:
        raise ValueError("bottleneck_supervoxel_iterations must be positive")

    node_map = np.full(active_band.shape, -1, dtype=np.int32)
    if active_count <= max(voxel_graph_limit, 4 * target):
        node_map[active_band] = np.arange(active_count, dtype=np.int32)
        return node_map, active_count, "voxel"

    if smoothed_image is None:
        intensity = np.zeros(active_band.shape, dtype=np.float32)
    else:
        values = smoothed_image[active_band]
        lower, upper = np.percentile(values, (1.0, 99.0))
        scale = max(float(upper - lower), 1.0)
        intensity = np.clip(
            (smoothed_image - float(lower)) / scale,
            0.0,
            1.0,
        ).astype(np.float32)
    features = np.stack((intensity, 1.5 * p3, 0.25 * p1), axis=-1)
    requested_segments = max(2, int(np.ceil(active_count / target)))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="One of the clusters is empty.*",
            category=UserWarning,
        )
        supervoxels = slic(
            features,
            n_segments=requested_segments,
            compactness=cfg.bottleneck_supervoxel_compactness,
            max_num_iter=cfg.bottleneck_supervoxel_iterations,
            sigma=0,
            spacing=tuple(float(value) for value in spacing_zyx),
            convert2lab=False,
            enforce_connectivity=True,
            min_size_factor=0.25,
            max_size_factor=4.0,
            mask=active_band,
            start_label=1,
            channel_axis=-1,
        )
    labels = np.unique(supervoxels[active_band])
    labels = labels[labels > 0]
    if labels.size < 2:
        node_map[active_band] = np.arange(active_count, dtype=np.int32)
        return node_map, active_count, "voxel_fallback"
    lookup = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    lookup[labels] = np.arange(labels.size, dtype=np.int32)
    node_map[active_band] = lookup[supervoxels[active_band]]
    return node_map, int(labels.size), "slic_supervoxel"


def _narrow_band_bottleneck_mincut(
    instance_mask: np.ndarray,
    initial_partition: np.ndarray,
    active_band: np.ndarray,
    source_seed: np.ndarray,
    sink_seed: np.ndarray,
    source_distance: np.ndarray,
    sink_distance: np.ndarray,
    smoothed_image: Optional[np.ndarray],
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    spacing_zyx: np.ndarray,
    cfg: SplitConfig,
) -> tuple[Optional[MinCutResult], dict]:
    """Optimize a minimum-capacity separating surface in an active band."""
    try:
        import maxflow
    except ImportError:
        return None, {"bottleneck_graph_fallback_reason": "PyMaxflow is unavailable"}

    active_band = np.asarray(active_band, dtype=bool) & instance_mask
    active_band |= source_seed | sink_seed
    active_count = int(active_band.sum())
    if active_count == 0:
        return None, {"bottleneck_graph_fallback_reason": "active band is empty"}
    logical_node_map, graph_node_count, graph_representation = (
        _bottleneck_graph_node_map(
            active_band,
            smoothed_image,
            p1,
            p3,
            spacing_zyx,
            cfg,
        )
    )
    if (
        cfg.bottleneck_max_graph_nodes > 0
        and graph_node_count > cfg.bottleneck_max_graph_nodes
    ):
        return None, {
            "bottleneck_graph_fallback_reason": (
                f"bottleneck graph has {graph_node_count} nodes, above the limit "
                f"{cfg.bottleneck_max_graph_nodes}"
            )
        }

    graph = maxflow.Graph[float](graph_node_count, graph_node_count * 8)
    graph_nodes = np.asarray(graph.add_nodes(graph_node_count))
    node_map = np.full(instance_mask.shape, -1, dtype=np.int32)
    node_map[active_band] = graph_nodes[logical_node_map[active_band]].astype(
        np.int32,
        copy=False,
    )

    voxel_volume = float(np.prod(spacing_zyx))
    finite_both = (
        np.isfinite(source_distance[active_band])
        & np.isfinite(sink_distance[active_band])
    )
    source_cost = np.full(active_count, 0.5, dtype=np.float64)
    sink_cost = np.full(active_count, 0.5, dtype=np.float64)
    if np.any(finite_both):
        source_values = source_distance[active_band][finite_both]
        sink_values = sink_distance[active_band][finite_both]
        denominator = np.maximum(source_values + sink_values, 1e-6)
        source_cost[finite_both] = source_values / denominator
        sink_cost[finite_both] = sink_values / denominator
    only_source = np.isfinite(source_distance[active_band]) & ~np.isfinite(
        sink_distance[active_band]
    )
    only_sink = np.isfinite(sink_distance[active_band]) & ~np.isfinite(
        source_distance[active_band]
    )
    source_cost[only_source], sink_cost[only_source] = 0.0, 1.0
    source_cost[only_sink], sink_cost[only_sink] = 1.0, 0.0
    unary_scale = float(cfg.bottleneck_unary_weight) * voxel_volume
    # SOURCE->node is cut for a sink assignment; node->SINK for source.
    active_nodes = node_map[active_band]
    source_caps = np.bincount(
        active_nodes,
        weights=unary_scale * sink_cost,
        minlength=graph_node_count,
    ).astype(np.float64, copy=False)
    sink_caps = np.bincount(
        active_nodes,
        weights=unary_scale * source_cost,
        minlength=graph_node_count,
    ).astype(np.float64, copy=False)

    if cfg.bottleneck_hu_sigma <= 0:
        raise ValueError("bottleneck_hu_sigma must be positive")
    edge_count = 0
    pairwise_capacity_sum = 0.0
    high_hu_score_sum = 0.0
    high_hu_discount_sum = 0.0
    high_hu_edge_count = 0
    for axis in range(3):
        current = [slice(None)] * 3
        following = [slice(None)] * 3
        current[axis] = slice(0, -1)
        following[axis] = slice(1, None)
        current_slice = tuple(current)
        following_slice = tuple(following)
        inside_pair = instance_mask[current_slice] & instance_mask[following_slice]
        if not np.any(inside_pair):
            continue

        edge_p1 = 0.5 * (p1[current_slice] + p1[following_slice])
        if smoothed_image is None:
            hu_affinity = np.ones(inside_pair.shape, dtype=np.float32)
            high_hu_discount = np.ones(inside_pair.shape, dtype=np.float32)
            high_hu_score = np.zeros(inside_pair.shape, dtype=np.float32)
        else:
            hu_difference = np.abs(
                smoothed_image[current_slice] - smoothed_image[following_slice]
            )
            hu_affinity = np.exp(
                -0.5 * (hu_difference / cfg.bottleneck_hu_sigma) ** 2
            ).astype(np.float32)
            high_hu_discount, high_hu_score = _high_hu_cut_discount(
                smoothed_image[current_slice],
                smoothed_image[following_slice],
                edge_p1,
                cfg,
            )
            if cfg.bottleneck_high_hu_cut_weight > 0:
                high_hu_score_sum += float(high_hu_score[inside_pair].sum())
                high_hu_discount_sum += float(high_hu_discount[inside_pair].sum())
                high_hu_edge_count += int(np.count_nonzero(inside_pair))
        edge_p2 = 0.5 * (p2[current_slice] + p2[following_slice])
        edge_p3 = np.maximum(p3[current_slice], p3[following_slice])
        continuity = (
            cfg.bottleneck_edge_floor
            + hu_affinity
            + cfg.bottleneck_label1_continuity * edge_p1
            + cfg.bottleneck_label2_continuity * edge_p2
        )
        continuity *= np.exp(-cfg.bottleneck_label3_discount * edge_p3)
        continuity *= high_hu_discount
        face_area = voxel_volume / float(spacing_zyx[axis])
        capacity = (
            cfg.bottleneck_surface_weight * face_area * continuity
        ).astype(np.float64)
        capacity[~inside_pair] = 0.0

        current_active = active_band[current_slice]
        following_active = active_band[following_slice]
        both_active = inside_pair & current_active & following_active
        if np.any(both_active):
            current_nodes = node_map[current_slice][both_active]
            following_nodes = node_map[following_slice][both_active]
            edge_capacity = capacity[both_active]
            distinct_nodes = current_nodes != following_nodes
            current_nodes = current_nodes[distinct_nodes]
            following_nodes = following_nodes[distinct_nodes]
            edge_capacity = edge_capacity[distinct_nodes]
            graph.add_edges(
                current_nodes,
                following_nodes,
                edge_capacity,
                edge_capacity,
            )
            edge_count += int(edge_capacity.size)
            pairwise_capacity_sum += float(edge_capacity.sum())

        current_to_fixed = inside_pair & current_active & ~following_active
        if np.any(current_to_fixed):
            nodes = node_map[current_slice][current_to_fixed]
            fixed_labels = initial_partition[following_slice][current_to_fixed]
            values = capacity[current_to_fixed]
            source_fixed = fixed_labels == 1
            sink_fixed = fixed_labels == 2
            np.add.at(source_caps, nodes[source_fixed], values[source_fixed])
            np.add.at(sink_caps, nodes[sink_fixed], values[sink_fixed])

        following_to_fixed = inside_pair & following_active & ~current_active
        if np.any(following_to_fixed):
            nodes = node_map[following_slice][following_to_fixed]
            fixed_labels = initial_partition[current_slice][following_to_fixed]
            values = capacity[following_to_fixed]
            source_fixed = fixed_labels == 1
            sink_fixed = fixed_labels == 2
            np.add.at(source_caps, nodes[source_fixed], values[source_fixed])
            np.add.at(sink_caps, nodes[sink_fixed], values[sink_fixed])

    source_nodes = np.unique(node_map[source_seed & active_band])
    sink_nodes = np.unique(node_map[sink_seed & active_band])
    if source_nodes.size == 0 or sink_nodes.size == 0:
        raise RuntimeError("both prompt regions must be present in the bottleneck graph")
    if np.intersect1d(source_nodes, sink_nodes).size:
        return None, {
            "bottleneck_graph_fallback_reason": (
                "a supervoxel contains both prompt regions"
            )
        }
    finite_sum = pairwise_capacity_sum + float(source_caps.sum() + sink_caps.sum())
    hard_capacity = max(1.0, 2.0 * finite_sum + 1.0)
    source_caps[source_nodes] += hard_capacity
    sink_caps[sink_nodes] += hard_capacity
    graph.add_grid_tedges(graph_nodes, source_caps, sink_caps)

    flow = float(graph.maxflow())
    segments = graph.get_grid_segments(graph_nodes)
    source_values = np.unique(segments[source_nodes])
    sink_values = np.unique(segments[sink_nodes])
    if (
        source_values.size != 1
        or sink_values.size != 1
        or bool(source_values[0]) == bool(sink_values[0])
    ):
        return None, {
            "bottleneck_graph_fallback_reason": "hard terminal ownership failed"
        }

    source_value = bool(source_values[0])
    source_mask = (initial_partition == 1) & ~active_band
    sink_mask = (initial_partition == 2) & ~active_band
    active_segments = segments[node_map[active_band]]
    source_mask[active_band] = active_segments == source_value
    sink_mask[active_band] = active_segments != source_value
    cut_mask = _partition_boundary_mask(source_mask, sink_mask, instance_mask)
    result = MinCutResult(source_mask, sink_mask, cut_mask, flow)
    diagnostics = {
        "bottleneck_graph_fallback_reason": None,
        "bottleneck_active_voxels": active_count,
        "bottleneck_graph_nodes": graph_node_count,
        "bottleneck_graph_edges": edge_count,
        "bottleneck_graph_representation": graph_representation,
        "bottleneck_pairwise_capacity_sum": pairwise_capacity_sum,
        "bottleneck_hard_terminal_capacity": hard_capacity,
        "bottleneck_high_hu_cut_applied": bool(
            high_hu_edge_count
            and high_hu_discount_sum < high_hu_edge_count - 1e-6
        ),
        "bottleneck_high_hu_cut_weight": float(
            cfg.bottleneck_high_hu_cut_weight
        ),
        "bottleneck_high_hu_edge_count": high_hu_edge_count,
        "bottleneck_high_hu_mean_score": (
            high_hu_score_sum / high_hu_edge_count
            if high_hu_edge_count
            else 0.0
        ),
        "bottleneck_high_hu_mean_capacity_multiplier": (
            high_hu_discount_sum / high_hu_edge_count
            if high_hu_edge_count
            else 1.0
        ),
    }
    return result, diagnostics


def _run_bottleneck_mincut_stage(
    instance_mask: np.ndarray,
    initial_partition: np.ndarray,
    active_band: np.ndarray,
    source_seed: np.ndarray,
    sink_seed: np.ndarray,
    source_distance: np.ndarray,
    sink_distance: np.ndarray,
    image: Optional[np.ndarray],
    smoothed_image: Optional[np.ndarray],
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    spacing_zyx: np.ndarray,
    cfg: SplitConfig,
) -> tuple[Optional[MinCutResult], dict]:
    """Crop and run one graph-cut stage around an active region."""
    active_band = np.asarray(active_band, dtype=bool) & instance_mask
    active_band |= source_seed | sink_seed
    smoothing_pad = int(
        np.ceil(3.0 * cfg.bottleneck_ct_smoothing_mm / float(np.min(spacing_zyx)))
    )
    graph_crop = _mask_bbox_slices(active_band, pad=max(1, smoothing_pad))
    if smoothed_image is not None:
        graph_smoothed_image = smoothed_image[graph_crop]
    else:
        graph_smoothed_image, _ = _bottleneck_image_features(
            None if image is None else image[graph_crop],
            instance_mask[graph_crop],
            spacing_zyx,
            cfg,
        )
    result, diagnostics = _narrow_band_bottleneck_mincut(
        instance_mask[graph_crop],
        initial_partition[graph_crop],
        active_band[graph_crop],
        source_seed[graph_crop],
        sink_seed[graph_crop],
        source_distance[graph_crop],
        sink_distance[graph_crop],
        graph_smoothed_image,
        p1[graph_crop],
        p2[graph_crop],
        p3[graph_crop],
        spacing_zyx,
        cfg,
    )
    crop_shape = tuple(int(value) for value in instance_mask[graph_crop].shape)
    diagnostics.update(
        {
            "bottleneck_graph_crop_shape": crop_shape,
            "bottleneck_graph_crop_voxels": int(np.prod(crop_shape)),
        }
    )
    if result is None:
        return None, diagnostics

    source_mask = initial_partition == 1
    sink_mask = initial_partition == 2
    source_mask[graph_crop] = result.source_mask
    sink_mask[graph_crop] = result.sink_mask
    return (
        MinCutResult(
            source_mask=source_mask,
            sink_mask=sink_mask,
            cut_mask=_partition_boundary_mask(
                source_mask,
                sink_mask,
                instance_mask,
            ),
            flow=result.flow,
        ),
        diagnostics,
    )


def _cut_contacts_search_boundary(
    cut_mask: np.ndarray,
    distance_to_reference: np.ndarray,
    band_width_mm: float,
    spacing_zyx: np.ndarray,
) -> tuple[bool, int]:
    """Detect a cut constrained by the outer edge of its search band."""
    tolerance_mm = float(np.max(spacing_zyx)) + 1e-6
    boundary_distance = max(0.0, float(band_width_mm) - tolerance_mm)
    contact = cut_mask & (distance_to_reference >= boundary_distance)
    count = int(np.count_nonzero(contact))
    return count > 0, count


def _bottleneck_growth_seeds(
    abbc_pred: np.ndarray,
    instance_mask: np.ndarray,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    cfg: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Expand centre prompts into stable core basins without fixing a cut."""
    if cfg.bottleneck_core_terminal_margin_mm < 0:
        raise ValueError("bottleneck_core_terminal_margin_mm must be non-negative")
    if not cfg.bottleneck_use_core_growth_seeds:
        return (
            source_interaction_mask.copy(),
            sink_interaction_mask.copy(),
            {
                "bottleneck_growth_seed_method": "prompt_only_refinement",
                "bottleneck_core_seed_fallback_reason": None,
                "source_growth_seed_voxels": int(source_interaction_mask.sum()),
                "sink_growth_seed_voxels": int(sink_interaction_mask.sum()),
            },
        )
    try:
        anchors = build_core_anchors(
            abbc_pred,
            instance_mask,
            source_interaction_mask,
            sink_interaction_mask,
            spacing_zyx=spacing_zyx,
            cfg=cfg,
        )
    except (ValueError, RuntimeError) as error:
        return (
            source_interaction_mask.copy(),
            sink_interaction_mask.copy(),
            {
                "bottleneck_growth_seed_method": "prompt_only",
                "bottleneck_core_seed_fallback_reason": str(error),
                "source_growth_seed_voxels": int(source_interaction_mask.sum()),
                "sink_growth_seed_voxels": int(sink_interaction_mask.sum()),
            },
        )

    source_seed = anchors.source_mask.copy()
    sink_seed = anchors.sink_mask.copy()
    method = anchors.diagnostics["core_partition_method"]
    if method == "shared_core_depth_watershed" and cfg.bottleneck_core_terminal_margin_mm > 0:
        interface = _partition_boundary_mask(
            source_seed,
            sink_seed,
            anchors.robust_core_mask,
        )
        if np.any(interface):
            distance_to_interface = distance_transform_edt(
                ~interface,
                sampling=spacing_zyx,
            )
            keep = distance_to_interface > cfg.bottleneck_core_terminal_margin_mm
            source_seed &= keep
            sink_seed &= keep
        method = "shared_core_interior_basins"

    source_seed |= source_interaction_mask
    sink_seed |= sink_interaction_mask
    source_seed[sink_interaction_mask] = False
    sink_seed[source_interaction_mask] = False
    if np.any(source_seed & sink_seed):
        raise RuntimeError("expanded bottleneck growth seeds overlap")
    if not np.any(source_seed) or not np.any(sink_seed):
        raise RuntimeError("expanded bottleneck growth seeds must remain non-empty")

    diagnostics = dict(anchors.diagnostics)
    diagnostics.update(
        {
            "bottleneck_growth_seed_method": method,
            "bottleneck_core_seed_fallback_reason": None,
            "source_growth_seed_voxels": int(source_seed.sum()),
            "sink_growth_seed_voxels": int(sink_seed.sum()),
        }
    )
    return source_seed, sink_seed, diagnostics


def geodesic_bottleneck_partition(
    abbc_pred: np.ndarray,
    instance_mask: np.ndarray,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    softmax: Optional[np.ndarray] = None,
    prob_label_3: Optional[np.ndarray] = None,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    cfg: Optional[SplitConfig] = None,
) -> tuple[MinCutResult, dict]:
    """Split two prompt-centred regions at their weakest connecting surface."""
    if cfg is None:
        cfg = SplitConfig()
    spacing = _validated_spacing_zyx(spacing_zyx)
    instance_mask = np.asarray(instance_mask, dtype=bool)
    source_interaction_mask = np.asarray(source_interaction_mask, dtype=bool)
    sink_interaction_mask = np.asarray(sink_interaction_mask, dtype=bool)
    if abbc_pred.shape != instance_mask.shape:
        raise ValueError("abbc_pred and instance_mask must have the same shape")
    if image is not None and image.shape != instance_mask.shape:
        raise ValueError("image and instance_mask must have the same shape")
    if (
        source_interaction_mask.shape != instance_mask.shape
        or sink_interaction_mask.shape != instance_mask.shape
    ):
        raise ValueError("interaction masks must have the same shape as instance_mask")
    if not np.any(instance_mask):
        raise ValueError("instance_mask must not be empty")
    if not np.any(source_interaction_mask) or not np.any(sink_interaction_mask):
        raise ValueError("both interaction masks must contain at least one voxel")
    if np.any(source_interaction_mask & sink_interaction_mask):
        raise ValueError("interaction masks must not overlap")
    if np.any(source_interaction_mask & ~instance_mask) or np.any(
        sink_interaction_mask & ~instance_mask
    ):
        raise ValueError("interaction masks must be completely inside the selected instance")
    if cfg.bottleneck_band_width_mm < 0:
        raise ValueError("bottleneck_band_width_mm must be non-negative")
    if cfg.bottleneck_coarse_band_width_mm < 0:
        raise ValueError("bottleneck_coarse_band_width_mm must be non-negative")
    if cfg.bottleneck_coarse_max_band_width_mm < 0:
        raise ValueError("bottleneck_coarse_max_band_width_mm must be non-negative")
    if (
        cfg.bottleneck_coarse_max_band_width_mm
        < cfg.bottleneck_coarse_band_width_mm
    ):
        raise ValueError(
            "bottleneck_coarse_max_band_width_mm must be at least "
            "bottleneck_coarse_band_width_mm"
        )
    if cfg.bottleneck_coarse_expansion_factor <= 1.0:
        raise ValueError("bottleneck_coarse_expansion_factor must be above one")
    if cfg.bottleneck_max_graph_nodes < 0:
        raise ValueError("bottleneck_max_graph_nodes must be non-negative")
    if cfg.bottleneck_max_full_feature_voxels < 0:
        raise ValueError("bottleneck_max_full_feature_voxels must be non-negative")
    for name in (
        "bottleneck_high_hu_cut_weight",
        "bottleneck_geodesic_gradient_weight",
        "bottleneck_geodesic_label3_weight",
        "bottleneck_surface_weight",
        "bottleneck_unary_weight",
        "bottleneck_label1_continuity",
        "bottleneck_label2_continuity",
        "bottleneck_label3_discount",
        "bottleneck_edge_floor",
    ):
        value = float(getattr(cfg, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative")
    if (
        not np.isfinite(cfg.bottleneck_high_hu_center_hu)
        or not np.isfinite(cfg.bottleneck_high_hu_scale_hu)
    ):
        raise ValueError("high-HU cut parameters must be finite")
    if cfg.bottleneck_high_hu_scale_hu <= 0:
        raise ValueError("bottleneck_high_hu_scale_hu must be positive")

    p1, p2, p3 = _probability_maps(abbc_pred, softmax, prob_label_3)
    source_growth_seed, sink_growth_seed, growth_diagnostics = (
        _bottleneck_growth_seeds(
            abbc_pred,
            instance_mask,
            source_interaction_mask,
            sink_interaction_mask,
            spacing,
            cfg,
        )
    )
    use_full_ct_features = bool(
        image is not None
        and instance_mask.size <= cfg.bottleneck_max_full_feature_voxels
    )
    if use_full_ct_features:
        smoothed_image, gradient = _bottleneck_image_features(
            image,
            instance_mask,
            spacing,
            cfg,
        )
    else:
        smoothed_image, gradient = None, None
    propagation_cost = np.ones(instance_mask.shape, dtype=np.float32)
    if gradient is not None:
        propagation_cost += cfg.bottleneck_geodesic_gradient_weight * gradient
    propagation_cost += cfg.bottleneck_geodesic_label3_weight * p3
    propagation_cost[~instance_mask] = np.inf
    initial_partition, source_distance, sink_distance, disconnected_diagnostics = (
        _geodesic_seed_ownership(
            instance_mask,
            source_growth_seed,
            sink_growth_seed,
            propagation_cost,
            spacing,
        )
    )
    source_initial = initial_partition == 1
    sink_initial = initial_partition == 2
    if not np.array_equal(source_initial | sink_initial, instance_mask):
        raise RuntimeError("geodesic seed growth did not cover the selected instance")
    initial_cut = _partition_boundary_mask(
        source_initial,
        sink_initial,
        instance_mask,
    )

    graph_diagnostics: dict = {
        "bottleneck_graph_fallback_reason": None,
        "bottleneck_active_voxels": 0,
        "bottleneck_graph_nodes": 0,
        "bottleneck_graph_edges": 0,
        "bottleneck_graph_representation": "none",
        "bottleneck_pairwise_capacity_sum": 0.0,
        "bottleneck_hard_terminal_capacity": 0.0,
        "bottleneck_high_hu_cut_applied": False,
        "bottleneck_high_hu_cut_weight": float(
            cfg.bottleneck_high_hu_cut_weight
        ),
        "bottleneck_high_hu_edge_count": 0,
        "bottleneck_high_hu_mean_score": 0.0,
        "bottleneck_high_hu_mean_capacity_multiplier": 1.0,
        "bottleneck_graph_crop_shape": None,
        "bottleneck_graph_crop_voxels": 0,
        "bottleneck_coarse_to_fine_attempted": False,
        "bottleneck_coarse_to_fine_used": False,
        "bottleneck_coarse_attempts": 0,
        "bottleneck_coarse_expansions": 0,
        "bottleneck_coarse_initial_band_width_mm": 0.0,
        "bottleneck_coarse_band_width_mm": 0.0,
        "bottleneck_coarse_boundary_contact": False,
        "bottleneck_coarse_boundary_contact_voxels": 0,
        "bottleneck_coarse_graph_nodes": 0,
        "bottleneck_coarse_graph_representation": "none",
        "bottleneck_coarse_fallback_reason": None,
        "bottleneck_fine_graph_nodes": 0,
        "bottleneck_fine_graph_representation": "none",
        "bottleneck_fine_fallback_reason": None,
        "bottleneck_refinement_stages": 0,
    }
    result: Optional[MinCutResult] = None
    effective_band_width = 0.0
    source_terminal_voxels = int(source_interaction_mask.sum())
    sink_terminal_voxels = int(sink_interaction_mask.sum())
    if np.any(initial_cut) and cfg.bottleneck_max_graph_nodes != 0:
        distance_to_interface = distance_transform_edt(
            ~initial_cut,
            sampling=spacing,
        )
        fine_width = float(cfg.bottleneck_band_width_mm)
        final_active_band: Optional[np.ndarray] = None
        final_partition = initial_partition

        if cfg.bottleneck_coarse_to_fine:
            graph_diagnostics["bottleneck_coarse_to_fine_attempted"] = True
            coarse_initial_width = max(
                fine_width,
                float(cfg.bottleneck_coarse_band_width_mm),
            )
            coarse_max_width = max(
                coarse_initial_width,
                float(cfg.bottleneck_coarse_max_band_width_mm),
            )
            coarse_width = coarse_initial_width
            coarse_result: Optional[MinCutResult] = None
            coarse_stage_diagnostics: dict = {}
            coarse_contact = False
            coarse_contact_voxels = 0
            coarse_attempts = 0
            coarse_expansions = 0
            coarse_success_width = 0.0
            coarse_success_active_band: Optional[np.ndarray] = None

            while True:
                coarse_attempts += 1
                coarse_active_band = instance_mask & (
                    distance_to_interface <= coarse_width + 1e-6
                )
                coarse_active_band |= (
                    source_interaction_mask | sink_interaction_mask
                )
                candidate, candidate_diagnostics = _run_bottleneck_mincut_stage(
                    instance_mask,
                    initial_partition,
                    coarse_active_band,
                    source_interaction_mask,
                    sink_interaction_mask,
                    source_distance,
                    sink_distance,
                    image,
                    smoothed_image,
                    p1,
                    p2,
                    p3,
                    spacing,
                    cfg,
                )
                if candidate is None:
                    graph_diagnostics["bottleneck_coarse_fallback_reason"] = (
                        candidate_diagnostics.get(
                            "bottleneck_graph_fallback_reason"
                        )
                    )
                    break
                coarse_result = candidate
                coarse_stage_diagnostics = candidate_diagnostics
                coarse_success_width = coarse_width
                coarse_success_active_band = coarse_active_band
                coarse_contact, coarse_contact_voxels = (
                    _cut_contacts_search_boundary(
                        candidate.cut_mask,
                        distance_to_interface,
                        coarse_width,
                        spacing,
                    )
                )
                if not coarse_contact or coarse_width >= coarse_max_width - 1e-6:
                    break
                expanded_width = min(
                    coarse_max_width,
                    max(
                        coarse_width * cfg.bottleneck_coarse_expansion_factor,
                        coarse_width + float(np.min(spacing)),
                    ),
                )
                if expanded_width <= coarse_width + 1e-6:
                    break
                coarse_width = float(expanded_width)
                coarse_expansions += 1

            graph_diagnostics.update(
                {
                    "bottleneck_coarse_attempts": coarse_attempts,
                    "bottleneck_coarse_expansions": coarse_expansions,
                    "bottleneck_coarse_initial_band_width_mm": coarse_initial_width,
                    "bottleneck_coarse_band_width_mm": (
                        coarse_success_width or coarse_width
                    ),
                    "bottleneck_coarse_boundary_contact": coarse_contact,
                    "bottleneck_coarse_boundary_contact_voxels": (
                        coarse_contact_voxels
                    ),
                    "bottleneck_coarse_graph_nodes": int(
                        coarse_stage_diagnostics.get(
                            "bottleneck_graph_nodes",
                            0,
                        )
                    ),
                    "bottleneck_coarse_graph_representation": (
                        coarse_stage_diagnostics.get(
                            "bottleneck_graph_representation",
                            "none",
                        )
                    ),
                }
            )
            if coarse_result is not None:
                result = coarse_result
                graph_diagnostics.update(coarse_stage_diagnostics)
                effective_band_width = coarse_success_width
                final_active_band = coarse_success_active_band
                coarse_partition = np.zeros(instance_mask.shape, dtype=np.uint8)
                coarse_partition[coarse_result.source_mask] = 1
                coarse_partition[coarse_result.sink_mask] = 2
                final_partition = coarse_partition

                if fine_width + 1e-6 < coarse_success_width:
                    distance_to_coarse_cut = distance_transform_edt(
                        ~coarse_result.cut_mask,
                        sampling=spacing,
                    )
                    fine_active_band = instance_mask & (
                        distance_to_coarse_cut <= fine_width + 1e-6
                    )
                    fine_active_band |= (
                        source_interaction_mask | sink_interaction_mask
                    )
                    fine_result, fine_diagnostics = _run_bottleneck_mincut_stage(
                        instance_mask,
                        coarse_partition,
                        fine_active_band,
                        source_interaction_mask,
                        sink_interaction_mask,
                        source_distance,
                        sink_distance,
                        image,
                        smoothed_image,
                        p1,
                        p2,
                        p3,
                        spacing,
                        cfg,
                    )
                    graph_diagnostics.update(
                        {
                            "bottleneck_fine_graph_nodes": int(
                                fine_diagnostics.get("bottleneck_graph_nodes", 0)
                            ),
                            "bottleneck_fine_graph_representation": (
                                fine_diagnostics.get(
                                    "bottleneck_graph_representation",
                                    "none",
                                )
                            ),
                        }
                    )
                    if fine_result is not None:
                        result = fine_result
                        graph_diagnostics.update(fine_diagnostics)
                        graph_diagnostics["bottleneck_coarse_to_fine_used"] = True
                        graph_diagnostics["bottleneck_refinement_stages"] = 2
                        effective_band_width = fine_width
                        final_active_band = fine_active_band
                        final_partition = coarse_partition
                    else:
                        graph_diagnostics["bottleneck_fine_fallback_reason"] = (
                            fine_diagnostics.get(
                                "bottleneck_graph_fallback_reason"
                            )
                        )
                        graph_diagnostics["bottleneck_refinement_stages"] = 1
                else:
                    graph_diagnostics["bottleneck_refinement_stages"] = 1

        if result is None:
            effective_band_width = fine_width
            fine_active_band = instance_mask & (
                distance_to_interface <= fine_width + 1e-6
            )
            fine_active_band |= source_interaction_mask | sink_interaction_mask
            result, fine_diagnostics = _run_bottleneck_mincut_stage(
                instance_mask,
                initial_partition,
                fine_active_band,
                source_interaction_mask,
                sink_interaction_mask,
                source_distance,
                sink_distance,
                image,
                smoothed_image,
                p1,
                p2,
                p3,
                spacing,
                cfg,
            )
            graph_diagnostics.update(fine_diagnostics)
            graph_diagnostics.update(
                {
                    "bottleneck_fine_graph_nodes": int(
                        fine_diagnostics.get("bottleneck_graph_nodes", 0)
                    ),
                    "bottleneck_fine_graph_representation": (
                        fine_diagnostics.get(
                            "bottleneck_graph_representation",
                            "none",
                        )
                    ),
                    "bottleneck_refinement_stages": int(result is not None),
                }
            )
            final_active_band = fine_active_band
            final_partition = initial_partition

        if final_active_band is not None:
            source_terminal_voxels = int(
                np.count_nonzero((final_partition == 1) & ~final_active_band)
                + source_interaction_mask.sum()
            )
            sink_terminal_voxels = int(
                np.count_nonzero((final_partition == 2) & ~final_active_band)
                + sink_interaction_mask.sum()
            )
    elif not np.any(initial_cut):
        graph_diagnostics["bottleneck_graph_fallback_reason"] = (
            "the two grown regions are already disconnected"
        )
    else:
        graph_diagnostics["bottleneck_graph_fallback_reason"] = (
            "bottleneck graph is disabled"
        )

    if result is None:
        result = MinCutResult(source_initial, sink_initial, initial_cut, 0.0)
        partition_method = "geodesic_seed_growth"
    else:
        partition_method = "geodesic_bottleneck_mincut"
    result, topology_diagnostics = _repair_terminal_connectivity(
        result,
        instance_mask,
        source_growth_seed,
        sink_growth_seed,
    )
    if not np.all(result.source_mask[source_interaction_mask]):
        raise RuntimeError("the source interaction was not preserved")
    if not np.all(result.sink_mask[sink_interaction_mask]):
        raise RuntimeError("the sink interaction was not preserved")
    if np.any(result.source_mask & result.sink_mask):
        raise RuntimeError("bottleneck partition outputs overlap")
    if not np.array_equal(result.source_mask | result.sink_mask, instance_mask):
        raise RuntimeError("bottleneck partition does not cover the selected instance")

    diagnostics = dict(disconnected_diagnostics)
    diagnostics.update(growth_diagnostics)
    diagnostics.update(graph_diagnostics)
    diagnostics.update(topology_diagnostics)
    diagnostics.update(
        {
            "mode": "geodesic_bottleneck_partition",
            "peripheral_partition_method": partition_method,
            "soft_fracture_signal_available": bool(
                softmax is not None or prob_label_3 is not None
            ),
            "bottleneck_full_ct_features": use_full_ct_features,
            "bottleneck_geodesic_ct_gradient_used": bool(gradient is not None),
            "bottleneck_band_width_mm": effective_band_width,
            "source_terminal_voxels": source_terminal_voxels,
            "sink_terminal_voxels": sink_terminal_voxels,
            "source_voxels": int(result.source_mask.sum()),
            "sink_voxels": int(result.sink_mask.sum()),
            "cut_voxels": int(result.cut_mask.sum()),
            "maxflow": float(result.flow),
        }
    )
    return result, diagnostics


def _select_core_component(
    labelled_core: np.ndarray,
    interaction_mask: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[int, np.ndarray, float, int]:
    """Select one core component, projecting the interaction when necessary."""
    overlap_labels = labelled_core[interaction_mask]
    overlap_labels = overlap_labels[overlap_labels > 0]
    marker = np.zeros(labelled_core.shape, dtype=bool)
    if overlap_labels.size:
        counts = np.bincount(overlap_labels)
        component_id = int(np.argmax(counts[1:]) + 1)
        marker = interaction_mask & (labelled_core == component_id)
        return component_id, marker, 0.0, int(np.unique(overlap_labels).size)

    core_coordinates = np.argwhere(labelled_core > 0)
    interaction_coordinates = np.argwhere(interaction_mask)
    if core_coordinates.size == 0:
        raise ValueError("the selected instance contains no usable Label-2 core")

    tree = cKDTree(core_coordinates * spacing_zyx)
    distances, nearest_indices = tree.query(interaction_coordinates * spacing_zyx, k=1)
    closest_interaction = int(np.argmin(distances))
    closest_core = core_coordinates[int(nearest_indices[closest_interaction])]
    marker[tuple(closest_core)] = True
    component_id = int(labelled_core[tuple(closest_core)])
    return component_id, marker, float(distances[closest_interaction]), 0


def _project_interaction_to_component(
    labelled_core: np.ndarray,
    component_id: int,
    interaction_mask: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[np.ndarray, float]:
    component_mask = labelled_core == component_id
    marker = interaction_mask & component_mask
    if np.any(marker):
        return marker, 0.0

    component_coordinates = np.argwhere(component_mask)
    interaction_coordinates = np.argwhere(interaction_mask)
    tree = cKDTree(component_coordinates * spacing_zyx)
    distances, nearest_indices = tree.query(interaction_coordinates * spacing_zyx, k=1)
    closest_interaction = int(np.argmin(distances))
    closest_core = component_coordinates[int(nearest_indices[closest_interaction])]
    marker = np.zeros(labelled_core.shape, dtype=bool)
    marker[tuple(closest_core)] = True
    return marker, float(distances[closest_interaction])


def _partition_core_by_depth_watershed(
    component_mask: np.ndarray,
    source_marker: np.ndarray,
    sink_marker: np.ndarray,
    spacing_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split one connected core at its prompt-separated morphological saddle."""
    source_coordinates = np.argwhere(source_marker)
    sink_coordinates = np.argwhere(sink_marker)
    if source_coordinates.size == 0 or sink_coordinates.size == 0:
        raise ValueError("both core markers must contain at least one voxel")

    source_center = np.mean(source_coordinates * spacing_zyx, axis=0)
    sink_center = np.mean(sink_coordinates * spacing_zyx, axis=0)
    direction = sink_center - source_center
    separation_mm = float(np.linalg.norm(direction))
    if not np.isfinite(separation_mm) or separation_mm <= 1e-6:
        raise ValueError("the two interactions project to the same core position")

    crop = _mask_bbox_slices(component_mask, pad=1)
    component_local = component_mask[crop]
    source_local = source_marker[crop]
    sink_local = sink_marker[crop]
    markers = np.zeros(component_local.shape, dtype=np.uint8)
    markers[source_local] = 1
    markers[sink_local] = 2
    core_depth = distance_transform_edt(component_local, sampling=spacing_zyx)
    partition = watershed(
        -core_depth,
        markers=markers,
        mask=component_local,
        connectivity=1,
        watershed_line=False,
    )
    source_anchor = np.zeros(component_mask.shape, dtype=bool)
    sink_anchor = np.zeros(component_mask.shape, dtype=bool)
    source_anchor[crop] = partition == 1
    sink_anchor[crop] = partition == 2

    # Marker ownership is hard even when two prompts lie near a saddle.
    source_anchor[source_marker] = True
    sink_anchor[source_marker] = False
    sink_anchor[sink_marker] = True
    source_anchor[sink_marker] = False
    if not np.array_equal(source_anchor | sink_anchor, component_mask):
        raise RuntimeError("depth watershed did not cover the selected core component")
    return source_anchor, sink_anchor, separation_mm


def _anchors_from_core_mask(
    core_mask: np.ndarray,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    min_anchor_voxels: int,
    distinct_max_distance_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    unfiltered_labels, unfiltered_component_count = nd_label(core_mask)
    if unfiltered_component_count == 0:
        raise ValueError("the selected instance contains no usable Label-2 core")
    component_sizes = np.bincount(unfiltered_labels.ravel())
    usable_lookup = component_sizes >= min_anchor_voxels
    usable_lookup[0] = False
    usable_core = usable_lookup[unfiltered_labels]
    labelled_core, component_count = nd_label(usable_core)
    if component_count == 0:
        raise ValueError(
            "the selected instance contains no core component above core_anchor_min_voxels"
        )

    source_component, source_marker, source_distance, source_overlaps = _select_core_component(
        labelled_core,
        source_interaction_mask,
        spacing_zyx,
    )
    sink_component, sink_marker, sink_distance, sink_overlaps = _select_core_component(
        labelled_core,
        sink_interaction_mask,
        spacing_zyx,
    )

    distinct_reassignment = None
    if source_component == sink_component and distinct_max_distance_mm > 0:
        component_sizes = np.bincount(labelled_core.ravel())
        alternative_components = [
            component_id
            for component_id in range(1, component_count + 1)
            if component_id != source_component
            and int(component_sizes[component_id]) >= min_anchor_voxels
        ]
        alternatives: list[tuple[float, str, int, np.ndarray]] = []
        for component_id in alternative_components:
            marker, distance = _project_interaction_to_component(
                labelled_core,
                component_id,
                source_interaction_mask,
                spacing_zyx,
            )
            alternatives.append((distance, "source", component_id, marker))
            marker, distance = _project_interaction_to_component(
                labelled_core,
                component_id,
                sink_interaction_mask,
                spacing_zyx,
            )
            alternatives.append((distance, "sink", component_id, marker))

        if alternatives:
            distance, side, component_id, marker = min(alternatives, key=lambda item: item[0])
            if distance <= distinct_max_distance_mm:
                if side == "source":
                    source_component = component_id
                    source_marker = marker
                    source_distance = distance
                    source_overlaps = 0
                else:
                    sink_component = component_id
                    sink_marker = marker
                    sink_distance = distance
                    sink_overlaps = 0
                distinct_reassignment = side

    prompt_separation_mm = None
    if source_component != sink_component:
        source_anchor = labelled_core == source_component
        sink_anchor = labelled_core == sink_component
        method = "separate_core_components"
    else:
        selected_component = labelled_core == source_component
        if np.any(source_marker & sink_marker):
            raise ValueError("both interactions project to the same core voxel")

        source_anchor, sink_anchor, prompt_separation_mm = (
            _partition_core_by_depth_watershed(
                selected_component,
                source_marker,
                sink_marker,
                spacing_zyx,
            )
        )
        method = "shared_core_depth_watershed"

    source_size = int(source_anchor.sum())
    sink_size = int(sink_anchor.sum())
    if min(source_size, sink_size) < min_anchor_voxels:
        raise ValueError(
            "core partition produced an anchor below core_anchor_min_voxels: "
            f"source={source_size}, sink={sink_size}, minimum={min_anchor_voxels}"
        )

    diagnostics = {
        "core_component_count": int(component_count),
        "unfiltered_core_component_count": int(unfiltered_component_count),
        "ignored_small_core_components": int(unfiltered_component_count - component_count),
        "core_partition_method": method,
        "source_core_component": source_component,
        "sink_core_component": sink_component,
        "source_core_anchor_voxels": source_size,
        "sink_core_anchor_voxels": sink_size,
        "source_projection_distance_mm": source_distance,
        "sink_projection_distance_mm": sink_distance,
        "source_intersected_core_components": source_overlaps,
        "sink_intersected_core_components": sink_overlaps,
        "distinct_component_reassignment": distinct_reassignment,
        "shared_core_prompt_separation_mm": prompt_separation_mm,
    }
    return source_anchor, sink_anchor, diagnostics


def build_core_anchors(
    abbc_pred: np.ndarray,
    instance_mask: np.ndarray,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    cfg: Optional[SplitConfig] = None,
) -> CoreAnchorResult:
    """Convert two physician interactions into robust Label-2 core terminals.

    Separate eroded core components become terminals directly. When both
    interactions select the same component, a spacing-aware depth watershed
    partitions that core at its morphological saddle. The resulting regions,
    rather than the small interaction masks, guide the full instance split.
    """
    if cfg is None:
        cfg = SplitConfig()
    if abbc_pred.shape != instance_mask.shape:
        raise ValueError("abbc_pred and instance_mask must have the same shape")
    if source_interaction_mask.shape != instance_mask.shape or sink_interaction_mask.shape != instance_mask.shape:
        raise ValueError("interaction masks must have the same shape as instance_mask")
    if cfg.core_anchor_erosion_mm < 0:
        raise ValueError("core_anchor_erosion_mm must be non-negative")
    if cfg.core_anchor_min_voxels < 1:
        raise ValueError("core_anchor_min_voxels must be positive")
    if cfg.core_anchor_distinct_max_distance_mm < 0:
        raise ValueError("core_anchor_distinct_max_distance_mm must be non-negative")

    spacing = _validated_spacing_zyx(spacing_zyx)
    instance_mask = np.asarray(instance_mask, dtype=bool)
    source_interaction_mask = np.asarray(source_interaction_mask, dtype=bool)
    sink_interaction_mask = np.asarray(sink_interaction_mask, dtype=bool)
    if not np.any(instance_mask):
        raise ValueError("instance_mask must not be empty")
    if not np.any(source_interaction_mask) or not np.any(sink_interaction_mask):
        raise ValueError("both interaction masks must contain at least one voxel")
    if np.any(source_interaction_mask & sink_interaction_mask):
        raise ValueError("interaction masks must not overlap")
    if np.any(source_interaction_mask & ~instance_mask) or np.any(sink_interaction_mask & ~instance_mask):
        raise ValueError("interaction masks must be completely inside the selected instance")

    raw_core = (abbc_pred == 2) & instance_mask
    if not np.any(raw_core):
        raise ValueError("the selected instance contains no Label-2 core")

    if cfg.core_anchor_erosion_mm > 0:
        core_depth = distance_transform_edt(raw_core, sampling=spacing)
        robust_core = raw_core & (core_depth > cfg.core_anchor_erosion_mm)
    else:
        robust_core = raw_core.copy()

    fallback_reason = None
    try:
        source_anchor, sink_anchor, diagnostics = _anchors_from_core_mask(
            robust_core,
            source_interaction_mask,
            sink_interaction_mask,
            spacing,
            cfg.core_anchor_min_voxels,
            cfg.core_anchor_distinct_max_distance_mm,
        )
        core_source = "eroded_label_2" if cfg.core_anchor_erosion_mm > 0 else "label_2"
    except (ValueError, RuntimeError) as error:
        if np.array_equal(robust_core, raw_core):
            raise
        fallback_reason = str(error)
        robust_core = raw_core.copy()
        source_anchor, sink_anchor, diagnostics = _anchors_from_core_mask(
            robust_core,
            source_interaction_mask,
            sink_interaction_mask,
            spacing,
            cfg.core_anchor_min_voxels,
            cfg.core_anchor_distinct_max_distance_mm,
        )
        core_source = "label_2_fallback"

    diagnostics.update(
        {
            "core_source": core_source,
            "core_anchor_erosion_mm": float(cfg.core_anchor_erosion_mm),
            "raw_core_voxels": int(raw_core.sum()),
            "robust_core_voxels": int(robust_core.sum()),
            "erosion_fallback_reason": fallback_reason,
        }
    )
    return CoreAnchorResult(
        source_mask=source_anchor,
        sink_mask=sink_anchor,
        raw_core_mask=raw_core,
        robust_core_mask=robust_core,
        diagnostics=diagnostics,
    )


def core_first_partition(
    abbc_pred: np.ndarray,
    instance_mask: np.ndarray,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    softmax: Optional[np.ndarray] = None,
    prob_label_3: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    cfg: Optional[SplitConfig] = None,
) -> tuple[MinCutResult, dict]:
    """Partition one instance with geodesic bottleneck or legacy core ownership."""
    if cfg is None:
        cfg = SplitConfig()
    if cfg.use_geodesic_bottleneck:
        return geodesic_bottleneck_partition(
            abbc_pred=abbc_pred,
            instance_mask=instance_mask,
            source_interaction_mask=source_interaction_mask,
            sink_interaction_mask=sink_interaction_mask,
            image=image,
            softmax=softmax,
            prob_label_3=prob_label_3,
            spacing_zyx=spacing_zyx,
            cfg=cfg,
        )
    anchors = build_core_anchors(
        abbc_pred,
        instance_mask,
        source_interaction_mask,
        sink_interaction_mask,
        spacing_zyx=spacing_zyx,
        cfg=cfg,
    )
    source_terminal = anchors.source_mask | source_interaction_mask
    sink_terminal = anchors.sink_mask | sink_interaction_mask
    if np.any(source_terminal & sink_terminal):
        raise ValueError(
            "the click-to-core assignments conflict; place the two points farther "
            "inside their intended fragments"
        )
    if cfg.core_first_max_graph_voxels < 0:
        raise ValueError("core_first_max_graph_voxels must be non-negative")
    has_soft_fracture_signal = softmax is not None or prob_label_3 is not None
    use_graph = (
        has_soft_fracture_signal
        and cfg.core_first_max_graph_voxels > 0
        and instance_mask.size <= cfg.core_first_max_graph_voxels
    )
    if use_graph:
        source_seed = Seed(
            mask=source_terminal,
            centroid=np.argwhere(source_terminal).mean(axis=0),
            source="core_anchor",
        )
        sink_seed = Seed(
            mask=sink_terminal,
            centroid=np.argwhere(sink_terminal).mean(axis=0),
            source="core_anchor",
        )
        cost = build_cost_field(
            abbc_pred,
            softmax,
            instance_mask,
            cfg,
            prob_label_3=prob_label_3,
        )
        result = min_cut_partition(instance_mask, source_seed, sink_seed, cost)
        if result is None:
            raise RuntimeError("core-anchored min-cut failed")
        peripheral_method = "min_cut"
        disconnected_diagnostics = {
            "unseeded_component_count": 0,
            "unseeded_components_assigned_source": 0,
            "unseeded_components_assigned_sink": 0,
            "unseeded_voxels_assigned_source": 0,
            "unseeded_voxels_assigned_sink": 0,
        }
    else:
        _, _, p3 = _probability_maps(abbc_pred, softmax, prob_label_3)
        markers = np.zeros(instance_mask.shape, dtype=np.uint8)
        markers[source_terminal] = 1
        markers[sink_terminal] = 2
        partition = watershed(
            p3,
            markers=markers,
            mask=instance_mask,
            connectivity=1,
            watershed_line=False,
        )
        disconnected_diagnostics = _assign_unseeded_instance_components(
            partition,
            instance_mask,
            source_terminal,
            sink_terminal,
            spacing_zyx,
        )
        source_mask = partition == 1
        sink_mask = partition == 2
        if not np.array_equal(source_mask | sink_mask, instance_mask):
            raise RuntimeError("core-seeded watershed did not cover the selected instance")
        cut_mask = np.zeros(instance_mask.shape, dtype=bool)
        for axis in range(3):
            current = [slice(None)] * 3
            following = [slice(None)] * 3
            current[axis] = slice(0, -1)
            following[axis] = slice(1, None)
            current_slice = tuple(current)
            following_slice = tuple(following)
            boundary = (
                (partition[current_slice] != partition[following_slice])
                & instance_mask[current_slice]
                & instance_mask[following_slice]
            )
            cut_mask[current_slice] |= boundary
            cut_mask[following_slice] |= boundary
        result = MinCutResult(source_mask, sink_mask, cut_mask, 0.0)
        peripheral_method = "marker_watershed"
    if not np.all(result.source_mask[anchors.source_mask]):
        raise RuntimeError("source core ownership was not preserved")
    if not np.all(result.sink_mask[anchors.sink_mask]):
        raise RuntimeError("sink core ownership was not preserved")
    if not np.all(result.source_mask[source_interaction_mask]):
        raise RuntimeError("the source interaction was not preserved")
    if not np.all(result.sink_mask[sink_interaction_mask]):
        raise RuntimeError("the sink interaction was not preserved")
    if not np.array_equal(result.source_mask | result.sink_mask, instance_mask):
        raise RuntimeError("core-first partition does not cover the selected instance")

    diagnostics = dict(anchors.diagnostics)
    diagnostics.update(disconnected_diagnostics)
    diagnostics.update(
        {
            "mode": "core_first_partition",
            "peripheral_partition_method": peripheral_method,
            "soft_fracture_signal_available": bool(has_soft_fracture_signal),
            "graph_bbox_voxels": int(instance_mask.size),
            "source_terminal_voxels": int(source_terminal.sum()),
            "sink_terminal_voxels": int(sink_terminal.sum()),
            "source_voxels": int(result.source_mask.sum()),
            "sink_voxels": int(result.sink_mask.sum()),
            "cut_voxels": int(result.cut_mask.sum()),
            "cut_raw_core_voxels": int(np.count_nonzero(result.cut_mask & anchors.raw_core_mask)),
            "maxflow": float(result.flow),
        }
    )
    return result, diagnostics


def split_instance_with_seeds(
    abbc_pred: np.ndarray,
    instances: np.ndarray,
    instance_id: int,
    source_seed_mask: np.ndarray,
    sink_seed_mask: np.ndarray,
    softmax: Optional[np.ndarray] = None,
    prob_label_3: Optional[np.ndarray] = None,
    cfg: Optional[SplitConfig] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split one selected instance using two physician-provided seed masks.

    The source side keeps ``instance_id`` and the sink side receives the next
    free positive ID. Seed masks must be non-empty, disjoint, and completely
    contained in the selected instance.
    """
    if cfg is None:
        cfg = SplitConfig()
    if abbc_pred.shape != instances.shape:
        raise ValueError(
            f"abbc_pred shape {abbc_pred.shape} != instances shape {instances.shape}"
        )
    if softmax is not None and (
        softmax.ndim != 4 or softmax.shape[0] < 4 or softmax.shape[1:] != instances.shape
    ):
        raise ValueError("softmax must have shape (>=4, *instances.shape)")
    if prob_label_3 is not None and prob_label_3.shape != instances.shape:
        raise ValueError("prob_label_3 must have the same shape as instances")
    if source_seed_mask.shape != instances.shape or sink_seed_mask.shape != instances.shape:
        raise ValueError("manual seed masks must have the same shape as instances")
    if int(instance_id) <= 0:
        raise ValueError("instance_id must be a positive integer")

    instance_mask = instances == int(instance_id)
    if not np.any(instance_mask):
        raise ValueError(f"instance_id {instance_id} is not present")

    source_seed_mask = np.asarray(source_seed_mask, dtype=bool)
    sink_seed_mask = np.asarray(sink_seed_mask, dtype=bool)
    if not np.any(source_seed_mask) or not np.any(sink_seed_mask):
        raise ValueError("both manual seed masks must contain at least one voxel")
    if np.any(source_seed_mask & sink_seed_mask):
        raise ValueError("manual seed masks must not overlap")
    if np.any(source_seed_mask & ~instance_mask) or np.any(sink_seed_mask & ~instance_mask):
        raise ValueError("manual seed masks must be completely inside the selected instance")

    # All graph and probability work is local to the selected instance. A raw
    # Charite volume can contain hundreds of millions of voxels while the
    # fracture ROI is typically below one million.
    crop = _mask_bbox_slices(instance_mask)
    local_abbc = abbc_pred[crop]
    local_instance = instance_mask[crop]
    local_source = source_seed_mask[crop]
    local_sink = sink_seed_mask[crop]
    local_softmax = None if softmax is None else softmax[(slice(None),) + crop]
    local_prob3 = None if prob_label_3 is None else prob_label_3[crop]

    source_seed = Seed(
        mask=local_source,
        centroid=np.argwhere(local_source).mean(axis=0),
        source="manual",
    )
    sink_seed = Seed(
        mask=local_sink,
        centroid=np.argwhere(local_sink).mean(axis=0),
        source="manual",
    )
    cost = build_cost_field(
        local_abbc,
        local_softmax,
        local_instance,
        cfg,
        prob_label_3=local_prob3,
    )
    result = min_cut_partition(local_instance, source_seed, sink_seed, cost)
    if result is None:
        raise RuntimeError(
            "min-cut failed; ensure PyMaxflow is installed and the two seeds are separable"
        )

    source_size = int(result.source_mask.sum())
    sink_size = int(result.sink_mask.sum())
    if source_size < cfg.min_split_piece_size or sink_size < cfg.min_split_piece_size:
        raise RuntimeError(
            "min-cut produced a piece below min_split_piece_size: "
            f"source={source_size}, sink={sink_size}, minimum={cfg.min_split_piece_size}"
        )
    if not np.array_equal(result.source_mask | result.sink_mask, local_instance):
        raise RuntimeError("min-cut partition does not cover the selected instance")

    next_id = int(max(0, int(instances.max()))) + 1
    instances_out = instances.copy()
    if np.issubdtype(instances_out.dtype, np.integer):
        max_value = np.iinfo(instances_out.dtype).max
        if next_id > max_value:
            instances_out = instances_out.astype(np.uint32)
    local_out = instances_out[crop]
    local_out[result.source_mask] = int(instance_id)
    local_out[result.sink_mask] = next_id

    _, _, p3 = _probability_maps(local_abbc, local_softmax, local_prob3)
    cut_prob = p3[result.cut_mask]
    instance_prob = p3[local_instance]
    cut_mask = np.zeros(instances.shape, dtype=bool)
    cut_mask[crop] = result.cut_mask
    diagnostics = {
        "mode": "manual_two_seed",
        "instance": int(instance_id),
        "new_instance": next_id,
        "source_voxels": source_size,
        "sink_voxels": sink_size,
        "cut_voxels": int(result.cut_mask.sum()),
        "crop_shape": [int(v) for v in local_instance.shape],
        "maxflow": float(result.flow),
        "mean_prob_label_3_on_cut": float(cut_prob.mean()) if cut_prob.size else 0.0,
        "mean_prob_label_3_in_instance": float(instance_prob.mean()),
        "fraction_cut_above_prob3_seed_threshold": (
            float(np.mean(cut_prob > cfg.prob3_seed_threshold)) if cut_prob.size else 0.0
        ),
    }
    return instances_out, cut_mask, diagnostics


def split_instance_core_first(
    abbc_pred: np.ndarray,
    instances: np.ndarray,
    instance_id: int,
    source_interaction_mask: np.ndarray,
    sink_interaction_mask: np.ndarray,
    softmax: Optional[np.ndarray] = None,
    prob_label_3: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    cfg: Optional[SplitConfig] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split one prompted instance while preserving its exact input support."""
    if cfg is None:
        cfg = SplitConfig()
    if abbc_pred.shape != instances.shape:
        raise ValueError(
            f"abbc_pred shape {abbc_pred.shape} != instances shape {instances.shape}"
        )
    if softmax is not None and (
        softmax.ndim != 4 or softmax.shape[0] < 4 or softmax.shape[1:] != instances.shape
    ):
        raise ValueError("softmax must have shape (>=4, *instances.shape)")
    if prob_label_3 is not None and prob_label_3.shape != instances.shape:
        raise ValueError("prob_label_3 must have the same shape as instances")
    if image is not None and image.shape != instances.shape:
        raise ValueError("image must have the same shape as instances")
    if source_interaction_mask.shape != instances.shape or sink_interaction_mask.shape != instances.shape:
        raise ValueError("interaction masks must have the same shape as instances")
    if int(instance_id) <= 0:
        raise ValueError("instance_id must be a positive integer")

    instance_mask = instances == int(instance_id)
    if not np.any(instance_mask):
        raise ValueError(f"instance_id {instance_id} is not present")
    source_interaction_mask = np.asarray(source_interaction_mask, dtype=bool)
    sink_interaction_mask = np.asarray(sink_interaction_mask, dtype=bool)
    if not np.any(source_interaction_mask) or not np.any(sink_interaction_mask):
        raise ValueError("both interaction masks must contain at least one voxel")
    if np.any(source_interaction_mask & sink_interaction_mask):
        raise ValueError("interaction masks must not overlap")
    if np.any(source_interaction_mask & ~instance_mask) or np.any(sink_interaction_mask & ~instance_mask):
        raise ValueError("interaction masks must be completely inside the selected instance")

    crop = _mask_bbox_slices(instance_mask)
    local_abbc = abbc_pred[crop]
    local_instance = instance_mask[crop]
    local_source = source_interaction_mask[crop]
    local_sink = sink_interaction_mask[crop]
    local_softmax = None if softmax is None else softmax[(slice(None),) + crop]
    local_prob3 = None if prob_label_3 is None else prob_label_3[crop]
    local_image = None if image is None else image[crop]
    result, diagnostics = core_first_partition(
        abbc_pred=local_abbc,
        instance_mask=local_instance,
        source_interaction_mask=local_source,
        sink_interaction_mask=local_sink,
        softmax=local_softmax,
        prob_label_3=local_prob3,
        image=local_image,
        spacing_zyx=spacing_zyx,
        cfg=cfg,
    )

    source_size = int(result.source_mask.sum())
    sink_size = int(result.sink_mask.sum())
    if min(source_size, sink_size) < cfg.min_split_piece_size:
        raise RuntimeError(
            "core-first split produced a piece below min_split_piece_size: "
            f"source={source_size}, sink={sink_size}, minimum={cfg.min_split_piece_size}"
        )

    next_id = int(max(0, int(instances.max()))) + 1
    instances_out = instances.copy()
    if np.issubdtype(instances_out.dtype, np.integer):
        max_value = np.iinfo(instances_out.dtype).max
        if next_id > max_value:
            instances_out = instances_out.astype(np.uint32)
    local_out = instances_out[crop]
    local_out[result.source_mask] = int(instance_id)
    local_out[result.sink_mask] = next_id

    _, _, p3 = _probability_maps(local_abbc, local_softmax, local_prob3)
    cut_prob = p3[result.cut_mask]
    instance_prob = p3[local_instance]
    cut_mask = np.zeros(instances.shape, dtype=bool)
    cut_mask[crop] = result.cut_mask
    diagnostics.update(
        {
            "mode": "core_first_two_seed",
            "instance": int(instance_id),
            "new_instance": next_id,
            "crop_shape": [int(v) for v in local_instance.shape],
            "mean_prob_label_3_on_cut": float(cut_prob.mean()) if cut_prob.size else 0.0,
            "mean_prob_label_3_in_instance": float(instance_prob.mean()),
            "fraction_cut_above_prob3_seed_threshold": (
                float(np.mean(cut_prob > cfg.prob3_seed_threshold)) if cut_prob.size else 0.0
            ),
        }
    )
    return instances_out, cut_mask, diagnostics


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------
def cortical_anchored_split(
    abbc_pred: np.ndarray,
    instances: np.ndarray,
    softmax: Optional[np.ndarray] = None,
    cfg: Optional[SplitConfig] = None,
    prob_label_3: Optional[np.ndarray] = None,
    use_prob3_seeds: bool = True,
    instance_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the v2 split on every initial instance.

    Parameters
    ----------
    abbc_pred : (D,H,W) int  — argmax of network (0/1/2/3)
    instances : (D,H,W) int  — current abbc2instance output
    softmax   : (4,D,H,W) float | None — full softmax. Strongly recommended
                for Case 2/3; Case 1 can work without it. None → fall back
                to argmax-derived hard probabilities.
    prob_label_3 : (D,H,W) float | None — standalone label-3 probability.
                   This overrides channel 3 of ``softmax`` when supplied.
    instance_ids : optional IDs to process; None processes all initial IDs.
    cfg       : SplitConfig | None
    """
    if cfg is None:
        cfg = SplitConfig()
    if abbc_pred.shape != instances.shape:
        raise ValueError(
            f"abbc_pred shape {abbc_pred.shape} != instances shape {instances.shape}"
        )

    instances_out = instances.copy()
    label3_added = np.zeros(abbc_pred.shape, dtype=bool)
    next_id = int(instances.max()) + 1
    diag: dict = {"per_instance": []}

    _, _, prob3 = _probability_maps(abbc_pred, softmax, prob_label_3)
    selected_ids = None if instance_ids is None else {int(i) for i in instance_ids}

    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        if selected_ids is not None and int(inst_id) not in selected_ids:
            continue
        inst_mask = instances == inst_id
        if int(inst_mask.sum()) < cfg.min_instance_voxels:
            continue
        cortical_mask = inst_mask & (abbc_pred == 1)
        if int(cortical_mask.sum()) < cfg.min_cortical_voxels:
            continue

        # Gather seeds from all sources
        seeds_cortical = cortical_truncation_seeds(cortical_mask, inst_mask, cfg)
        seeds_prob3 = prob3_peak_seeds(prob3, inst_mask, cfg) if use_prob3_seeds else []
        seeds_cores = isolated_core_seeds(abbc_pred, inst_mask, cfg) if cfg.use_isolated_cores else []
        all_seeds = seeds_cortical + seeds_prob3 + seeds_cores

        diag_entry = {
            "instance": int(inst_id),
            "cortical_voxels": int(cortical_mask.sum()),
            "seeds_cortical": len(seeds_cortical),
            "seeds_prob3": len(seeds_prob3),
            "seeds_cores": len(seeds_cores),
            "pairs": [],
            "splits": 0,
        }
        if len(all_seeds) < 2:
            diag["per_instance"].append(diag_entry)
            continue

        # Hard upper bound on splits for this instance: a merged blob that
        # contains K disconnected cores is at most K real fragments, so at
        # most K-1 cuts are needed. Cap to avoid over-splitting cascades when
        # cortical seeds keep firing meaningless cuts after the cores are
        # already each in their own piece.
        cores_in_blob = (abbc_pred == 2) & inst_mask
        _, n_cores_in_blob = nd_label(cores_in_blob)
        max_splits_for_this_instance = max(0, n_cores_in_blob - 1)
        if max_splits_for_this_instance == 0:
            # only 1 (or 0) core in the whole blob → no honest split is
            # possible without a stronger (e.g. softmax-driven) signal
            diag_entry["max_splits_cap"] = 0
            diag["per_instance"].append(diag_entry)
            continue
        diag_entry["max_splits_cap"] = max_splits_for_this_instance

        pairs = pair_seeds(all_seeds, inst_mask, cfg)
        cost = build_cost_field(
            abbc_pred,
            softmax,
            inst_mask,
            cfg,
            prob_label_3=prob_label_3,
        )

        for p in pairs:
            if diag_entry["splits"] >= max_splits_for_this_instance:
                break
            result = min_cut_partition(inst_mask, p.src, p.snk, cost)
            if result is None or not np.any(result.cut_mask):
                diag_entry["pairs"].append(
                    {"case": p.case, "score": p.score, "result": "no_cut"}
                )
                continue

            source_size = int(result.source_mask.sum())
            sink_size = int(result.sink_mask.sum())
            if min(source_size, sink_size) < cfg.min_split_piece_size:
                diag_entry["pairs"].append(
                    {
                        "case": p.case,
                        "score": p.score,
                        "result": "piece_too_small",
                        "source_voxels": source_size,
                        "sink_voxels": sink_size,
                    }
                )
                continue

            # Reject pieces that don't contain a core voxel — they are
            # geometric chips, not real fragments. This avoids over-splitting
            # when N>=3 cuts cascade and a min-cut nicks off a thin sliver.
            cores_global = abbc_pred == 2
            if source_size >= sink_size:
                keep_mask, new_mask = result.source_mask, result.sink_mask
            else:
                keep_mask, new_mask = result.sink_mask, result.source_mask
            if not np.any(new_mask & cores_global):
                # cut disconnected only chip-off slivers without cores → revert
                diag_entry["pairs"].append(
                    {"case": p.case, "score": p.score, "result": "all_chips_no_cores"}
                )
                continue

            # Assign the entire binary partition. Unlike the old validation
            # approach, cut-boundary voxels do not remain as an ID bridge.
            instances_out[keep_mask] = int(inst_id)
            instances_out[new_mask] = next_id
            next_id += 1
            label3_added |= result.cut_mask
            diag_entry["splits"] += 1
            diag_entry["pairs"].append(
                {"case": p.case, "score": float(p.score),
                 "result": "split_into_2",
                 "source_voxels": source_size,
                 "sink_voxels": sink_size,
                 "cut_voxels": int(result.cut_mask.sum()),
                 "maxflow": float(result.flow)}
            )
            inst_mask = instances_out == inst_id  # refresh for the next pair

        diag["per_instance"].append(diag_entry)

    return instances_out, label3_added, diag
