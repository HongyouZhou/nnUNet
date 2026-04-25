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
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label as nd_label,
)
from skimage.morphology import ball, skeletonize


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
    source: str                 # 'cortical_truncation' | 'prob3_peak' | 'isolated_core'
    normal: Optional[np.ndarray] = None   # cortical seeds carry an outward normal
    score: float = 0.0


@dataclasses.dataclass
class SeedPair:
    src: Seed
    snk: Seed
    case: int                   # 1, 2, or 3 — best-guess case label
    distance: float
    score: float                # higher = more likely a true fracture pair


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
def min_cut_surface(
    instance_mask: np.ndarray,
    src_seed: Seed,
    snk_seed: Seed,
    cost_field: np.ndarray,
    bbox_pad: int = 4,
) -> Optional[np.ndarray]:
    """3D min-cut between src/snk seeds, returning the 2D cut surface.

    The cut surface is the set of voxels whose label flips from "source side"
    to "sink side" along any 6-connected edge. We mark these voxels in a
    boolean mask (full-volume).

    Returns None if PyMaxflow isn't installed or the seeds aren't separable.
    """
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

    # Build graph. Use float32 capacities. PyMaxflow's add_grid_edges
    # supports a per-voxel capacity for axial edges directly.
    g = maxflow.Graph[float]()
    node_ids = g.add_grid_nodes(inside.shape)

    # Pairwise edges along ±z, ±y, ±x with capacity = avg cost between voxels
    structure_axes = [
        np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),  # +x
        np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),  # +y? (the 2nd entry encodes +y)
        np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),  # +z direction depends; we just use 6 axial neighbours via add_grid_edges with 3 calls below
    ]
    # Simpler: call add_grid_edges three times, one per axis,
    # using a per-voxel capacity array of avg cost between neighbours.
    big_cap = float(cost.max() + 1.0) * 10.0
    cost_inside = np.where(inside, cost, big_cap)
    for axis in range(3):
        struct = np.zeros((3, 3, 3), dtype=int)
        center = (1, 1, 1)
        nbr = [1, 1, 1]
        nbr[axis] = 2
        struct[tuple(nbr)] = 1
        # capacity per voxel = avg cost with axial neighbour
        c_self = cost_inside
        c_next = np.roll(cost_inside, shift=-1, axis=axis)
        cap = 0.5 * (c_self + c_next)
        g.add_grid_edges(node_ids, weights=cap, structure=struct, symmetric=True)

    # Terminal edges: src ↔ source with infinite cap, snk ↔ sink with infinite cap.
    # Outside instance: tied to sink with infinite cap (no node should be assigned to source).
    INF = big_cap * 10.0
    src_caps = np.zeros(inside.shape, dtype=np.float32)
    snk_caps = np.zeros(inside.shape, dtype=np.float32)
    src_caps[src_local] = INF
    snk_caps[snk_local] = INF
    snk_caps[~inside] = INF       # outside-of-instance tied to sink
    g.add_grid_tedges(node_ids, src_caps, snk_caps)

    g.maxflow()
    sgm = g.get_grid_segments(node_ids)   # True = sink-side, False = source-side

    # Cut voxels = those whose 6-neighbour has different segment label, AND inside instance.
    # We mark BOTH sides of the cut to get a 2-vx-thick surface that reliably disconnects.
    cut_mask = np.zeros(inside.shape, dtype=bool)
    for axis in range(3):
        diff = np.diff(sgm.astype(np.int8), axis=axis) != 0
        sl_self = [slice(None)] * 3
        sl_next = [slice(None)] * 3
        sl_self[axis] = slice(0, -1)
        sl_next[axis] = slice(1, None)
        cut_mask[tuple(sl_self)] |= diff
        cut_mask[tuple(sl_next)] |= diff
    cut_mask &= inside

    full = np.zeros(instance_mask.shape, dtype=bool)
    full[sl] = cut_mask
    return full


# ----------------------------------------------------------------------------
# Cost field
# ----------------------------------------------------------------------------
def build_cost_field(
    abbc_pred: np.ndarray,
    softmax: Optional[np.ndarray],
    instance_mask: np.ndarray,
    cfg: SplitConfig,
) -> np.ndarray:
    """Voxel cost: low where we'd like the cut to go, high where it shouldn't.

    cost = w_min
         + w_prob1 * prob_label_1     (cortical: high → cut should avoid)
         + w_prob2 * prob_label_2     (core: medium repel)
         - w_prob3 * prob_label_3     (border: low cost — cut prefers)
    Outside the instance: w_bg_outside (effectively infinite).
    """
    if softmax is not None:
        p1 = softmax[1]
        p2 = softmax[2]
        p3 = softmax[3]
    else:
        # fall back to argmax — softer signals lost, only Case 1 viable
        p1 = (abbc_pred == 1).astype(np.float32)
        p2 = (abbc_pred == 2).astype(np.float32)
        p3 = (abbc_pred == 3).astype(np.float32)

    cost = (
        cfg.w_min
        + cfg.w_prob1 * p1
        + cfg.w_prob2 * p2
        - cfg.w_prob3 * p3
    )
    cost = np.maximum(cost, cfg.w_min)
    cost[~instance_mask] = cfg.w_bg_outside
    return cost.astype(np.float32)


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------
def cortical_anchored_split(
    abbc_pred: np.ndarray,
    instances: np.ndarray,
    softmax: Optional[np.ndarray] = None,
    cfg: Optional[SplitConfig] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the v2 split on every initial instance.

    Parameters
    ----------
    abbc_pred : (D,H,W) int  — argmax of network (0/1/2/3)
    instances : (D,H,W) int  — current abbc2instance output
    softmax   : (4,D,H,W) float | None — full softmax. Strongly recommended
                for Case 2/3; Case 1 can work without it. None → fall back
                to argmax-derived hard probabilities.
    cfg       : SplitConfig | None
    """
    if cfg is None:
        cfg = SplitConfig()

    instances_out = instances.copy()
    label3_added = np.zeros(abbc_pred.shape, dtype=bool)
    next_id = int(instances.max()) + 1
    diag: dict = {"per_instance": []}

    prob3 = softmax[3] if softmax is not None else (abbc_pred == 3).astype(np.float32)

    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        inst_mask = instances == inst_id
        if int(inst_mask.sum()) < cfg.min_instance_voxels:
            continue
        cortical_mask = inst_mask & (abbc_pred == 1)
        if int(cortical_mask.sum()) < cfg.min_cortical_voxels:
            continue

        # Gather seeds from all sources
        seeds_cortical = cortical_truncation_seeds(cortical_mask, inst_mask, cfg)
        seeds_prob3 = prob3_peak_seeds(prob3, inst_mask, cfg) if softmax is not None else []
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
        cost = build_cost_field(abbc_pred, softmax, inst_mask, cfg)

        for p in pairs:
            if diag_entry["splits"] >= max_splits_for_this_instance:
                break
            cut = min_cut_surface(inst_mask, p.src, p.snk, cost)
            if cut is None or not np.any(cut):
                diag_entry["pairs"].append(
                    {"case": p.case, "score": p.score, "result": "no_cut"}
                )
                continue
            # Validate disconnection
            split_mask = inst_mask & ~cut
            labelled, n = nd_label(split_mask)
            if n <= 1:
                diag_entry["pairs"].append(
                    {"case": p.case, "score": p.score, "result": "did_not_disconnect"}
                )
                continue
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            keep_id_local = int(sizes.argmax())

            # Reject pieces that don't contain a core voxel — they are
            # geometric chips, not real fragments. This avoids over-splitting
            # when N>=3 cuts cascade and a min-cut nicks off a thin sliver.
            cores_global = abbc_pred == 2
            n_new_pieces = 0
            for piece_id in range(1, n + 1):
                if piece_id == keep_id_local:
                    continue
                piece_mask = labelled == piece_id
                if int(piece_mask.sum()) < cfg.min_split_piece_size:
                    continue
                if not np.any(piece_mask & cores_global):
                    continue
                instances_out[piece_mask] = next_id
                next_id += 1
                n_new_pieces += 1
            if n_new_pieces == 0:
                # cut disconnected only chip-off slivers without cores → revert
                diag_entry["pairs"].append(
                    {"case": p.case, "score": p.score, "result": "all_chips_no_cores"}
                )
                continue
            label3_added |= cut
            diag_entry["splits"] += n_new_pieces
            diag_entry["pairs"].append(
                {"case": p.case, "score": float(p.score),
                 "result": f"split_into_{n_new_pieces + 1}",
                 "cut_voxels": int(cut.sum())}
            )
            inst_mask = instances_out == inst_id  # refresh for the next pair

        diag["per_instance"].append(diag_entry)

    return instances_out, label3_added, diag
