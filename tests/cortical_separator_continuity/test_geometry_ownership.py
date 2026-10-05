import numpy as np
import pytest
from scipy.ndimage import label

from tools.charite_cortical.continuity_workflow import evaluate_instances
from tools.charite_cortical.geometry_ownership import (
    GeometryConfig, constrained_partition, geometry_ownership, patch_graph,
    sheet_relations, supported_repulsion,
)
from tools.charite_cortical.run_geometry_ownership import audit_graph, ownership_evaluation, prediction_crop, sparse_evaluation


def test_repulsive_ownership_survives_a_stronger_smooth_residual_bridge():
    # Fracture evidence at 0--1; an unbroken 0--2--3--1 hinge remains.
    edges = np.array([[0, 1], [0, 2], [2, 3], [3, 1]])
    output, rejected = constrained_partition(4, edges, np.array([0, 100, 99, 98]), np.array([1, 0, 0, 0], bool))
    assert output[0] != output[1]
    assert rejected == 1
    assert len(np.unique(output)) == 2


def test_smooth_connected_curved_sheet_can_remain_one_owner():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    output, rejected = constrained_partition(4, edges, np.ones(4), np.zeros(4, bool))
    assert len(np.unique(output)) == 1
    assert rejected == 0


def test_all_repulsive_relations_hold_under_global_agglomeration():
    rng = np.random.default_rng(778)
    for _ in range(10):
        edges = np.column_stack(np.triu_indices(35, 1))
        repulsive = rng.random(len(edges)) < 0.08
        output, _ = constrained_partition(35, edges, rng.random(len(edges)), repulsive)
        assert np.all(output[edges[repulsive, 0]] != output[edges[repulsive, 1]])


def test_patches_do_not_alias_nearby_disconnected_sheets_in_one_grid_cell():
    base = np.zeros((5, 5, 5), np.int32)
    base[1, 1, 1] = base[3, 1, 1] = 1
    _, patches, edges, _, _ = patch_graph(base, np.ones(3), 10)
    assert patches[0] != patches[1]
    assert len(edges) == 0


def test_patches_preserve_diagonal_connectivity_and_frozen_ownership():
    base = np.zeros((4, 4, 4), np.int32)
    base[0, 0, 0] = base[1, 1, 1] = 1
    base[2, 2, 2] = 2
    _, patches, edges, _, _ = patch_graph(base, np.ones(3), 10)
    assert patches[0] == patches[1]
    assert patches[1] != patches[2]
    assert len(edges) == 0


def _normal(degrees):
    angles = np.radians(degrees)
    return np.column_stack((np.sin(angles), np.zeros(len(angles)), np.cos(angles)))


def test_multiscale_relation_distinguishes_direction_jump_from_smooth_curvature():
    edges = np.array([[0, 1]])
    fine = (_normal([0, 35]), np.ones(2))
    context_smooth = (_normal([0, 34]), np.ones(2))
    context_jump = (_normal([10, 25]), np.ones(2))
    args = (np.zeros(2), np.ones(1), GeometryConfig())
    assert not sheet_relations(edges, fine, context_smooth, *args)[1][0]
    assert sheet_relations(edges, fine, context_jump, *args)[1][0]
    fine = (fine[0] * np.array([[1], [-1]]), fine[1])
    assert sheet_relations(edges, fine, context_jump, *args)[1][0]


def test_uncertain_corner_and_separator_alone_do_not_create_repulsion():
    edges = np.array([[0, 1]])
    fine = (_normal([0, 70]), np.zeros(2))
    context = (_normal([30, 40]), np.ones(2))
    assert not sheet_relations(edges, fine, context, np.ones(2), np.ones(1), GeometryConfig())[1][0]


def test_isolated_orientation_outlier_does_not_pass_seam_support():
    edges = np.array([[0, 1], [2, 3], [3, 4], [4, 5], [5, 6]])
    keep = supported_repulsion(edges, np.ones(5, bool), 7, 4)
    np.testing.assert_array_equal(keep, [False, True, True, True, True])


@pytest.mark.parametrize("spacing", [(1, 1, 1), (0.6, 0.6, 1.2)])
def test_ct_normal_curvature_control_has_no_geometry_false_split(spacing):
    spacing = np.array(spacing)
    shape = tuple(np.ceil(np.array([36, 24, 20]) / spacing).astype(int))
    x, y, z = np.indices(shape) * spacing[:, None, None, None]
    x -= 18; z -= 8
    distance = z - 0.015 * x**2
    ct = (1200 * np.exp(-0.5 * (distance / 0.7)**2)).astype(np.float32)
    semantic = (np.abs(distance) <= 1).astype(np.int16)
    output, baseline, record = geometry_ownership(semantic, np.zeros(shape, np.float32), ct, spacing)
    assert record["supported_repulsive_edges"] == 0
    assert len(np.unique(output[output > 0])) == 1
    np.testing.assert_array_equal(output > 0, baseline > 0)


def test_automatic_ct_direction_evidence_can_separate_a_connected_sharp_sheet():
    spacing = np.array([0.6, 0.6, 0.6])
    shape = (70, 50, 85)
    x, y, z = np.indices(shape) * spacing[:, None, None, None]
    x -= 21; z -= 10
    distance = z - 0.9 * np.abs(x)
    ct = (1200 * np.exp(-0.5 * (distance / 0.65)**2)).astype(np.float32)
    semantic = (np.abs(distance) <= 0.8).astype(np.int16)
    assert label(semantic, structure=np.ones((3, 3, 3)))[1] == 1
    output, _, record = geometry_ownership(semantic, np.zeros(shape, np.float32), ct, spacing)
    target = np.where(semantic, np.where(x < 0, 1, 2), 0)
    metrics = sparse_evaluation(output, semantic > 0, target, np.full(shape, 3))
    assert record["supported_repulsive_edges"] > 0
    assert metrics["recovered_child_count"] == 2
    assert metrics["false_split_child_count"] == 0
    # The synthetic ownership is stipulated. This does not certify that an
    # anatomically intact sharp ridge should be split on real CT.


def test_graph_audit_measures_same_fragment_repulsion_without_affecting_prediction(tmp_path):
    graph = tmp_path / "graph.npz"
    np.savez(graph, positions=np.arange(6), voxel_patch=np.arange(6), edges=np.array([[0, 1], [1, 2], [2, 3], [4, 5]]),
             repulsive=np.array([True, False, True, True]), partition=np.arange(1, 7),
             fine_angles_deg=np.array([35, 10, 40, 45]), excess_deg=np.array([15, 2, 20, 25]))
    gt = np.array([1, 1, 1, 2, 2, 3]); validity = np.array([3, 3, 3, 3, 3, 1])
    result = audit_graph(graph, gt, validity)
    assert result["audited_edges"] == 3
    assert result["excluded_edges"] == 1
    assert result["same_gt_repulsive_edges"] == 1
    assert result["different_gt_repulsive_edges"] == 1
    assert result["different_gt_fraction_of_audited_repulsion"] == 0.5


def test_continuity_controls_exclude_scan_boundary_truncation():
    gt = np.zeros((90, 90, 90), np.int16)
    gt[10:41, 20:70, 40] = 1; gt[55:75, 20:70, 40] = 2
    result = ownership_evaluation(gt, gt, np.full(gt.shape, 3, np.int16), [], np.ones(3))
    controls = result["same_instance_controls"]
    assert len(controls) == 2
    for control in controls:
        assert control["minimum_scan_boundary_distance_mm"] >= 8 + 4 * (0.5 + 2.7)
        assert not control["false_split"]


def test_empty_automatic_prediction_can_still_be_audited(tmp_path):
    graph = tmp_path / "graph.npz"
    zero = np.zeros((4, 4, 4), np.float32)
    geometry_ownership(zero, zero, zero, (1, 1, 1), graph_output=graph)
    result = audit_graph(graph, np.ones(zero.shape, np.int32), np.full(zero.shape, 3, np.int16))
    assert result["total_edges"] == 0
    assert result["different_gt_fraction_of_audited_repulsion"] is None


def test_prediction_crop_depends_only_on_prediction_and_physical_context():
    semantic = np.zeros((70, 70, 70), np.int16); semantic[35, 35, 35] = 1
    crop = prediction_crop(semantic, np.array([0.5, 1, 2]), GeometryConfig())
    assert 35 - crop[0].start > 35 - crop[1].start > 35 - crop[2].start
    assert all(s.start <= 35 < s.stop for s in crop)


def test_sparse_evaluation_matches_whole_grid_with_validity_and_missing_gt():
    rng = np.random.default_rng(778)
    predicted = rng.integers(0, 5, size=(12, 9, 8))
    gt = rng.integers(0, 4, size=predicted.shape)
    validity = rng.integers(0, 4, size=predicted.shape)
    assert sparse_evaluation(predicted, predicted > 0, gt, validity) == evaluate_instances(predicted, predicted > 0, gt, validity)


def test_empty_mask_and_invalid_physical_parameters():
    zero = np.zeros((4, 4, 4), np.float32)
    output, _, _ = geometry_ownership(zero, zero, zero, (1, 1, 1))
    assert not output.any()
    with pytest.raises(ValueError):
        geometry_ownership(zero, zero, zero, (1, 0, 1))
    with pytest.raises(ValueError):
        GeometryConfig(separator_support_weight=0.8).validate()
