import numpy as np
from skimage.segmentation import watershed

from tools.charite_cortical.continuity_workflow import minimax_watershed_instances
from tools.charite_cortical.diagnose_continuity import diagnose_arrays, oracle_seed_partition, separator_markers


def _case():
    gt = np.ones((7, 7, 7), dtype=np.int32)
    gt[:, :, 4:] = 2
    semantic = np.ones_like(gt)
    probability = np.full(gt.shape, 0.05, dtype=np.float32)
    return gt, semantic, probability, np.full(gt.shape, 3, dtype=np.int16)


def test_one_low_probability_bridge_merges_two_otherwise_separated_seeds():
    gt, semantic, probability, validity = _case()
    probability[:, :, 3] = 0.95
    probability[3, 3, 3] = 0.05
    predicted = minimax_watershed_instances(semantic, probability)
    result = diagnose_arrays(semantic, probability, predicted, gt, validity)
    assert result["marker_count"] == 1
    assert result["mixed_seed_child_count"] == 2
    assert result["touching_mixed_seed_pair_count"] == 1
    assert result["coverage_upper_bound_recovery_fraction"] == 1.0
    assert result["metrics"]["child_recovery_fraction"] < 1.0


def test_missing_cortex_is_detected_even_with_perfect_instance_assignment():
    gt, semantic, probability, validity = _case()
    semantic[:, :, 4:] = 0
    predicted = minimax_watershed_instances(semantic, probability)
    result = diagnose_arrays(semantic, probability, predicted, gt, validity)
    assert result["coverage_limited_child_count"] == 1
    assert result["coverage_upper_bound_recovery_fraction"] == 0.5
    assert result["mixed_marker_count"] == 0


def test_marker_reconstruction_matches_frozen_postprocessor_with_fallback():
    gt, semantic, probability, _ = _case()
    probability[:] = 0.95
    union = semantic > 0
    markers, _, fallback = separator_markers(union, probability)
    reconstructed = watershed(probability, markers=markers, mask=union, connectivity=np.ones((3, 3, 3)), watershed_line=False)
    assert fallback == 1
    np.testing.assert_array_equal(reconstructed, minimax_watershed_instances(semantic, probability))


def test_invalid_ownership_does_not_create_a_false_mixed_seed():
    gt, semantic, probability, validity = _case()
    validity[gt == 2] = 1
    predicted = minimax_watershed_instances(semantic, probability)
    result = diagnose_arrays(semantic, probability, predicted, gt, validity)
    assert result["mixed_marker_count"] == 0
    assert result["touching_gt_pairs"] == []


def test_independent_gt_markers_can_split_a_connected_single_marker_region():
    gt, semantic, probability, validity = _case()
    _, predicted = oracle_seed_partition(semantic, probability, gt, validity)
    assert len(np.unique(predicted)) == 2
    np.testing.assert_array_equal(predicted > 0, semantic > 0)
    assert predicted[0, 0, 0] != predicted[0, 0, -1]


def test_oracle_seeds_cannot_recover_missing_cortical_voxels():
    gt, semantic, probability, validity = _case()
    semantic[gt == 2] = 0
    markers, predicted = oracle_seed_partition(semantic, probability, gt, validity)
    assert not np.any(markers == 2)
    assert np.all(predicted[gt == 2] == 0)
