from __future__ import annotations

import unittest

import numpy as np

from tools.charite_cortical.continuity_postprocess import (
    aggregate_patient_macro,
    evaluate_instance_segmentation,
    evaluate_partition_grouping,
    evaluate_split_calibration,
    hungarian_match_iou_matrix,
)


def _volume(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint16).reshape(1, 1, -1)


class HungarianEvaluatorTests(unittest.TestCase):
    def test_hungarian_solves_a_greedy_counterexample(self) -> None:
        matrix = np.asarray([[0.90, 0.80], [0.85, 0.10]], dtype=np.float64)
        assignments = hungarian_match_iou_matrix(matrix)
        self.assertEqual(set(assignments), {(0, 1), (1, 0)})
        self.assertAlmostEqual(
            sum(matrix[row, column] for row, column in assignments),
            1.65,
        )

    def test_permuted_instance_ids_are_perfect(self) -> None:
        ground_truth = _volume([1, 1, 2, 2])
        prediction = _volume([9, 9, 4, 4])
        result = evaluate_instance_segmentation(
            prediction,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertEqual(result.true_positives, 2)
        self.assertAlmostEqual(result.panoptic_quality, 1.0)
        self.assertTrue(result.all_child_recovery)
        self.assertEqual(result.normalized_vi_merge, 0.0)
        self.assertEqual(result.normalized_vi_split, 0.0)

    def test_kind_is_explicit_not_inferred_from_small_label_values(self) -> None:
        ground_truth = _volume([1, 1, 2, 2, 3, 3])
        result = evaluate_instance_segmentation(
            ground_truth,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertEqual(result.ground_truth_count, 3)
        with self.assertRaisesRegex(ValueError, "explicit"):
            evaluate_instance_segmentation(
                ground_truth,
                ground_truth,
                prediction_kind="semantic",  # type: ignore[arg-type]
                ground_truth_kind="instance",
            )

    def test_vi_directions_distinguish_merge_and_split(self) -> None:
        merge = evaluate_instance_segmentation(
            _volume([1, 1, 1, 1]),
            _volume([1, 1, 2, 2]),
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertGreater(merge.vi_merge, 0.0)
        self.assertAlmostEqual(merge.vi_split, 0.0)

        split = evaluate_instance_segmentation(
            _volume([1, 1, 2, 2]),
            _volume([1, 1, 1, 1]),
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertAlmostEqual(split.vi_merge, 0.0)
        self.assertGreater(split.vi_split, 0.0)
        self.assertTrue(split.intact_false_split)

    def test_extra_child_fails_all_child_recovery_and_reduces_pq(self) -> None:
        result = evaluate_instance_segmentation(
            _volume([1, 3, 2, 2]),
            _volume([1, 1, 2, 2]),
            prediction_kind="instance",
            ground_truth_kind="instance",
            iou_threshold=0.5,
        )
        self.assertFalse(result.all_child_recovery)
        self.assertEqual(result.false_positives, 1)
        self.assertLess(result.panoptic_quality, 1.0)

    def test_valid_mask_excludes_ambiguous_overlap_voxels(self) -> None:
        ground_truth = _volume([1, 2, 2])
        prediction = _volume([9, 9, 8])
        valid = np.asarray([False, True, True]).reshape(1, 1, 3)
        result = evaluate_instance_segmentation(
            prediction,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
            valid_mask=valid,
        )
        self.assertEqual(result.ground_truth_count, 1)


class PartitionAndCalibrationTests(unittest.TestCase):
    def test_patient_macro_first_averages_repeated_patient_records(self) -> None:
        aggregation = aggregate_patient_macro(
            [
                {"patient_id": "a", "metrics": {"pq": 0.0}},
                {"patient_id": "a", "metrics": {"pq": 1.0}},
                {"patient_id": "b", "metrics": {"pq": 1.0}},
            ],
            ["pq"],
            bootstrap_iterations=100,
        )
        self.assertEqual(aggregation.patient_count, 2)
        self.assertEqual(aggregation.record_count, 3)
        self.assertAlmostEqual(aggregation.metrics["pq"].mean, 0.75)

    def test_partition_metrics_are_label_invariant_and_detect_a_merge(self) -> None:
        ground_truth = _volume([1, 1, 2, 2])
        permuted = _volume([8, 8, 3, 3])
        perfect = evaluate_partition_grouping(
            permuted,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertTrue(perfect.identity_grouping_correct)
        self.assertEqual(perfect.adjusted_rand_index, 1.0)
        self.assertEqual(perfect.pairwise_f1, 1.0)

        merged = evaluate_partition_grouping(
            np.ones_like(ground_truth),
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
        )
        self.assertFalse(merged.exact_k)
        self.assertFalse(merged.identity_grouping_correct)
        self.assertLess(merged.adjusted_rand_index, 1.0)
        self.assertGreater(merged.vi_merge, 0.0)

    def test_calibration_reports_brier_ece_risk_coverage_and_intact_risk(self) -> None:
        evaluation = evaluate_split_calibration(
            [0.9, 0.8, 0.2, 0.1],
            [True, True, False, False],
            bins=5,
            operating_threshold=0.75,
            intact_negative=[False, False, True, True],
        )
        self.assertAlmostEqual(evaluation.brier_score, 0.025)
        self.assertAlmostEqual(evaluation.expected_calibration_error, 0.15)
        self.assertEqual(evaluation.intact_false_split_rate, 0.0)
        self.assertEqual(evaluation.risk_coverage[-1].coverage, 1.0)
        self.assertEqual(evaluation.risk_coverage[1].risk, 0.0)


if __name__ == "__main__":
    unittest.main()
