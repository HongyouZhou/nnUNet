from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from tools.PENGWIN.postprocess.benchmark_interactive_mincut import (
    _affected_pq,
    _core_recognition_metrics,
    _mask_sha256,
    _match_binary_instance_pair,
    _overlap_metrics,
    _sample_centers,
    _sphere_seed,
    _write_roi_image,
)


class InteractiveBenchmarkTests(unittest.TestCase):
    def test_physical_sphere_seed_stays_inside_allowed_mask(self) -> None:
        allowed = np.zeros((21, 21, 21), dtype=bool)
        allowed[2:19, 2:19, 2:19] = True
        spacing = np.ones(3, dtype=float)
        center = np.asarray([10, 10, 10])

        seed = _sphere_seed(allowed, center, spacing, radius_mm=2.0)

        self.assertEqual(int(seed.sum()), 33)
        self.assertFalse(np.any(seed & ~allowed))

    def test_sampled_centers_respect_radius_and_clearance(self) -> None:
        allowed = np.zeros((25, 25, 25), dtype=bool)
        allowed[2:23, 2:23, 2:23] = True
        spacing = np.asarray([1.0, 1.0, 1.0])

        centers, depth = _sample_centers(
            allowed,
            spacing,
            radius_mm=3.0,
            clearance_mm=1.0,
            trials=8,
            rng=np.random.default_rng(95),
        )

        self.assertEqual(len(centers), 8)
        self.assertEqual(len({tuple(center) for center in centers}), 8)
        self.assertTrue(all(depth[tuple(center)] >= 4.0 for center in centers))

    def test_core_center_sampling_does_not_require_contained_brush(self) -> None:
        allowed = np.zeros((9, 9, 9), dtype=bool)
        allowed[2:7, 2:7, 2:7] = True

        centers, depth = _sample_centers(
            allowed,
            np.ones(3),
            radius_mm=4.0,
            clearance_mm=0.5,
            trials=5,
            rng=np.random.default_rng(95),
            require_contained_seed=False,
        )

        self.assertEqual(len({tuple(center) for center in centers}), 5)
        self.assertTrue(all(depth[tuple(center)] >= 0.5 for center in centers))

    def test_random_sampling_has_no_forced_deepest_center(self) -> None:
        allowed = np.zeros((15, 15, 15), dtype=bool)
        allowed[2:13, 2:13, 2:13] = True

        class FirstCandidates:
            @staticmethod
            def choice(count: int, size: int, replace: bool) -> np.ndarray:
                self.assertFalse(replace)
                self.assertGreaterEqual(count, size)
                return np.arange(size)

        centers, depth = _sample_centers(
            allowed,
            np.ones(3),
            radius_mm=1.0,
            clearance_mm=0.5,
            trials=10,
            rng=FirstCandidates(),
            include_deepest=False,
        )
        deepest = tuple(np.unravel_index(int(np.argmax(depth)), depth.shape))

        self.assertEqual(len({tuple(center) for center in centers}), 10)
        self.assertNotIn(deepest, {tuple(center) for center in centers})

    def test_affected_pq_requires_both_instance_matches(self) -> None:
        true_positives, false_positives, false_negatives, pq = _affected_pq(0.8, 0.9, 0.5)
        self.assertEqual((true_positives, false_positives, false_negatives), (2, 0, 0))
        self.assertAlmostEqual(pq, 0.85)

        true_positives, false_positives, false_negatives, pq = _affected_pq(0.8, 0.4, 0.5)
        self.assertEqual((true_positives, false_positives, false_negatives), (1, 1, 1))
        self.assertAlmostEqual(pq, 0.4)

    def test_binary_instance_matching_is_label_permutation_invariant(self) -> None:
        source_gt = np.zeros((3, 3, 8), dtype=bool)
        sink_gt = np.zeros_like(source_gt)
        source_gt[:, :, :3] = True
        sink_gt[:, :, 5:] = True

        source_prediction, sink_prediction, metrics = _match_binary_instance_pair(
            sink_gt.copy(),
            source_gt.copy(),
            source_gt,
            sink_gt,
        )

        self.assertEqual(metrics["matched_orientation"], "swapped")
        self.assertEqual(metrics["source_iou"], 1.0)
        self.assertEqual(metrics["sink_iou"], 1.0)
        self.assertTrue(np.array_equal(source_prediction, source_gt))
        self.assertTrue(np.array_equal(sink_prediction, sink_gt))

    def test_overlap_metrics(self) -> None:
        prediction = np.asarray([True, True, False, False])
        ground_truth = np.asarray([True, False, True, False])
        dice, iou = _overlap_metrics(prediction, ground_truth)
        self.assertAlmostEqual(dice, 0.5)
        self.assertAlmostEqual(iou, 1.0 / 3.0)

    def test_mask_hash_tracks_voxel_level_partition(self) -> None:
        first = np.zeros((3, 3, 3), dtype=bool)
        second = first.copy()
        first[1, 1, 1] = True
        second[1, 1, 2] = True

        self.assertEqual(_mask_sha256(first), _mask_sha256(first.copy()))
        self.assertNotEqual(_mask_sha256(first), _mask_sha256(second))

    def test_core_recognition_ignores_peripheral_boundary_error(self) -> None:
        source_core = np.zeros((3, 3, 8), dtype=bool)
        sink_core = np.zeros_like(source_core)
        source_core[:, :, 1:3] = True
        sink_core[:, :, 5:7] = True
        source_prediction = np.zeros_like(source_core)
        source_prediction[:, :, :4] = True
        sink_prediction = ~source_prediction

        metrics = _core_recognition_metrics(
            source_prediction,
            sink_prediction,
            source_core,
            sink_core,
            recall_threshold=0.9,
        )

        self.assertEqual(metrics["source_core_recall"], 1.0)
        self.assertEqual(metrics["sink_core_recall"], 1.0)
        self.assertTrue(metrics["fragment_recognition_success"])

    def test_roi_output_keeps_physical_location(self) -> None:
        grid = {
            "spacing": (0.5, 0.75, 1.0),
            "origin": (10.0, 20.0, 30.0),
            "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "roi.nii.gz"
            _write_roi_image(path, np.zeros((3, 4, 5), dtype=np.uint8), grid, (2, 4, 6))
            image = sitk.ReadImage(str(path))

        self.assertEqual(image.GetSpacing(), grid["spacing"])
        self.assertEqual(image.GetOrigin(), (11.0, 23.0, 36.0))


if __name__ == "__main__":
    unittest.main()
