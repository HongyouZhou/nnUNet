from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.PENGWIN.postprocess.find_interactive_split_candidates import (
    _core_pair_diagnostics,
    _hard_label_watershed,
    _write_aggregate,
)


class CandidateScannerTests(unittest.TestCase):
    def test_hard_label_watershed_preserves_foreground_and_two_cores(self) -> None:
        abbc = np.zeros((15, 15, 25), dtype=np.uint8)
        abbc[3:12, 3:12, 2:23] = 1
        abbc[5:9, 5:9, 4:8] = 2
        abbc[5:9, 5:9, 17:21] = 2
        instances = _hard_label_watershed(
            abbc,
            np.zeros_like(abbc, dtype=np.float32),
            min_core_size=10,
        )

        self.assertTrue(np.array_equal(instances > 0, abbc > 0))
        self.assertEqual(set(np.unique(instances)), {0, 1, 2})

    def test_core_pair_diagnostics_distinguishes_separate_and_shared_core(self) -> None:
        shape = (15, 15, 25)
        predicted = np.zeros(shape, dtype=bool)
        predicted[3:12, 3:12, 2:23] = True
        gt = np.zeros(shape, dtype=np.uint8)
        gt[3:12, 3:12, 2:11] = 1
        gt[3:12, 3:12, 14:23] = 2

        separate_abbc = predicted.astype(np.uint8)
        separate_abbc[5:10, 5:10, 4:9] = 2
        separate_abbc[5:10, 5:10, 16:21] = 2
        separate = _core_pair_diagnostics(
            separate_abbc,
            predicted,
            gt,
            1,
            2,
            (1.0, 1.0, 1.0),
            erosion_mm=0.0,
            minimum_component_voxels=20,
        )
        self.assertEqual(separate["core_status"], "separate")

        shared_abbc = predicted.astype(np.uint8)
        shared_abbc[5:10, 5:10, 4:21] = 2
        shared = _core_pair_diagnostics(
            shared_abbc,
            predicted,
            gt,
            1,
            2,
            (1.0, 1.0, 1.0),
            erosion_mm=0.0,
            minimum_component_voxels=20,
        )
        self.assertEqual(shared["core_status"], "shared")

        missing_abbc = predicted.astype(np.uint8)
        missing_abbc[5:10, 5:10, 4:9] = 2
        missing = _core_pair_diagnostics(
            missing_abbc,
            predicted,
            gt,
            1,
            2,
            (1.0, 1.0, 1.0),
            erosion_mm=0.0,
            minimum_component_voxels=20,
        )
        self.assertEqual(missing["core_status"], "missing")

    def test_aggregate_distinguishes_two_gt_and_multi_gt_merges(self) -> None:
        def candidate(first: int, second: int, predicted: int, status: str) -> dict:
            return {
                "case": "case_a",
                "predicted_instance": predicted,
                "first_gt_label": first,
                "second_gt_label": second,
                "core_status": status,
                "minimum_gt_coverage": 0.8,
                "first_overlap_voxels": 1000,
                "second_overlap_voxels": 900,
            }

        case_results = [
            {
                "case": "case_a",
                "status": "ok",
                "candidates": [
                    candidate(1, 2, 1, "missing"),
                    candidate(1, 3, 1, "missing"),
                    candidate(2, 3, 1, "missing"),
                    candidate(4, 5, 2, "shared"),
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            _write_aggregate(Path(directory), case_results)
            summary = json.loads(Path(directory, "summary.json").read_text())

        self.assertEqual(summary["candidate_merged_instances"], 2)
        self.assertEqual(summary["two_gt_merged_instances"], 1)
        self.assertEqual(summary["two_gt_core_status_counts"]["shared"], 1)
        self.assertEqual(summary["two_gt_core_status_counts"]["missing"], 0)


if __name__ == "__main__":
    unittest.main()
