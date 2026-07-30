from __future__ import annotations

import unittest

import numpy as np

from tools.charite_cortical.continuity_postprocess import (
    controlled_event_manifest,
    run_o1_event,
    run_o2_event,
)


class ControlledOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fragments = np.asarray(
            [[[1, 1, 2, 2, 3, 3]]],
            dtype=np.uint16,
        )
        self.cortical = self.fragments.copy()
        self.spacing = (1.0, 1.0, 1.0)

    def test_manifest_contains_adjacent_pairs_connected_multiway_and_intact(self) -> None:
        manifest = controlled_event_manifest(
            self.fragments,
            self.cortical,
            self.spacing,
            maximum_surface_distance_mm=1.1,
        )
        binary = [event for event in manifest.events if event.kind == "binary"]
        multiway = [event for event in manifest.events if event.kind == "multiway"]
        intact = [event for event in manifest.events if event.kind == "intact"]
        self.assertEqual([event.child_ids for event in binary], [(1, 2), (2, 3)])
        self.assertEqual([event.child_ids for event in multiway], [(1, 2, 3)])
        self.assertEqual(len(intact), 3)

    def test_manifest_reports_fragments_without_valid_cortex(self) -> None:
        cortical = self.cortical.copy()
        cortical[cortical == 3] = 0
        manifest = controlled_event_manifest(
            self.fragments,
            cortical,
            self.spacing,
            maximum_surface_distance_mm=1.1,
        )
        self.assertEqual(manifest.cortex_missing_fragment_ids, (3,))
        self.assertNotIn(3, manifest.eligible_fragment_ids)

    def test_o1_and_o2_recover_a_three_way_controlled_event(self) -> None:
        manifest = controlled_event_manifest(
            self.fragments,
            self.cortical,
            self.spacing,
            maximum_surface_distance_mm=1.1,
        )
        event = next(event for event in manifest.events if event.kind == "multiway")
        o1 = run_o1_event(
            event,
            self.fragments,
            self.cortical,
            self.spacing,
        )
        self.assertTrue(o1.evaluation.all_child_recovery)
        self.assertAlmostEqual(o1.evaluation.panoptic_quality, 1.0)
        self.assertAlmostEqual(
            o1.cortical_grouping_evaluation.panoptic_quality,
            1.0,
        )

        o2 = run_o2_event(
            event,
            self.fragments,
            self.cortical,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            self.spacing,
        )
        self.assertTrue(o2.evaluation.all_child_recovery)
        self.assertAlmostEqual(o2.evaluation.panoptic_quality, 1.0)
        self.assertAlmostEqual(
            o2.cortical_grouping_evaluation.panoptic_quality,
            1.0,
        )

    def test_overlap_mask_is_excluded_from_oracle_evaluation(self) -> None:
        overlap = np.zeros(self.fragments.shape, dtype=bool)
        overlap[0, 0, 0] = True
        manifest = controlled_event_manifest(
            self.fragments,
            self.cortical,
            self.spacing,
            maximum_surface_distance_mm=1.1,
            overlap_mask=overlap,
        )
        event = next(
            event
            for event in manifest.events
            if event.kind == "binary" and event.child_ids == (1, 2)
        )
        result = run_o1_event(
            event,
            self.fragments,
            self.cortical,
            self.spacing,
            overlap_mask=overlap,
        )
        self.assertTrue(result.evaluation.all_child_recovery)


if __name__ == "__main__":
    unittest.main()
