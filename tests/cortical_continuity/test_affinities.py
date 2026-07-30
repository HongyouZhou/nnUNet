from __future__ import annotations

import unittest

import numpy as np

from nnunetv2.training.cortical_continuity.affinities import build_continuity_targets
from nnunetv2.training.cortical_continuity.schema import (
    AffinityOffset,
    build_cortical_continuity_schema,
)


def _channel(schema, offset_zyx: tuple[int, int, int]) -> int:
    return next(
        index
        for index, offset in enumerate(schema.affinity_offsets)
        if offset.voxel_offset_zyx == offset_zyx
    )


class AffinityTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))

    def test_local_same_and_different_instance_edges(self) -> None:
        instances = np.zeros((8, 8, 8), dtype=np.int16)
        instances[1, 1, 1:3] = 1
        instances[2, 2, 2] = 1
        instances[2, 2, 3] = 2
        targets = build_continuity_targets(instances, self.schema)
        channel = _channel(self.schema, (0, 0, 1))
        self.assertTrue(targets.affinity_valid[channel, 1, 1, 1])
        self.assertEqual(targets.affinity_target[channel, 1, 1, 1], 1)
        self.assertTrue(targets.affinity_valid[channel, 2, 2, 2])
        self.assertEqual(targets.affinity_target[channel, 2, 2, 2], 0)
        self.assertFalse(targets.affinity_valid[channel, 0, 0, 0])
        self.assertFalse(targets.affinity_valid[channel, 7, 7, 7])

    def test_lifted_affinity_uses_physical_schema_offset(self) -> None:
        instances = np.zeros((8, 8, 8), dtype=np.int16)
        instances[1, 4, 4] = instances[4, 4, 4] = 7
        targets = build_continuity_targets(instances, self.schema)
        channel = _channel(self.schema, (3, 0, 0))
        self.assertTrue(targets.affinity_valid[channel, 1, 4, 4])
        self.assertEqual(targets.affinity_target[channel, 1, 4, 4], 1)

    def test_overlap_is_cortex_positive_but_relation_invalid(self) -> None:
        instances = np.zeros((8, 8, 8), dtype=np.int16)
        instances[1, 1, 1:3] = 1
        overlap = np.zeros_like(instances, dtype=bool)
        overlap[1, 1, 2] = True
        targets = build_continuity_targets(
            instances,
            self.schema,
            cortex_mask=instances > 0,
            overlap_mask=overlap,
        )
        channel = _channel(self.schema, (0, 0, 1))
        self.assertEqual(targets.cortex_target[0, 1, 1, 2], 1)
        self.assertTrue(targets.cortex_valid[0, 1, 1, 2])
        self.assertFalse(targets.affinity_valid[channel, 1, 1, 1])

    def test_semantic_and_relation_masks_are_independent(self) -> None:
        instances = np.zeros((8, 8, 8), dtype=np.int16)
        instances[2, 2, 2:4] = 3
        semantic_valid = np.ones_like(instances, dtype=bool)
        semantic_valid[2, 2, 2] = False
        targets = build_continuity_targets(
            instances,
            self.schema,
            cortex_valid_mask=semantic_valid,
            relation_valid_mask=np.ones_like(instances, dtype=bool),
        )
        channel = _channel(self.schema, (0, 0, 1))
        self.assertFalse(targets.cortex_valid[0, 2, 2, 2])
        self.assertTrue(targets.affinity_valid[channel, 2, 2, 2])

    def test_background_case_has_finite_empty_targets(self) -> None:
        targets = build_continuity_targets(
            np.zeros((4, 5, 6), dtype=np.uint16),
            self.schema,
        )
        self.assertEqual(targets.cortex_target.shape, (1, 4, 5, 6))
        self.assertEqual(targets.affinity_target.shape, (19, 4, 5, 6))
        self.assertFalse(targets.cortex_target.any())
        self.assertFalse(targets.affinity_valid.any())
        self.assertTrue(np.isfinite(targets.affinity_target).all())

    def test_offset_larger_than_patch_does_not_wrap(self) -> None:
        offset = AffinityOffset(
            voxel_offset_zyx=(0, 0, 10),
            physical_offset_mm_zyx=(0.0, 0.0, 10.0),
            family="test",
            requested_distance_mm=10.0,
        )
        targets = build_continuity_targets(
            np.ones((3, 3, 3), dtype=np.int16),
            (offset,),
        )
        self.assertEqual(targets.affinity_valid.shape, (1, 3, 3, 3))
        self.assertFalse(targets.affinity_valid.any())

    def test_instance_map_validation(self) -> None:
        fixtures = (
            (np.zeros((3, 3), dtype=np.int16), "must be 3D"),
            (np.zeros((3, 3, 3), dtype=np.float32), "integer instance IDs"),
            (-np.ones((3, 3, 3), dtype=np.int16), "non-negative"),
        )
        for instance_map, error in fixtures:
            with self.subTest(error=error), self.assertRaisesRegex((TypeError, ValueError), error):
                build_continuity_targets(instance_map, self.schema)

    def test_mask_shape_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_continuity_targets(
                np.zeros((3, 3, 3), dtype=np.int16),
                self.schema,
                overlap_mask=np.zeros((2, 2, 2), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
