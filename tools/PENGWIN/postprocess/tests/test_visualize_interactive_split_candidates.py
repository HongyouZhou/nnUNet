from __future__ import annotations

import unittest

import numpy as np

from tools.PENGWIN.postprocess.visualize_interactive_split_candidates import (
    _all_boundaries_2d,
    _apply_split_to_instances,
)


class FullInstanceVisualizationTests(unittest.TestCase):
    def test_apply_split_preserves_every_unaffected_instance(self) -> None:
        instances = np.zeros((4, 5, 6), dtype=np.uint16)
        instances[:, :, :2] = 1
        instances[:, :, 2:4] = 3
        instances[:, :, 4:] = 7
        source = np.zeros_like(instances, dtype=bool)
        sink = np.zeros_like(instances, dtype=bool)
        source[:2, :, 2:4] = True
        sink[2:, :, 2:4] = True

        result, new_instance_id = _apply_split_to_instances(instances, 3, source, sink)

        self.assertEqual(new_instance_id, 8)
        self.assertTrue(np.array_equal(result[instances == 1], instances[instances == 1]))
        self.assertTrue(np.array_equal(result[instances == 7], instances[instances == 7]))
        self.assertTrue(np.all(result[source] == 3))
        self.assertTrue(np.all(result[sink] == 8))
        self.assertEqual(set(np.unique(result)), {1, 3, 7, 8})

    def test_apply_split_rejects_an_incomplete_partition(self) -> None:
        instances = np.ones((3, 3, 3), dtype=np.uint16)
        source = np.zeros_like(instances, dtype=bool)
        sink = np.zeros_like(instances, dtype=bool)
        source[0] = True
        sink[2] = True

        with self.assertRaisesRegex(ValueError, "exactly partition"):
            _apply_split_to_instances(instances, 1, source, sink)

    def test_all_boundaries_includes_background_and_instance_interfaces(self) -> None:
        labels = np.asarray(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
            ],
            dtype=np.uint8,
        )

        boundary = _all_boundaries_2d(labels)

        self.assertTrue(boundary[0, 1])
        self.assertTrue(boundary[1, 2])
        self.assertTrue(boundary[1, 3])
        self.assertFalse(boundary[1, 0])


if __name__ == "__main__":
    unittest.main()
