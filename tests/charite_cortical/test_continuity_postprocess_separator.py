from __future__ import annotations

import unittest

import numpy as np

from tools.charite_cortical.continuity_postprocess import refine_with_separator


def _separator_case(
    length: int,
    separator_positions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    provisional = np.full((1, 1, length), 7, dtype=np.uint16)
    softmax = np.zeros((3,) + provisional.shape, dtype=np.float32)
    softmax[0] = 0.1
    softmax[1] = 0.9
    for position in separator_positions:
        softmax[1, 0, 0, position] = 0.0
        softmax[2, 0, 0, position] = 0.9
    return provisional, softmax


class SeparatorBaselineTests(unittest.TestCase):
    def test_no_separator_is_a_k1_no_op(self) -> None:
        provisional, softmax = _separator_case(6, ())
        result = refine_with_separator(
            provisional,
            softmax,
            spacing_zyx=(1.0, 1.0, 1.0),
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].status, "unchanged")
        self.assertEqual(result.decisions[0].reason, "single_cortical_cluster")

    def test_one_separator_produces_unknown_k_two(self) -> None:
        provisional, softmax = _separator_case(6, (2,))
        result = refine_with_separator(
            provisional,
            softmax,
            spacing_zyx=(1.0, 1.0, 1.0),
        )
        self.assertEqual(result.decisions[0].status, "accepted")
        self.assertEqual(result.decisions[0].raw_clusters, 2)
        self.assertEqual(set(np.unique(result.full_instances)), {7, 8})
        self.assertTrue(
            np.array_equal(result.full_instances > 0, provisional > 0)
        )
        self.assertTrue(np.all(result.cortical_instances > 0))

    def test_two_separators_produce_unknown_k_three(self) -> None:
        provisional, softmax = _separator_case(8, (2, 5))
        result = refine_with_separator(
            provisional,
            softmax,
            spacing_zyx=(1.0, 1.0, 1.0),
        )
        self.assertEqual(result.decisions[0].status, "accepted")
        self.assertEqual(result.decisions[0].raw_clusters, 3)
        self.assertEqual(len(np.unique(result.full_instances)), 3)

    def test_disconnected_cortex_without_separator_evidence_abstains(self) -> None:
        provisional, softmax = _separator_case(6, ())
        softmax[:, 0, 0, 2] = (1.0, 0.0, 0.0)
        result = refine_with_separator(
            provisional,
            softmax,
            spacing_zyx=(1.0, 1.0, 1.0),
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].status, "abstained")
        self.assertEqual(
            result.decisions[0].reason,
            "separator_not_incident_to_all_clusters",
        )


if __name__ == "__main__":
    unittest.main()
