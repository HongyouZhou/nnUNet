from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.charite_cortical.continuity_postprocess import (
    AlwaysAbstainScorer,
    ContinuityConfig,
    ContinuityInputs,
    SignedEdge,
    SignedGraph,
    SplitFeatures,
    build_signed_graph,
    controlled_merge,
    fit_logistic_split_scorer,
    hysteresis_cortex_support,
    load_inputs_npz,
    load_scorer,
    mutex_watershed,
    oracle_affinity_logits,
    propagate_uniform,
    refine,
    save_inputs_npz,
    save_scorer,
)


def _line_graph(probabilities: list[float]) -> tuple[SignedGraph, np.ndarray]:
    shape = (1, 1, len(probabilities) + 1)
    support = np.ones(shape, dtype=bool)
    affinity = np.full((1,) + shape, 0.5, dtype=np.float32)
    affinity[0, 0, 0, : len(probabilities)] = probabilities
    graph = build_signed_graph(
        support,
        affinity,
        np.asarray([[0, 0, 1]], dtype=np.int16),
        ContinuityConfig(),
        affinity_kind="probabilities",
    )
    return graph, support


class CortexSupportTests(unittest.TestCase):
    def test_hysteresis_keeps_only_low_support_reached_from_high_seed(self) -> None:
        probability = np.zeros((1, 1, 7), dtype=np.float32)
        probability[0, 0, 0] = 0.9
        probability[0, 0, 1:3] = 0.4
        probability[0, 0, 5:7] = 0.4
        result = hysteresis_cortex_support(
            probability,
            np.ones_like(probability, dtype=bool),
            0.3,
            0.6,
        )
        self.assertEqual(np.flatnonzero(result).tolist(), [0, 1, 2])


class MutexWatershedTests(unittest.TestCase):
    def test_one_continuous_fragment(self) -> None:
        graph, _ = _line_graph([0.9, 0.9, 0.9, 0.9])
        result = mutex_watershed(graph)
        self.assertEqual(result.cluster_count, 1)
        self.assertEqual(set(result.node_labels), {1})

    def test_two_touching_fragments_split_by_mutex(self) -> None:
        graph, _ = _line_graph([0.9, 0.1, 0.9, 0.9])
        result = mutex_watershed(graph)
        self.assertEqual(result.cluster_count, 2)
        self.assertEqual(result.node_labels.tolist(), [1, 1, 2, 2, 2])
        self.assertGreater(result.normalized_energy_gain, 0)

    def test_three_way_unknown_k(self) -> None:
        graph, _ = _line_graph([0.9, 0.1, 0.9, 0.1, 0.9])
        result = mutex_watershed(graph)
        self.assertEqual(result.cluster_count, 3)
        self.assertEqual(result.node_labels.tolist(), [1, 1, 2, 2, 3, 3])

    def test_lifted_affinity_bridges_a_semantic_gap(self) -> None:
        support = np.zeros((1, 1, 3), dtype=bool)
        support[0, 0, (0, 2)] = True
        affinity = np.full((1, 1, 1, 3), 0.5, dtype=np.float32)
        affinity[0, 0, 0, 0] = 0.95
        graph = build_signed_graph(
            support,
            affinity,
            np.asarray([[0, 0, 2]], dtype=np.int16),
            ContinuityConfig(),
            affinity_kind="probabilities",
        )
        result = mutex_watershed(graph)
        self.assertEqual(result.cluster_count, 1)

    def test_repulsive_edge_wins_an_exact_tie_and_order_is_irrelevant(self) -> None:
        edges = (
            SignedEdge(0, 1, 0.4, False),
            SignedEdge(1, 2, 0.4, False),
            SignedEdge(0, 2, 0.4, True),
        )
        graph = SignedGraph(np.arange(3), edges, (1, 1, 3))
        reversed_graph = SignedGraph(np.arange(3), tuple(reversed(edges)), (1, 1, 3))
        first = mutex_watershed(graph)
        second = mutex_watershed(reversed_graph)
        self.assertEqual(first.cluster_count, 2)
        self.assertTrue(
            np.array_equal(
                first.node_labels[:, None] == first.node_labels[None, :],
                second.node_labels[:, None] == second.node_labels[None, :],
            )
        )
        self.assertNotEqual(first.node_labels[0], first.node_labels[2])

    def test_offset_channel_permutation_does_not_change_partition(self) -> None:
        support = np.ones((1, 1, 4), dtype=bool)
        offsets = np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.int16)
        affinity = np.full((2, 1, 1, 4), 0.5, dtype=np.float32)
        affinity[0, 0, 0, :3] = [0.9, 0.1, 0.9]
        affinity[1, 0, 0, :2] = [0.1, 0.1]
        first = mutex_watershed(
            build_signed_graph(
                support,
                affinity,
                offsets,
                ContinuityConfig(),
                affinity_kind="probabilities",
            )
        )
        second = mutex_watershed(
            build_signed_graph(
                support,
                affinity[::-1],
                offsets[::-1],
                ContinuityConfig(),
                affinity_kind="probabilities",
            )
        )
        self.assertTrue(
            np.array_equal(
                first.node_labels[:, None] == first.node_labels[None, :],
                second.node_labels[:, None] == second.node_labels[None, :],
            )
        )

    def test_edge_weight_is_scaled_by_both_endpoint_confidences(self) -> None:
        support = np.ones((1, 1, 2), dtype=bool)
        affinity = np.full((1, 1, 1, 2), 0.9, dtype=np.float32)
        confidence = np.asarray([0.5, 0.25], dtype=np.float32).reshape(1, 1, 2)
        graph = build_signed_graph(
            support,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            ContinuityConfig(),
            affinity_kind="probabilities",
            node_confidence=confidence,
        )
        self.assertEqual(len(graph.edges), 1)
        self.assertAlmostEqual(graph.edges[0].weight, 0.4 * 0.5 * 0.25, places=6)


class PropagationAndPipelineTests(unittest.TestCase):
    def test_uniform_propagation_preserves_support_and_seed_ownership(self) -> None:
        mask = np.zeros((1, 1, 7), dtype=bool)
        mask[0, 0, 0:5] = True
        mask[0, 0, 6] = True
        seeds = np.zeros(mask.shape, dtype=np.int32)
        seeds[0, 0, 0] = 1
        seeds[0, 0, 4] = 2
        result = propagate_uniform(mask, seeds, spacing_zyx=(1.0, 1.0, 2.0))
        self.assertTrue(np.array_equal(result.labels > 0, mask))
        self.assertEqual(result.labels[0, 0, 0], 1)
        self.assertEqual(result.labels[0, 0, 4], 2)
        self.assertEqual(result.labels[0, 0, 6], 2)
        self.assertEqual(result.unseeded_components, 1)

    def test_end_to_end_split_is_support_preserving_and_ids_are_stable(self) -> None:
        provisional = np.full((1, 1, 6), 7, dtype=np.uint16)
        cortex = np.ones(provisional.shape, dtype=np.float32)
        affinity = np.full((1,) + provisional.shape, 0.5, dtype=np.float32)
        affinity[0, 0, 0, :5] = [0.9, 0.9, 0.1, 0.9, 0.9]
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (1.0, 1.0, 1.0),
            affinity_kind="probabilities",
        )
        self.assertTrue(np.array_equal(result.full_instances > 0, provisional > 0))
        self.assertEqual(set(np.unique(result.full_instances)), {7, 8})
        self.assertEqual(result.full_instances[0, 0, :3].tolist(), [7, 7, 7])
        self.assertEqual(result.full_instances[0, 0, 3:].tolist(), [8, 8, 8])
        self.assertEqual(result.decisions[0].status, "accepted")
        self.assertEqual(
            result.cortical_instances[0, 0].tolist(),
            result.full_instances[0, 0].tolist(),
        )

    def test_no_repulsive_evidence_abstains(self) -> None:
        provisional = np.full((1, 1, 5), 3, dtype=np.uint16)
        cortex = np.zeros(provisional.shape, dtype=np.float32)
        cortex[0, 0, (0, 4)] = 1.0
        affinity = np.full((1,) + provisional.shape, 0.5, dtype=np.float32)
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (1.0, 1.0, 1.0),
            affinity_kind="probabilities",
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].status, "abstained")
        self.assertEqual(
            result.decisions[0].reason, "insufficient_repulsive_evidence"
        )

    def test_pluggable_always_abstain_scorer(self) -> None:
        provisional = np.full((1, 1, 4), 1, dtype=np.uint16)
        cortex = np.ones(provisional.shape, dtype=np.float32)
        affinity = np.full((1,) + provisional.shape, 0.5, dtype=np.float32)
        affinity[0, 0, 0, :3] = [0.9, 0.1, 0.9]
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (1.0, 1.0, 1.0),
            affinity_kind="probabilities",
            scorer=AlwaysAbstainScorer("test_abstention"),
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].reason, "test_abstention")

    def test_graph_resource_gate_abstains_without_changing_the_instance(self) -> None:
        provisional = np.full((1, 1, 4), 5, dtype=np.uint16)
        cortex = np.ones(provisional.shape, dtype=np.float32)
        affinity = np.full((1,) + provisional.shape, 0.9, dtype=np.float32)
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (1.0, 1.0, 1.0),
            affinity_kind="probabilities",
            config=ContinuityConfig(max_graph_nodes=2),
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].status, "abstained")
        self.assertEqual(result.decisions[0].reason, "abstain_resource_limit")
        self.assertIn("max_graph_nodes", result.decisions[0].diagnostic_detail)

    def test_graph_edge_resource_gate_is_checked_before_edge_objects(self) -> None:
        provisional = np.full((1, 1, 4), 5, dtype=np.uint16)
        cortex = np.ones(provisional.shape, dtype=np.float32)
        affinity = np.full((1,) + provisional.shape, 0.9, dtype=np.float32)
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (1.0, 1.0, 1.0),
            affinity_kind="probabilities",
            config=ContinuityConfig(max_graph_edges=1),
        )
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].reason, "abstain_resource_limit")
        self.assertIn("max_graph_edges", result.decisions[0].diagnostic_detail)

    def test_each_instance_is_refined_on_its_exact_bbox(self) -> None:
        provisional = np.zeros((9, 11, 13), dtype=np.uint16)
        provisional[4, 5, 6:10] = 2
        cortex = (provisional > 0).astype(np.float32)
        affinity = np.full((1,) + provisional.shape, 0.5, dtype=np.float32)
        affinity[0, 4, 5, 6:9] = [0.9, 0.1, 0.9]
        from tools.charite_cortical.continuity_postprocess import pipeline

        with patch.object(
            pipeline,
            "build_signed_graph",
            wraps=pipeline.build_signed_graph,
        ) as graph_builder:
            refine(
                provisional,
                cortex,
                affinity,
                np.asarray([[0, 0, 1]], dtype=np.int16),
                (1.0, 1.0, 1.0),
                affinity_kind="probabilities",
            )
        self.assertEqual(graph_builder.call_args.args[0].shape, (1, 1, 4))
        self.assertEqual(graph_builder.call_args.args[1].shape, (1, 1, 1, 4))

    def test_high_confidence_gate_uses_physical_volume(self) -> None:
        provisional = np.full((1, 1, 4), 1, dtype=np.uint16)
        cortex = np.ones(provisional.shape, dtype=np.float32)
        affinity = np.full((1,) + provisional.shape, 0.5, dtype=np.float32)
        affinity[0, 0, 0, :3] = [0.9, 0.1, 0.9]
        result = refine(
            provisional,
            cortex,
            affinity,
            np.asarray([[0, 0, 1]], dtype=np.int16),
            (0.5, 1.0, 1.0),
            affinity_kind="probabilities",
            config=ContinuityConfig(min_high_confidence_volume_mm3=1.5),
        )
        # Each child has two high-confidence voxels, but only 1 mm3 support.
        self.assertTrue(np.array_equal(result.full_instances, provisional))
        self.assertEqual(result.decisions[0].reason, "weak_cortical_cluster")


class ScoringAndIoTests(unittest.TestCase):
    def _features(self, positive: bool, index: int) -> SplitFeatures:
        if positive:
            return SplitFeatures(0.8 + index * 1e-3, 2, 20, 4, 0.3, 0.8, 0)
        return SplitFeatures(0.01 + index * 1e-4, 2, 2, 0, 0.01, 0.1, 2)

    def test_logistic_fit_honours_intact_false_split_limit_and_round_trips(self) -> None:
        features = [self._features(True, i) for i in range(20)]
        features += [self._features(False, i) for i in range(20)]
        labels = np.asarray([True] * 20 + [False] * 20)
        intact = np.asarray([False] * 20 + [True] * 20)
        fitted = fit_logistic_split_scorer(features, labels, intact)
        self.assertEqual(fitted.status, "ok")
        self.assertLessEqual(fitted.intact_false_split_rate, 0.05)
        self.assertGreater(fitted.accepted_true_positives, 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scorer.json"
            save_scorer(path, fitted.scorer)
            loaded = load_scorer(path)
            self.assertEqual(
                loaded.evaluate(features[0]).accepted,
                fitted.scorer.evaluate(features[0]).accepted,
            )

    def test_insufficient_calibration_data_is_always_abstain(self) -> None:
        features = [self._features(True, 0), self._features(False, 0)]
        fitted = fit_logistic_split_scorer(
            features,
            [True, False],
            [False, True],
        )
        self.assertIsInstance(fitted.scorer, AlwaysAbstainScorer)

    def test_input_npz_round_trip_is_explicit(self) -> None:
        provisional = np.ones((1, 2, 3), dtype=np.uint16)
        inputs = ContinuityInputs(
            provisional_instances=provisional,
            cortex_probability=np.ones_like(provisional, dtype=np.float32),
            affinity=np.zeros((1,) + provisional.shape, dtype=np.float32),
            affinity_offsets_zyx=np.asarray([[0, 0, 1]], dtype=np.int16),
            spacing_zyx=np.asarray([1.2, 0.7, 0.7]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            save_inputs_npz(path, inputs)
            loaded = load_inputs_npz(path)
        self.assertTrue(
            np.array_equal(
                loaded.provisional_instances, inputs.provisional_instances
            )
        )
        self.assertEqual(loaded.affinity_kind, "logits")

    def test_oracle_helpers_preserve_support_and_encode_identity(self) -> None:
        fragments = np.asarray([[[1, 1, 2, 2, 3]]], dtype=np.uint16)
        merged = controlled_merge(fragments, (1, 2))
        self.assertTrue(
            np.array_equal(merged.provisional_instances > 0, fragments > 0)
        )
        self.assertEqual(
            merged.provisional_instances.tolist(), [[[1, 1, 1, 1, 3]]]
        )
        logits = oracle_affinity_logits(
            fragments,
            np.asarray([[0, 0, 1]], dtype=np.int16),
        )
        self.assertGreater(logits[0, 0, 0, 0], 0)
        self.assertLess(logits[0, 0, 0, 1], 0)


if __name__ == "__main__":
    unittest.main()
