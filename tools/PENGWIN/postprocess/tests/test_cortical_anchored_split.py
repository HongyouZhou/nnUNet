from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as nd_label

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    MinCutResult,
    SplitConfig,
    _high_hu_cut_discount,
    _narrow_band_bottleneck_mincut,
    _repair_terminal_connectivity,
    build_core_anchors,
    build_cost_field,
    cortical_anchored_split,
    split_instance_core_first,
    split_instance_with_seeds,
)
from tools.PENGWIN.postprocess.run_cortical_split import main as cli_main


class ManualMinCutSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (24, 20, 20)
        self.instance_mask = np.zeros(self.shape, dtype=bool)
        self.instance_mask[2:22, 2:18, 2:18] = True

        self.instances = np.zeros(self.shape, dtype=np.int32)
        self.instances[self.instance_mask] = 7
        self.abbc = np.zeros(self.shape, dtype=np.int16)
        self.abbc[self.instance_mask] = 2

        self.prob3 = np.zeros(self.shape, dtype=np.float32)
        self.prob3[11:13, 2:18, 2:18] = 1.0

        self.source_seed = np.zeros(self.shape, dtype=bool)
        self.source_seed[4:8, 5:15, 5:15] = True
        self.sink_seed = np.zeros(self.shape, dtype=bool)
        self.sink_seed[16:20, 5:15, 5:15] = True

        self.cfg = SplitConfig(
            w_prob1=0.0,
            w_prob2=1.0,
            w_prob3=4.0,
            w_min=0.05,
            min_split_piece_size=100,
        )

    def test_standalone_label3_probability_lowers_cut_cost(self) -> None:
        cost = build_cost_field(
            self.abbc,
            None,
            self.instance_mask,
            self.cfg,
            prob_label_3=self.prob3,
        )

        self.assertAlmostEqual(float(cost[6, 10, 10]), 1.05, places=5)
        self.assertAlmostEqual(float(cost[11, 10, 10]), 0.05, places=5)
        self.assertEqual(float(cost[0, 0, 0]), self.cfg.w_bg_outside)

    def test_manual_split_follows_label3_plane_without_losing_voxels(self) -> None:
        result, cut, diagnostics = split_instance_with_seeds(
            abbc_pred=self.abbc,
            instances=self.instances,
            instance_id=7,
            source_seed_mask=self.source_seed,
            sink_seed_mask=self.sink_seed,
            prob_label_3=self.prob3,
            cfg=self.cfg,
        )

        self.assertEqual(set(np.unique(result)), {0, 7, 8})
        self.assertTrue(np.all(result[self.source_seed] == 7))
        self.assertTrue(np.all(result[self.sink_seed] == 8))
        self.assertEqual(np.count_nonzero(result), np.count_nonzero(self.instances))
        self.assertTrue(np.array_equal(result != 0, self.instances != 0))

        cut_z = np.argwhere(cut)[:, 0]
        self.assertGreater(cut_z.size, 0)
        self.assertGreaterEqual(float(cut_z.mean()), 10.5)
        self.assertLessEqual(float(cut_z.mean()), 12.5)
        self.assertGreater(diagnostics["mean_prob_label_3_on_cut"], 0.95)
        self.assertEqual(diagnostics["new_instance"], 8)

    def test_automatic_core_seed_split_does_not_leave_an_id_bridge(self) -> None:
        abbc = self.abbc.copy()
        abbc[11:13, 2:18, 2:18] = 3
        abbc[3, 3, 3] = 1
        cfg = SplitConfig(
            min_cortical_voxels=1,
            min_instance_voxels=1,
            w_prob1=0.0,
            w_prob2=1.0,
            w_prob3=4.0,
            w_min=0.05,
            min_split_piece_size=100,
        )

        result, cut, diagnostics = cortical_anchored_split(
            abbc,
            self.instances,
            prob_label_3=self.prob3,
            cfg=cfg,
        )

        self.assertEqual(set(np.unique(result)), {0, 7, 8})
        self.assertTrue(np.array_equal(result != 0, self.instances != 0))
        self.assertGreater(int(cut.sum()), 0)
        self.assertEqual(diagnostics["per_instance"][0]["splits"], 1)

    def test_rejects_overlapping_manual_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            split_instance_with_seeds(
                self.abbc,
                self.instances,
                7,
                self.source_seed,
                self.source_seed,
                prob_label_3=self.prob3,
                cfg=self.cfg,
            )

    def test_rejects_seed_outside_selected_instance(self) -> None:
        invalid_source = self.source_seed.copy()
        invalid_source[0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "completely inside"):
            split_instance_with_seeds(
                self.abbc,
                self.instances,
                7,
                invalid_source,
                self.sink_seed,
                prob_label_3=self.prob3,
                cfg=self.cfg,
            )

    def test_manual_cli_round_trip_preserves_nifti_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "abbc": root / "abbc.nii.gz",
                "instances": root / "instances.nii.gz",
                "prob3": root / "prob3.nii.gz",
                "source": root / "source.nii.gz",
                "sink": root / "sink.nii.gz",
            }
            arrays = {
                "abbc": self.abbc,
                "instances": self.instances,
                "prob3": self.prob3,
                "source": self.source_seed.astype(np.uint8),
                "sink": self.sink_seed.astype(np.uint8),
            }
            reference_spacing = (0.6, 0.7, 1.2)
            for name, array in arrays.items():
                image = sitk.GetImageFromArray(array)
                image.SetSpacing(reference_spacing)
                sitk.WriteImage(image, str(paths[name]))

            outdir = root / "out"
            argv = [
                "run_cortical_split.py",
                "--abbc-pred",
                str(paths["abbc"]),
                "--instances",
                str(paths["instances"]),
                "--prob-label-3",
                str(paths["prob3"]),
                "--instance-id",
                "7",
                "--source-seed",
                str(paths["source"]),
                "--sink-seed",
                str(paths["sink"]),
                "--min-split-piece-size",
                "100",
                "--outdir",
                str(outdir),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                cli_main()

            output_image = sitk.ReadImage(str(outdir / "instances_split.nii.gz"))
            output = sitk.GetArrayFromImage(output_image)
            self.assertEqual(set(np.unique(output)), {0, 7, 8})
            self.assertTrue(np.allclose(output_image.GetSpacing(), reference_spacing))
            with (outdir / "diagnostics.json").open(encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            self.assertEqual(diagnostics["mode"], "core_first_two_seed")


class HighHuCutPreferenceTests(unittest.TestCase):
    def test_disabled_high_hu_preference_is_an_exact_no_op(self) -> None:
        cfg = SplitConfig(bottleneck_high_hu_cut_weight=0.0)
        current_hu = np.asarray([100.0, 1_000.0], dtype=np.float32)
        following_hu = current_hu.copy()
        cortical_probability = np.zeros(2, dtype=np.float32)

        discount, score = _high_hu_cut_discount(
            current_hu,
            following_hu,
            cortical_probability,
            cfg,
        )

        self.assertTrue(np.array_equal(discount, np.ones(2, dtype=np.float32)))
        self.assertTrue(np.array_equal(score, np.zeros(2, dtype=np.float32)))

    def test_cortical_protection_blocks_the_high_hu_discount(self) -> None:
        cfg = SplitConfig(
            bottleneck_high_hu_cut_weight=2.0,
            bottleneck_high_hu_center_hu=500.0,
            bottleneck_high_hu_scale_hu=100.0,
            bottleneck_high_hu_protect_cortical=True,
        )
        high_hu = np.asarray([1_000.0], dtype=np.float32)
        cortical_probability = np.ones(1, dtype=np.float32)

        protected_discount, protected_score = _high_hu_cut_discount(
            high_hu,
            high_hu,
            cortical_probability,
            cfg,
        )
        unprotected_discount, unprotected_score = _high_hu_cut_discount(
            high_hu,
            high_hu,
            cortical_probability,
            dataclasses.replace(cfg, bottleneck_high_hu_protect_cortical=False),
        )

        self.assertEqual(float(protected_discount[0]), 1.0)
        self.assertEqual(float(protected_score[0]), 0.0)
        self.assertLess(float(unprotected_discount[0]), 0.2)
        self.assertGreater(float(unprotected_score[0]), 0.99)

    def test_high_hu_preference_moves_the_cut_to_a_dense_plane(self) -> None:
        shape = (8, 8, 24)
        instance_mask = np.ones(shape, dtype=bool)
        initial_partition = np.zeros(shape, dtype=np.uint8)
        initial_partition[:, :, :12] = 1
        initial_partition[:, :, 12:] = 2
        source_seed = np.zeros(shape, dtype=bool)
        sink_seed = np.zeros(shape, dtype=bool)
        source_seed[:, :, 1] = True
        sink_seed[:, :, 22] = True
        x_coordinates = np.arange(shape[2], dtype=np.float64)[None, None, :]
        source_distance = np.broadcast_to(
            np.abs(x_coordinates - 1),
            shape,
        ).copy()
        sink_distance = np.broadcast_to(
            np.abs(x_coordinates - 22),
            shape,
        ).copy()
        image = np.full(shape, 100.0, dtype=np.float32)
        image[:, :, 16:18] = 1_000.0
        probabilities = np.zeros(shape, dtype=np.float32)
        cfg = SplitConfig(
            bottleneck_hu_sigma=1e9,
            bottleneck_unary_weight=0.03,
            bottleneck_voxel_graph_max_nodes=100_000,
            bottleneck_high_hu_center_hu=500.0,
            bottleneck_high_hu_scale_hu=150.0,
        )

        baseline, baseline_diagnostics = _narrow_band_bottleneck_mincut(
            instance_mask,
            initial_partition,
            instance_mask,
            source_seed,
            sink_seed,
            source_distance,
            sink_distance,
            image,
            probabilities,
            probabilities,
            probabilities,
            (1.0, 1.0, 1.0),
            cfg,
        )
        preferred, preferred_diagnostics = _narrow_band_bottleneck_mincut(
            instance_mask,
            initial_partition,
            instance_mask,
            source_seed,
            sink_seed,
            source_distance,
            sink_distance,
            image,
            probabilities,
            probabilities,
            probabilities,
            (1.0, 1.0, 1.0),
            dataclasses.replace(cfg, bottleneck_high_hu_cut_weight=1.0),
        )

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(preferred)
        baseline_x = np.argwhere(baseline.cut_mask)[:, 2]
        preferred_x = np.argwhere(preferred.cut_mask)[:, 2]
        self.assertEqual(float(np.median(baseline_x)), 11.5)
        self.assertEqual(float(np.median(preferred_x)), 16.5)
        self.assertFalse(baseline_diagnostics["bottleneck_high_hu_cut_applied"])
        self.assertTrue(preferred_diagnostics["bottleneck_high_hu_cut_applied"])
        self.assertLess(
            preferred_diagnostics["bottleneck_high_hu_mean_capacity_multiplier"],
            1.0,
        )


class CoreFirstSplitTests(unittest.TestCase):
    def _two_lobes(self, core_bridge: bool = False) -> tuple[np.ndarray, ...]:
        shape = (28, 28, 44)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[3:25, 3:25, 2:18] = True
        instance_mask[3:25, 3:25, 26:42] = True
        instance_mask[12:16, 12:16, 18:26] = True

        instances = np.zeros(shape, dtype=np.int32)
        instances[instance_mask] = 7
        abbc = np.zeros(shape, dtype=np.int16)
        abbc[instance_mask] = 1
        abbc[6:22, 6:22, 5:18] = 2
        abbc[6:22, 6:22, 26:39] = 2
        if core_bridge:
            abbc[13, 13, 18:26] = 2

        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[10:14, 10:14, 9:13] = True
        sink[10:14, 10:14, 31:35] = True
        return abbc, instances, source, sink, instance_mask

    def _config(self) -> SplitConfig:
        return SplitConfig(
            core_anchor_erosion_mm=1.0,
            core_anchor_min_voxels=20,
            w_prob1=6.0,
            w_prob2=4.0,
            w_prob3=4.0,
            min_split_piece_size=100,
        )

    def test_topology_cleanup_reassigns_an_unanchored_source_island(self) -> None:
        shape = (9, 9, 16)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[1:8, 1:8, 1:15] = True
        instance_mask[0, 0, 0] = True
        source_mask = instance_mask.copy()
        source_mask[:, :, 8:] = False
        sink_mask = instance_mask & ~source_mask
        floating_source = (4, 4, 11)
        source_mask[floating_source] = True
        sink_mask[floating_source] = False
        source_terminal = np.zeros(shape, dtype=bool)
        sink_terminal = np.zeros(shape, dtype=bool)
        source_terminal[4, 4, 3] = True
        sink_terminal[4, 4, 12] = True
        initial = MinCutResult(
            source_mask,
            sink_mask,
            np.zeros(shape, dtype=bool),
            1.0,
        )

        repaired, diagnostics = _repair_terminal_connectivity(
            initial,
            instance_mask,
            source_terminal,
            sink_terminal,
        )

        self.assertFalse(repaired.source_mask[floating_source])
        self.assertTrue(repaired.sink_mask[floating_source])
        self.assertTrue(repaired.source_mask[0, 0, 0])
        self.assertTrue(
            np.array_equal(repaired.source_mask | repaired.sink_mask, instance_mask)
        )
        self.assertEqual(
            diagnostics["source_floating_voxels_reassigned_to_sink"],
            1,
        )
        self.assertEqual(diagnostics["source_unanchored_voxels_remaining"], 1)

    def test_separate_cores_ignore_a_cortical_bridge(self) -> None:
        abbc, instances, source, sink, instance_mask = self._two_lobes()

        result, cut, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            cfg=self._config(),
        )

        self.assertEqual(diagnostics["core_partition_method"], "separate_core_components")
        self.assertEqual(set(np.unique(result)), {0, 7, 8})
        self.assertTrue(np.array_equal(result != 0, instance_mask))
        self.assertTrue(np.all(result[source] == 7))
        self.assertTrue(np.all(result[sink] == 8))
        self.assertEqual(nd_label(result == 7)[1], 1)
        self.assertEqual(nd_label(result == 8)[1], 1)
        self.assertGreater(int(cut.sum()), 0)
        self.assertEqual(
            diagnostics["peripheral_partition_method"],
            "geodesic_bottleneck_mincut",
        )
        self.assertEqual(
            diagnostics["bottleneck_growth_seed_method"],
            "separate_core_components",
        )

    def test_soft_fracture_signal_enables_graph_for_a_small_roi(self) -> None:
        abbc, instances, source, sink, _ = self._two_lobes()
        prob3 = np.zeros(abbc.shape, dtype=np.float32)
        cfg = self._config()
        cfg.use_geodesic_bottleneck = False

        result, _, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            prob_label_3=prob3,
            cfg=cfg,
        )

        self.assertEqual(diagnostics["peripheral_partition_method"], "min_cut")
        self.assertTrue(np.all(result[source] == 7))
        self.assertTrue(np.all(result[sink] == 8))

    def test_cortical_clicks_remain_in_their_requested_outputs(self) -> None:
        abbc, instances, _, _, _ = self._two_lobes()
        source = np.zeros(abbc.shape, dtype=bool)
        sink = np.zeros(abbc.shape, dtype=bool)
        source[4, 4, 3] = True
        sink[4, 4, 40] = True

        result, _, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            cfg=self._config(),
        )

        self.assertEqual(
            diagnostics["peripheral_partition_method"],
            "geodesic_bottleneck_mincut",
        )
        self.assertEqual(int(result[4, 4, 3]), 7)
        self.assertEqual(int(result[4, 4, 40]), 8)

    def test_cortical_interactions_project_to_nearest_core(self) -> None:
        abbc, _, _, _, instance_mask = self._two_lobes()
        source = np.zeros(instance_mask.shape, dtype=bool)
        sink = np.zeros(instance_mask.shape, dtype=bool)
        source[4, 4, 3] = True
        sink[4, 4, 40] = True

        anchors = build_core_anchors(
            abbc,
            instance_mask,
            source,
            sink,
            spacing_zyx=(1.0, 1.0, 1.0),
            cfg=self._config(),
        )

        self.assertEqual(anchors.diagnostics["core_partition_method"], "separate_core_components")
        self.assertGreater(anchors.diagnostics["source_projection_distance_mm"], 0.0)
        self.assertGreater(anchors.diagnostics["sink_projection_distance_mm"], 0.0)

    def test_two_interactions_can_enforce_distinct_nearby_core_components(self) -> None:
        abbc, _, source, _, instance_mask = self._two_lobes()
        sink = np.zeros(instance_mask.shape, dtype=bool)
        sink[13, 13, 20] = True
        abbc[13, 13, 20] = 2
        cfg = self._config()
        cfg.core_anchor_erosion_mm = 0.0

        anchors = build_core_anchors(
            abbc,
            instance_mask,
            source,
            sink,
            cfg=cfg,
        )

        self.assertEqual(anchors.diagnostics["core_partition_method"], "separate_core_components")
        self.assertEqual(anchors.diagnostics["distinct_component_reassignment"], "sink")
        self.assertEqual(anchors.diagnostics["ignored_small_core_components"], 1)
        self.assertNotEqual(
            anchors.diagnostics["source_core_component"],
            anchors.diagnostics["sink_core_component"],
        )

    def test_core_erosion_breaks_a_thin_core_bridge(self) -> None:
        abbc, instances, source, sink, instance_mask = self._two_lobes(core_bridge=True)
        raw_core = (abbc == 2) & instance_mask
        self.assertEqual(nd_label(raw_core)[1], 1)

        anchors = build_core_anchors(
            abbc,
            instance_mask,
            source,
            sink,
            cfg=self._config(),
        )

        self.assertEqual(anchors.diagnostics["core_partition_method"], "separate_core_components")
        self.assertGreaterEqual(anchors.diagnostics["core_component_count"], 2)
        self.assertFalse(np.any(anchors.source_mask & anchors.sink_mask))

    def test_shared_core_is_partitioned_by_prompts_before_full_instance_cut(self) -> None:
        shape = (28, 28, 36)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[2:26, 2:26, 2:34] = True
        instances = np.zeros(shape, dtype=np.int32)
        instances[instance_mask] = 4
        abbc = np.zeros(shape, dtype=np.int16)
        abbc[instance_mask] = 1
        abbc[5:23, 5:23, 5:31] = 2
        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[10:13, 10:13, 9:12] = True
        sink[15:18, 15:18, 24:27] = True

        anchors = build_core_anchors(
            abbc,
            instance_mask,
            source,
            sink,
            cfg=self._config(),
        )
        result, _, diagnostics = split_instance_core_first(
            abbc,
            instances,
            4,
            source,
            sink,
            cfg=self._config(),
        )

        self.assertEqual(
            anchors.diagnostics["core_partition_method"],
            "shared_core_depth_watershed",
        )
        self.assertTrue(
            np.array_equal(anchors.source_mask | anchors.sink_mask, anchors.robust_core_mask)
        )
        self.assertGreater(int(anchors.source_mask.sum()), 2_000)
        self.assertGreater(int(anchors.sink_mask.sum()), 2_000)
        self.assertTrue(np.all(result[source] == 4))
        self.assertTrue(np.all(result[sink] == 5))
        self.assertGreater(int(np.count_nonzero(result == 4)), 2_000)
        self.assertGreater(int(np.count_nonzero(result == 5)), 2_000)
        self.assertEqual(
            diagnostics["core_partition_method"],
            "shared_core_depth_watershed",
        )
        self.assertEqual(
            diagnostics["bottleneck_growth_seed_method"],
            "shared_core_interior_basins",
        )
        self.assertEqual(
            diagnostics["peripheral_partition_method"],
            "geodesic_bottleneck_mincut",
        )

    def test_shared_core_depth_watershed_splits_an_unequal_dumbbell_at_its_neck(
        self,
    ) -> None:
        shape = (24, 32, 58)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[3:21, 3:29, 2:32] = True
        instance_mask[7:17, 8:24, 39:55] = True
        instance_mask[10:14, 13:19, 32:39] = True
        abbc = np.zeros(shape, dtype=np.uint8)
        abbc[instance_mask] = 2
        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[10:14, 14:18, 12:16] = True
        sink[10:14, 14:18, 45:49] = True
        cfg = self._config()
        cfg.core_anchor_erosion_mm = 0.0

        anchors = build_core_anchors(
            abbc,
            instance_mask,
            source,
            sink,
            spacing_zyx=(1.2, 0.8, 0.8),
            cfg=cfg,
        )

        self.assertEqual(
            anchors.diagnostics["core_partition_method"],
            "shared_core_depth_watershed",
        )
        self.assertTrue(
            np.array_equal(
                anchors.source_mask | anchors.sink_mask,
                instance_mask,
            )
        )
        self.assertTrue(np.all(anchors.source_mask[source]))
        self.assertTrue(np.all(anchors.sink_mask[sink]))
        self.assertTrue(np.all(anchors.source_mask[3:21, 3:29, 2:28]))
        self.assertTrue(np.all(anchors.sink_mask[7:17, 8:24, 42:55]))
        self.assertLess(int(anchors.sink_mask.sum()), int(anchors.source_mask.sum()))

    def test_bottleneck_cut_follows_a_curved_ct_discontinuity(self) -> None:
        shape = (22, 30, 54)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[2:20, 3:27, 2:52] = True
        instances = np.zeros(shape, dtype=np.int32)
        instances[instance_mask] = 4
        abbc = np.zeros(shape, dtype=np.int16)
        abbc[instance_mask] = 2

        zz, yy, xx = np.indices(shape)
        curved_surface = 27.0 + 3.5 * np.sin((yy - 3.0) * np.pi / 12.0)
        image = np.full(shape, -200.0, dtype=np.float32)
        image[instance_mask & (xx < curved_surface)] = 150.0
        image[instance_mask & (xx >= curved_surface)] = 850.0
        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[8:14, 11:17, 8:12] = True
        sink[8:14, 11:17, 42:46] = True
        cfg = self._config()
        cfg.bottleneck_supervoxel_target_voxels = 100_000
        cfg.bottleneck_band_width_mm = 8.0
        cfg.bottleneck_core_terminal_margin_mm = 4.0
        cfg.bottleneck_max_full_feature_voxels = 0

        result, cut, diagnostics = split_instance_core_first(
            abbc,
            instances,
            4,
            source,
            sink,
            image=image,
            cfg=cfg,
        )

        cut_coordinates = np.argwhere(cut)
        expected_x = 27.0 + 3.5 * np.sin(
            (cut_coordinates[:, 1] - 3.0) * np.pi / 12.0
        )
        absolute_error = np.abs(cut_coordinates[:, 2] - expected_x)
        self.assertEqual(
            diagnostics["peripheral_partition_method"],
            "geodesic_bottleneck_mincut",
        )
        self.assertEqual(diagnostics["bottleneck_graph_representation"], "voxel")
        self.assertFalse(diagnostics["bottleneck_full_ct_features"])
        self.assertFalse(diagnostics["bottleneck_geodesic_ct_gradient_used"])
        self.assertGreater(diagnostics["bottleneck_graph_crop_voxels"], 0)
        self.assertLessEqual(
            diagnostics["bottleneck_graph_crop_voxels"],
            int(np.prod(shape)),
        )
        self.assertLess(float(np.median(absolute_error)), 2.0)
        self.assertGreater(float(np.std(cut_coordinates[:, 2])), 1.5)
        self.assertTrue(np.all(result[source] == 4))
        self.assertTrue(np.all(result[sink] == 5))
        self.assertTrue(np.array_equal(result != 0, instance_mask))

    def test_coarse_to_fine_recovers_a_bottleneck_outside_the_fine_band(
        self,
    ) -> None:
        shape = (20, 28, 64)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[2:18, 3:25, 2:62] = True
        instances = np.zeros(shape, dtype=np.int32)
        instances[instance_mask] = 4
        abbc = np.zeros(shape, dtype=np.uint8)
        abbc[instance_mask] = 1

        _, _, xx = np.indices(shape)
        image = np.full(shape, -200.0, dtype=np.float32)
        image[instance_mask & (xx < 20)] = 200.0
        image[instance_mask & (xx >= 20)] = 900.0
        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[8:12, 11:17, 7:11] = True
        sink[8:12, 11:17, 53:57] = True
        cfg = dataclasses.replace(
            self._config(),
            bottleneck_use_core_growth_seeds=False,
            bottleneck_band_width_mm=1.0,
            bottleneck_coarse_to_fine=True,
            bottleneck_max_full_feature_voxels=0,
            bottleneck_supervoxel_target_voxels=100_000,
        )

        _, legacy_cut, legacy_diagnostics = split_instance_core_first(
            abbc,
            instances,
            4,
            source,
            sink,
            image=image,
            cfg=dataclasses.replace(cfg, bottleneck_coarse_to_fine=False),
        )
        result, adaptive_cut, diagnostics = split_instance_core_first(
            abbc,
            instances,
            4,
            source,
            sink,
            image=image,
            cfg=cfg,
        )

        legacy_x = np.argwhere(legacy_cut)[:, 2]
        adaptive_x = np.argwhere(adaptive_cut)[:, 2]
        self.assertGreater(float(np.median(legacy_x)), 30.0)
        self.assertLess(abs(float(np.median(adaptive_x)) - 19.5), 1.0)
        self.assertFalse(legacy_diagnostics["bottleneck_coarse_to_fine_attempted"])
        self.assertTrue(diagnostics["bottleneck_coarse_to_fine_used"])
        self.assertEqual(diagnostics["bottleneck_coarse_attempts"], 2)
        self.assertEqual(diagnostics["bottleneck_coarse_expansions"], 1)
        self.assertEqual(diagnostics["bottleneck_refinement_stages"], 2)
        self.assertEqual(diagnostics["bottleneck_fine_graph_representation"], "voxel")
        self.assertTrue(np.all(result[source] == 4))
        self.assertTrue(np.all(result[sink] == 5))
        self.assertTrue(np.array_equal(result != 0, instance_mask))

    def test_bottleneck_partition_falls_back_to_prompts_without_label_2(self) -> None:
        abbc, instances, source, sink, _ = self._two_lobes()
        abbc[abbc == 2] = 1
        result, _, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            cfg=self._config(),
        )

        self.assertEqual(diagnostics["bottleneck_growth_seed_method"], "prompt_only")
        self.assertIn("no Label-2 core", diagnostics["bottleneck_core_seed_fallback_reason"])
        self.assertTrue(np.all(result[source] == 7))
        self.assertTrue(np.all(result[sink] == 8))
        self.assertTrue(np.array_equal(result != 0, instances != 0))

    def test_large_roi_uses_core_seeded_watershed_without_losing_voxels(self) -> None:
        abbc, instances, source, sink, instance_mask = self._two_lobes()
        cfg = self._config()
        cfg.core_first_max_graph_voxels = 0
        cfg.use_geodesic_bottleneck = False

        result, cut, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            cfg=cfg,
        )

        self.assertEqual(diagnostics["peripheral_partition_method"], "marker_watershed")
        self.assertTrue(np.array_equal(result != 0, instance_mask))
        self.assertTrue(np.all(result[source] == 7))
        self.assertTrue(np.all(result[sink] == 8))
        self.assertGreater(int(cut.sum()), 0)

    def test_disconnected_instance_components_keep_exact_input_support(self) -> None:
        shape = (24, 24, 60)
        instance_mask = np.zeros(shape, dtype=bool)
        instance_mask[3:21, 3:21, 2:16] = True
        instance_mask[3:21, 3:21, 20:24] = True
        instance_mask[3:21, 3:21, 42:58] = True
        instances = np.zeros(shape, dtype=np.int32)
        instances[instance_mask] = 7

        abbc = np.zeros(shape, dtype=np.int16)
        abbc[instance_mask] = 1
        abbc[6:18, 6:18, 5:13] = 2
        abbc[6:18, 6:18, 46:54] = 2
        abbc[0, 0, 0] = 3

        source = np.zeros(shape, dtype=bool)
        sink = np.zeros(shape, dtype=bool)
        source[9:13, 9:13, 7:11] = True
        sink[9:13, 9:13, 48:52] = True
        cfg = self._config()
        cfg.core_first_max_graph_voxels = 0

        result, cut, diagnostics = split_instance_core_first(
            abbc,
            instances,
            7,
            source,
            sink,
            spacing_zyx=(1.5, 0.75, 0.5),
            cfg=cfg,
        )

        self.assertTrue(np.array_equal(result != 0, instance_mask))
        self.assertEqual(int(result[0, 0, 0]), 0)
        self.assertTrue(np.all(result[3:21, 3:21, 20:24] == 7))
        self.assertTrue(np.all(result[sink] == 8))
        self.assertEqual(diagnostics["unseeded_component_count"], 1)
        self.assertEqual(diagnostics["unseeded_components_assigned_source"], 1)
        self.assertEqual(diagnostics["unseeded_components_assigned_sink"], 0)
        self.assertEqual(int(cut.sum()), 0)


if __name__ == "__main__":
    unittest.main()
