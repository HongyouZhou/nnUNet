from __future__ import annotations

import unittest

import numpy as np

from nnunetv2.training.cortical_continuity.configure_plans import (
    freeze_cortical_continuity_plans,
)
from nnunetv2.training.cortical_continuity.packing import (
    PREPROCESSED_SOURCE_CHANNELS,
    PackedTargetLayout,
    build_source_segmentation,
    pack_targets_from_augmented_source,
)
from nnunetv2.training.cortical_continuity.sampling import (
    CONTINUITY_CONTACT_KEY,
    CONTINUITY_INSTANCE_KEY_PREFIX,
    CONTINUITY_SUPPORT_KEY,
    available_sampling_distribution,
)
from nnunetv2.training.cortical_continuity.schema import (
    SCHEMA_PLANS_KEY,
    build_cortical_continuity_schema,
)
from nnunetv2.training.cortical_continuity.sidecar_contract import (
    assert_exact_grid,
    assert_instance_retention,
)
from nnunetv2.training.cortical_continuity.trainer import (
    FORMAL_INITIAL_LR,
    FORMAL_ITERATIONS_PER_EPOCH,
    FORMAL_NUM_EPOCHS,
)
from nnunetv2.training.nnUNetTrainer.variants.charite_cortical.nnUNetTrainerCorticalContinuity import (
    nnUNetTrainerCorticalContinuity,
)


class SourcePackingTests(unittest.TestCase):
    def test_relation_below_native_through_plane_resolution_is_ignored(self) -> None:
        schema = build_cortical_continuity_schema((0.5, 0.5, 0.5))
        source = np.zeros(
            (len(PREPROCESSED_SOURCE_CHANNELS), 3, 2, 2),
            dtype=np.int16,
        )
        source[0, :2, 0, 0] = 1
        source[1, :2, 0, 0] = 1
        source[3] = 3
        source[4] = 1
        source[5] = 2  # native z spacing requires at least two 0.5-mm steps
        source[6] = 1
        source[7] = 1
        packed = pack_targets_from_augmented_source(source, schema)
        z_channel = next(
            index
            for index, offset in enumerate(schema.affinity_offsets)
            if offset.voxel_offset_zyx == (1, 0, 0)
        )
        affinity_valid_start = 2 + schema.num_affinity_channels
        self.assertEqual(
            int(packed[affinity_valid_start + z_channel].sum()),
            0,
        )

    def test_semantic_ignore_is_not_cortex(self) -> None:
        semantic = np.asarray([[[0, 1, 2, 3]]], dtype=np.uint8)
        instance = np.asarray([[[0, 1, 0, 0]]], dtype=np.int16)
        overlap = np.zeros_like(semantic)
        valid = np.full_like(semantic, 3)
        support = np.asarray([[[0, 1, 1, 0]]], dtype=np.uint8)
        source = build_source_segmentation(semantic, instance, overlap, valid, support)
        np.testing.assert_array_equal(source[0], np.asarray([[[0, 1, 1, 0]]]))

    def test_packed_layout_and_overlap_mask(self) -> None:
        schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))
        source = np.zeros((5, 4, 4, 4), dtype=np.int16)
        source[0, 1, 1, 1:3] = 1
        source[1, 1, 1, 1:3] = 4
        source[2, 1, 1, 2] = 1
        source[3] = 3
        source[4] = 1
        packed = pack_targets_from_augmented_source(source, schema)
        self.assertEqual(
            packed.shape[0],
            PackedTargetLayout.from_schema(schema).total_channels,
        )
        channel = next(
            index
            for index, offset in enumerate(schema.affinity_offsets)
            if offset.voxel_offset_zyx == (0, 0, 1)
        )
        affinity_valid_start = 2 + schema.num_affinity_channels
        self.assertEqual(packed[affinity_valid_start + channel, 1, 1, 1], 0)


class GridContractTests(unittest.TestCase):
    def test_thin_cortex_retention_is_a_hard_gate(self) -> None:
        source = np.asarray([[[0, 1, 2]]], dtype=np.int16)
        report = assert_instance_retention(source, source.copy())
        self.assertEqual(report["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "retention gate failed"):
            assert_instance_retention(
                source,
                np.asarray([[[0, 1, 0]]], dtype=np.int16),
            )

    def test_nibabel_affines_are_accepted_and_compared(self) -> None:
        image = np.zeros((1, 2, 3, 4), dtype=np.float32)
        affine = np.eye(4)
        properties = {
            "spacing": [0.5, 0.5, 0.5],
            "nibabel_stuff": {
                "original_affine": affine,
                "reoriented_affine": affine,
            },
        }
        assert_exact_grid(
            image,
            properties,
            image.copy(),
            properties,
            role="validity",
            path="validMasksTr/case.nii.gz",
        )
        changed = {
            **properties,
            "nibabel_stuff": {
                **properties["nibabel_stuff"],
                "reoriented_affine": affine.copy(),
            },
        }
        changed["nibabel_stuff"]["reoriented_affine"][0, 3] = 1.0
        with self.assertRaisesRegex(ValueError, "reoriented_affine"):
            assert_exact_grid(
                image,
                properties,
                image.copy(),
                changed,
                role="validity",
                path="validMasksTr/case.nii.gz",
            )

    def test_simpleitk_geometry_is_supported(self) -> None:
        image = np.zeros((1, 2, 3, 4), dtype=np.float32)
        properties = {
            "spacing": [0.5, 0.5, 0.5],
            "sitk_stuff": {
                "spacing": (0.5, 0.5, 0.5),
                "origin": (0.0, 0.0, 0.0),
                "direction": tuple(np.eye(3).ravel()),
            },
        }
        assert_exact_grid(
            image, properties, image.copy(), properties, role="support", path="support.nii.gz"
        )


class SamplingAndPlansTests(unittest.TestCase):
    def test_sampling_weights_and_missing_category_renormalization(self) -> None:
        point = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        full = {
            CONTINUITY_CONTACT_KEY: point,
            (CONTINUITY_INSTANCE_KEY_PREFIX, 1): point,
            CONTINUITY_SUPPORT_KEY: point,
        }
        self.assertEqual(
            available_sampling_distribution(full),
            {"contact": 0.4, "instance": 0.3, "support": 0.2, "random": 0.1},
        )
        missing = {
            CONTINUITY_CONTACT_KEY: np.empty((0, 4), dtype=np.int64),
            (CONTINUITY_INSTANCE_KEY_PREFIX, 1): point,
            CONTINUITY_SUPPORT_KEY: np.empty((0, 4), dtype=np.int64),
        }
        distribution = available_sampling_distribution(missing)
        self.assertAlmostEqual(distribution["instance"], 0.75)
        self.assertAlmostEqual(distribution["random"], 0.25)

    def test_plans_freeze_and_formal_schedule(self) -> None:
        plans = {
            "configurations": {
                "3d_fullres": {
                    "spacing": [0.5, 0.5, 0.5],
                    "preprocessor_name": "ChariteCorticalPreprocessor",
                }
            }
        }
        frozen = freeze_cortical_continuity_plans(plans)
        self.assertEqual(frozen[SCHEMA_PLANS_KEY]["heads"][-1]["stop"], 20)
        self.assertEqual(FORMAL_NUM_EPOCHS, 500)
        self.assertEqual(FORMAL_ITERATIONS_PER_EPOCH, 250)
        self.assertEqual(FORMAL_INITIAL_LR, 1e-3)
        self.assertFalse(frozen["cortical_continuity_training_schedule"]["deep_supervision"])

    def test_trainer_wrapper_is_importable_by_declared_name(self) -> None:
        self.assertEqual(
            nnUNetTrainerCorticalContinuity.__name__,
            "nnUNetTrainerCorticalContinuity",
        )


if __name__ == "__main__":
    unittest.main()
