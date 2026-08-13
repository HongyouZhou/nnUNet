from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from tools.PENGWIN.postprocess.assign_fragments_to_bones import (
    AssignmentConfig,
    assign_from_overlap_counts,
    build_named_segmentations,
    build_assignment_report,
    compute_array_overlap_counts,
    export_named_segmentations,
    main,
)
from tools.PENGWIN.postprocess.assign_fragments_to_bones_batch import (
    process_assignment_directories,
)


class FragmentBoneAssignmentTests(unittest.TestCase):
    def test_builds_komo_named_masks_and_bone_aware_aggregate(self) -> None:
        instances = np.zeros((6, 4, 3), dtype=np.uint16)
        instances[0:3] = 7
        instances[3:5] = 2
        instances[5:6] = 9
        assignments = {
            "7": {"bone": "tibia"},
            "2": {"bone": "tibia"},
            "9": {"bone": "fibula"},
        }

        masks, aggregate = build_named_segmentations(instances, assignments, "L")

        self.assertEqual(
            set(masks),
            {"tibia_L.nii.gz", "tibia_L_fragment_1.nii.gz", "fibula_L.nii.gz"},
        )
        self.assertEqual(set(np.unique(masks["tibia_L.nii.gz"])), {0, 255})
        self.assertEqual(set(np.unique(aggregate)), {1, 2, 21})
        self.assertTrue(np.all(aggregate[instances == 7] == 1))
        self.assertTrue(np.all(aggregate[instances == 2] == 2))
        self.assertTrue(np.all(aggregate[instances == 9] == 21))

    def test_named_export_rejects_unknown_bones(self) -> None:
        with self.assertRaisesRegex(ValueError, "no usable bone name"):
            build_named_segmentations(
                np.ones((2, 2, 2), dtype=np.uint8),
                {"1": {"bone": "unknown"}},
                "L",
            )

    def test_multiple_fragments_receive_the_same_bone_prefix(self) -> None:
        instances = np.zeros((8, 8, 8), dtype=np.uint16)
        instances[0:2, :, :] = 1
        instances[2:4, :, :] = 2
        instances[4:6, :, :] = 3

        tibia = np.isin(instances, (1, 2))
        fibula = instances == 3
        counts, overlaps = compute_array_overlap_counts(
            instances,
            {"tibia": tibia, "fibula": fibula},
        )
        assignments = assign_from_overlap_counts(counts, overlaps)

        self.assertEqual(assignments["1"]["name"], "tibia_1")
        self.assertEqual(assignments["2"]["name"], "tibia_2")
        self.assertEqual(assignments["3"]["name"], "fibula_3")
        self.assertEqual(assignments["2"]["best_overlap"], 1.0)

    def test_low_overlap_fragment_is_unknown(self) -> None:
        assignments = assign_from_overlap_counts(
            {4: 100},
            {"tibia": {4: 49}},
            AssignmentConfig(min_overlap=0.5, min_margin=0.2),
        )

        self.assertEqual(assignments["4"]["name"], "unknown_4")
        self.assertEqual(assignments["4"]["reason"], "overlap_below_threshold")

    def test_ambiguous_overlap_is_unknown(self) -> None:
        assignments = assign_from_overlap_counts(
            {7: 100},
            {"tibia": {7: 70}, "fibula": {7: 60}},
            AssignmentConfig(min_overlap=0.5, min_margin=0.2),
        )

        self.assertEqual(assignments["7"]["bone"], "unknown")
        self.assertEqual(assignments["7"]["best_bone"], "tibia")
        self.assertEqual(assignments["7"]["reason"], "ambiguous_overlap_margin")

    def test_missing_masks_degrades_to_unknown(self) -> None:
        assignments = assign_from_overlap_counts({9: 25}, {})

        self.assertEqual(assignments["9"]["name"], "unknown_9")
        self.assertEqual(assignments["9"]["best_overlap"], 0.0)

    def test_rejects_invalid_instances_and_mask_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_array_overlap_counts(np.full((2, 2, 2), -1), {})
        with self.assertRaisesRegex(ValueError, "discrete integers"):
            compute_array_overlap_counts(np.full((2, 2, 2), 1.5), {})
        with self.assertRaisesRegex(ValueError, "does not match"):
            compute_array_overlap_counts(
                np.ones((2, 2, 2), dtype=np.uint8),
                {"tibia": np.ones((2, 2, 3), dtype=np.uint8)},
            )

    def test_rejects_invalid_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_overlap"):
            AssignmentConfig(min_overlap=1.1).validate()
        with self.assertRaisesRegex(ValueError, "min_margin"):
            AssignmentConfig(min_margin=-0.1).validate()


class FragmentBoneAssignmentNiftiTests(unittest.TestCase):
    @staticmethod
    def _save(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
        nib.save(nib.Nifti1Image(array, affine), str(path))

    def test_chunked_report_unions_sided_masks_and_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = root / "totalseg"
            masks.mkdir()
            affine = np.diag([0.5, 0.5, 1.0, 1.0])
            instances = np.zeros((5, 4, 7), dtype=np.uint16)
            instances[0:2, :, 0:3] = 1
            instances[0:2, :, 4:7] = 2
            instances[3:5, :, 1:6] = 3
            instance_path = root / "instances.nii.gz"
            self._save(instance_path, instances, affine)

            self._save(masks / "tibia_left.nii.gz", (instances == 1).astype(np.uint8), affine)
            self._save(masks / "tibia_right.nii.gz", (instances == 2).astype(np.uint8), affine)
            self._save(masks / "fibula.nii.gz", (instances == 3).astype(np.uint8), affine)

            report = build_assignment_report(
                instance_path,
                masks,
                chunk_depth=2,
            )
            self.assertEqual(report["instances"]["1"]["name"], "tibia_1")
            self.assertEqual(report["instances"]["2"]["name"], "tibia_2")
            self.assertEqual(report["instances"]["3"]["name"], "fibula_3")
            self.assertEqual(report["summary"]["assigned_instances"], 3)
            self.assertEqual(report["missing_bone_masks"], ["femur", "patella", "fabella"])

            output = root / "result" / "instance_classes.json"
            exit_code = main(
                [
                    "--instances",
                    str(instance_path),
                    "--totalseg-dir",
                    str(masks),
                    "--output",
                    str(output),
                    "--chunk-depth",
                    "2",
                ]
            )
            self.assertEqual(exit_code, 0)
            with output.open(encoding="utf-8") as handle:
                written = json.load(handle)
            self.assertEqual(written["instances"]["2"]["bone"], "tibia")

    def test_postprocessing_writes_seg_nifti_and_named_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.array(
                [[-0.7, 0, 0, 12], [0, -0.7, 0, 30], [0, 0, 1.2, -4], [0, 0, 0, 1]],
                dtype=float,
            )
            instances = np.zeros((6, 5, 4), dtype=np.uint16)
            instances[0:3] = 4
            instances[3:5] = 8
            instances[5:6] = 9
            instance_path = root / "internal_instances.nii.gz"
            self._save(instance_path, instances, affine)
            report = {
                "instances": {
                    "4": {"bone": "tibia"},
                    "8": {"bone": "tibia"},
                    "9": {"bone": "fibula"},
                }
            }

            named, aggregate_path = export_named_segmentations(
                instance_path,
                report,
                root / "output",
                "L",
            )

            self.assertEqual(aggregate_path.name, "seg.nii.gz")
            self.assertTrue(aggregate_path.is_file())
            aggregate_image = nib.load(str(aggregate_path))
            self.assertEqual(aggregate_image.shape, instances.shape)
            self.assertTrue(np.allclose(aggregate_image.affine, affine))
            self.assertEqual(
                set(np.unique(np.asanyarray(aggregate_image.dataobj))),
                {1, 2, 21},
            )
            self.assertEqual(
                {path.name for path in named},
                {
                    "tibia_L.nii.gz",
                    "tibia_L_fragment_1.nii.gz",
                    "fibula_L.nii.gz",
                },
            )

    def test_rejects_shape_and_affine_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masks = root / "totalseg"
            masks.mkdir()
            instance_path = root / "instances.nii.gz"
            identity = np.eye(4)
            self._save(instance_path, np.ones((3, 3, 3), dtype=np.uint8), identity)

            self._save(masks / "tibia.nii.gz", np.ones((3, 3, 4), dtype=np.uint8), identity)
            with self.assertRaisesRegex(ValueError, "shape does not match"):
                build_assignment_report(instance_path, masks)

            (masks / "tibia.nii.gz").unlink()
            shifted = identity.copy()
            shifted[0, 3] = 2.0
            self._save(masks / "tibia.nii.gz", np.ones((3, 3, 3), dtype=np.uint8), shifted)
            with self.assertRaisesRegex(ValueError, "affine does not match"):
                build_assignment_report(instance_path, masks)


class FragmentBoneAssignmentBatchTests(unittest.TestCase):
    @staticmethod
    def _save(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array, affine), str(path))

    def test_pairs_case_directories_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances_root = root / "instances"
            totalseg_root = root / "totalseg"
            affine = np.diag([0.7, 0.7, 1.2, 1.0])

            for case_name, instance_id, assigned_bone in (
                ("ct", 1, "tibia"),
                ("followup", 3, "femur"),
            ):
                instances = np.full((3, 4, 5), instance_id, dtype=np.uint16)
                self._save(
                    instances_root / case_name / f"{case_name}_instance.nii.gz",
                    instances,
                    affine,
                )
                for bone in ("tibia", "fibula", "femur", "patella", "fabella"):
                    mask = (instances > 0).astype(np.uint8) if bone == assigned_bone else np.zeros_like(instances)
                    self._save(
                        totalseg_root / case_name / f"{bone}.nii.gz",
                        mask,
                        affine,
                    )

            outputs = process_assignment_directories(
                instances_root,
                totalseg_root,
                chunk_depth=2,
            )

            self.assertEqual(
                outputs,
                [
                    instances_root / "ct" / "instance_classes.json",
                    instances_root / "followup" / "instance_classes.json",
                ],
            )
            with outputs[0].open(encoding="utf-8") as handle:
                ct_report = json.load(handle)
            with outputs[1].open(encoding="utf-8") as handle:
                followup_report = json.load(handle)
            self.assertEqual(ct_report["instances"]["1"]["name"], "tibia_1")
            self.assertEqual(followup_report["instances"]["3"]["name"], "femur_3")

    def test_requires_all_masks_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.eye(4)
            instances = np.ones((2, 2, 2), dtype=np.uint16)
            self._save(root / "instances" / "ct" / "ct_instance.nii.gz", instances, affine)
            self._save(root / "totalseg" / "ct" / "tibia.nii.gz", instances, affine)

            with self.assertRaisesRegex(FileNotFoundError, "fibula, femur, patella, fabella"):
                process_assignment_directories(root / "instances", root / "totalseg")

    def test_rejects_empty_instance_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instances").mkdir()
            (root / "totalseg").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "no \*_instance"):
                process_assignment_directories(root / "instances", root / "totalseg")


if __name__ == "__main__":
    unittest.main()
