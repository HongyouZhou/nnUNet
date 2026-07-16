from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from tools.PENGWIN.postprocess.cortical_anchored_split import SplitConfig
from tools.PENGWIN.postprocess.interactive_split_session import (
    ImageGrid,
    InteractiveSplitSession,
    spacing_aware_prompt_mask,
)


class InteractiveSplitSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (28, 28, 44)
        instance_mask = np.zeros(self.shape, dtype=bool)
        instance_mask[3:25, 3:25, 2:18] = True
        instance_mask[3:25, 3:25, 26:42] = True
        instance_mask[12:16, 12:16, 18:26] = True

        self.ct = np.zeros(self.shape, dtype=np.float32)
        self.abbc = np.zeros(self.shape, dtype=np.uint8)
        self.abbc[instance_mask] = 1
        self.abbc[6:22, 6:22, 5:18] = 2
        self.abbc[6:22, 6:22, 26:39] = 2
        self.instances = np.zeros(self.shape, dtype=np.uint16)
        self.instances[instance_mask] = 7
        self.instances[1, 1, 1] = 3
        self.grid = ImageGrid(
            size_xyz=tuple(reversed(self.shape)),
            spacing_xyz=(0.5, 0.75, 1.5),
            origin_xyz=(10.0, 20.0, 30.0),
            direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        cfg = SplitConfig(
            core_anchor_erosion_mm=0.0,
            core_anchor_min_voxels=20,
            w_prob1=6.0,
            w_prob2=4.0,
            w_prob3=4.0,
            min_split_piece_size=100,
        )
        self.session = InteractiveSplitSession(
            self.ct, self.abbc, self.instances, self.grid, cfg=cfg
        )

    def test_select_prompts_split_commit_and_undo(self) -> None:
        selected = self.session.select_instance_at((12, 12, 10))
        self.assertEqual(selected, 7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))

        preview = self.session.compute_split(prompt_radius_mm=1.0)
        self.assertEqual(preview.selected_instance, 7)
        self.assertEqual(preview.new_instance, 8)
        self.assertTrue(np.all(preview.instances_after[12, 12, 9:12] == 7))
        self.assertTrue(np.all(preview.instances_after[12, 12, 31:34] == 8))
        self.assertEqual(int(preview.instances_after[1, 1, 1]), 3)
        self.assertTrue(
            np.array_equal(preview.instances_after != 0, self.instances != 0)
        )

        event = self.session.commit_split(preview)
        self.assertEqual(event["new_instance"], 8)
        self.assertEqual(self.session.revision, 1)
        self.assertIsNone(self.session.source_point_zyx)
        self.assertGreater(int(self.session.last_cut_mask.sum()), 0)
        split_instances = self.session.instances.copy()

        record = self.session.undo_last_split()
        self.assertEqual(record.new_instance, 8)
        self.assertTrue(np.array_equal(self.session.instances, self.instances))
        self.assertEqual(self.session.revision, 2)

        redo_record = self.session.redo_last_split()
        self.assertEqual(redo_record.new_instance, 8)
        self.assertTrue(np.array_equal(self.session.instances, split_instances))
        self.assertEqual(self.session.revision, 3)
        self.assertFalse(self.session.redo_stack)

    def test_multiple_undo_redo_round_trip(self) -> None:
        original = self.session.instances.copy()
        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))
        self.session.commit_split(self.session.compute_split(prompt_radius_mm=1.0))
        after_first = self.session.instances.copy()

        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 7))
        self.session.set_prompt("sink", (12, 12, 15))
        self.session.commit_split(self.session.compute_split(prompt_radius_mm=1.0))
        after_second = self.session.instances.copy()

        self.assertEqual(self.session.undo_last_split().new_instance, 9)
        self.assertTrue(np.array_equal(self.session.instances, after_first))
        self.assertEqual(self.session.undo_last_split().new_instance, 8)
        self.assertTrue(np.array_equal(self.session.instances, original))
        self.assertEqual(self.session.redo_last_split().new_instance, 8)
        self.assertTrue(np.array_equal(self.session.instances, after_first))
        self.assertEqual(self.session.redo_last_split().new_instance, 9)
        self.assertTrue(np.array_equal(self.session.instances, after_second))

    def test_new_split_clears_redo_stack(self) -> None:
        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))
        first = self.session.compute_split(prompt_radius_mm=1.0)
        self.session.commit_split(first)
        self.session.undo_last_split()
        self.assertTrue(self.session.redo_stack)

        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))
        replacement = self.session.compute_split(prompt_radius_mm=1.0)
        self.session.commit_split(replacement)

        self.assertFalse(self.session.redo_stack)
        with self.assertRaisesRegex(ValueError, "no undone split"):
            self.session.redo_last_split()

    def test_source_and_sink_can_be_placed_on_different_slices(self) -> None:
        self.session.select_instance_id(7)
        source = self.session.set_prompt("source", (8, 12, 10))
        sink = self.session.set_prompt("sink", (20, 12, 32))

        preview = self.session.compute_split(prompt_radius_mm=1.0)

        self.assertNotEqual(source[0], sink[0])
        self.assertEqual(int(preview.instances_after[source]), 7)
        self.assertEqual(int(preview.instances_after[sink]), preview.new_instance)

    def test_repeated_split_uses_prompt_only_refinement(self) -> None:
        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))
        first = self.session.compute_split(prompt_radius_mm=1.0)
        self.assertFalse(first.diagnostics["interactive_refinement_mode"])
        self.session.commit_split(first)

        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 7))
        self.session.set_prompt("sink", (12, 12, 15))
        second = self.session.compute_split(prompt_radius_mm=1.0)

        self.assertTrue(second.diagnostics["interactive_refinement_mode"])
        self.assertEqual(
            second.diagnostics["bottleneck_growth_seed_method"],
            "prompt_only_refinement",
        )

    def test_rejects_background_selection_and_wrong_instance_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "instance 0 is not present"):
            self.session.select_instance_at((0, 0, 0))
        self.session.select_instance_id(7)
        with self.assertRaisesRegex(ValueError, "not selected instance"):
            self.session.set_prompt("source", (1, 1, 1))

    def test_spacing_aware_prompt_is_clipped_to_selected_instance(self) -> None:
        allowed = self.instances == 7
        prompt = spacing_aware_prompt_mask(
            self.shape,
            (12, 12, 10),
            self.grid.spacing_zyx,
            radius_mm=1.5,
            allowed_mask=allowed,
        )
        self.assertTrue(prompt[12, 12, 10])
        self.assertFalse(np.any(prompt & ~allowed))
        # Z spacing is 1.5 mm, while X spacing is 0.5 mm.
        self.assertTrue(prompt[11, 12, 10])
        self.assertFalse(prompt[10, 12, 10])
        self.assertTrue(prompt[12, 12, 13])
        self.assertFalse(prompt[12, 12, 14])

    def test_save_preserves_grid_and_interaction_log(self) -> None:
        self.session.select_instance_id(7)
        self.session.set_prompt("source", (12, 12, 10))
        self.session.set_prompt("sink", (12, 12, 32))
        self.session.commit_split(self.session.compute_split(prompt_radius_mm=1.0))

        with tempfile.TemporaryDirectory() as tmp:
            output = self.session.save_instances(Path(tmp) / "instances.nii.gz")
            log = self.session.save_interactions(Path(tmp) / "interactions.json")
            image = sitk.ReadImage(str(output))
            self.assertEqual(image.GetSize(), self.grid.size_xyz)
            self.assertTrue(np.allclose(image.GetSpacing(), self.grid.spacing_xyz))
            self.assertTrue(np.allclose(image.GetOrigin(), self.grid.origin_xyz))
            self.assertTrue(log.is_file())

    def test_from_files_rejects_grid_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ct_image = self.grid.image_from_array(self.ct)
            abbc_image = self.grid.image_from_array(self.abbc)
            abbc_image.SetOrigin((11.0, 20.0, 30.0))
            ct_path = root / "ct.nii.gz"
            abbc_path = root / "abbc.nii.gz"
            instances_path = root / "instances.nii.gz"
            sitk.WriteImage(ct_image, str(ct_path))
            sitk.WriteImage(abbc_image, str(abbc_path))
            sitk.WriteImage(self.grid.image_from_array(self.instances), str(instances_path))

            with self.assertRaisesRegex(ValueError, "origin_xyz"):
                InteractiveSplitSession.from_files(
                    ct_path, abbc_path, instances_path=instances_path
                )

    def test_from_files_keeps_instance_support_independent_from_abbc(self) -> None:
        abbc = self.abbc.copy()
        abbc[0, 0, 0] = 1
        instances = self.instances.copy()
        instances[3, 3, 3] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "ct": root / "ct.nii.gz",
                "abbc": root / "abbc.nii.gz",
                "instances": root / "instances.nii.gz",
            }
            for name, array in (
                ("ct", self.ct),
                ("abbc", abbc),
                ("instances", instances),
            ):
                sitk.WriteImage(self.grid.image_from_array(array), str(paths[name]))

            session = InteractiveSplitSession.from_files(
                paths["ct"], paths["abbc"], instances_path=paths["instances"]
            )

            self.assertTrue(np.array_equal(session.instances, instances))
            self.assertFalse(session.instances[0, 0, 0] > 0)
            self.assertEqual(int(session.abbc[0, 0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
