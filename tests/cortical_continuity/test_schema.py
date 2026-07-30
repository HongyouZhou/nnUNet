from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnunetv2.training.cortical_continuity.schema import (
    AXIAL_19_DIRECTION_SET,
    DENSE_39_DIRECTION_SET,
    CorticalContinuityHeadSchema,
    build_cortical_continuity_schema,
)


class CorticalContinuitySchemaTests(unittest.TestCase):
    def test_default_schema_has_formal_c_plus_19_affinity_channels(self) -> None:
        schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))
        self.assertEqual(schema.direction_set, AXIAL_19_DIRECTION_SET)
        self.assertEqual(schema.num_affinity_channels, 19)
        self.assertEqual(schema.total_channels, 20)
        self.assertEqual((schema.head("cortex").start, schema.head("cortex").stop), (0, 1))
        self.assertEqual((schema.head("affinity").start, schema.head("affinity").stop), (1, 20))
        self.assertEqual(sum(i.family == "local_26_half" for i in schema.affinity_offsets), 13)
        self.assertEqual(sum(i.family == "lifted_axial" for i in schema.affinity_offsets), 6)
        self.assertFalse(schema.directional_affinity_mirror_tta)

    def test_dense39_is_explicit_gate_controlled_alternative(self) -> None:
        schema = build_cortical_continuity_schema(
            (1.0, 1.0, 1.0),
            direction_set=DENSE_39_DIRECTION_SET,
        )
        self.assertEqual(schema.num_affinity_channels, 39)
        self.assertEqual(schema.total_channels, 40)
        self.assertEqual(sum(i.family == "local_26_half" for i in schema.affinity_offsets), 13)
        self.assertEqual(sum(i.family == "lifted_dense" for i in schema.affinity_offsets), 26)

    def test_coarse_spacing_deduplicates_lifted_offsets(self) -> None:
        schema = build_cortical_continuity_schema((4.0, 4.0, 4.0))
        self.assertTrue(13 <= schema.num_affinity_channels <= 19)
        offsets = [i.voxel_offset_zyx for i in schema.affinity_offsets]
        self.assertEqual(len(offsets), len(set(offsets)))

    def test_schema_json_and_file_round_trip(self) -> None:
        schema = build_cortical_continuity_schema((0.6, 0.7, 1.2))
        self.assertEqual(CorticalContinuityHeadSchema.loads(schema.dumps()), schema)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "head_schema.json"
            schema.save(output)
            self.assertEqual(CorticalContinuityHeadSchema.load(output), schema)
            self.assertEqual(json.loads(output.read_text())["version"], 1)

    def test_split_and_activation_use_declared_channel_axis(self) -> None:
        schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))
        logits = np.zeros((2, schema.total_channels, 3, 4, 5), dtype=np.float32)
        logits[:, 0] = -2.0
        logits[:, 1:] = 2.0
        split = schema.split(logits, channel_axis=1)
        activated = schema.activate(logits, channel_axis=1)
        self.assertEqual(split["cortex"].shape, (2, 1, 3, 4, 5))
        self.assertEqual(split["affinity"].shape, (2, 19, 3, 4, 5))
        self.assertTrue(np.allclose(activated["cortex"], 1.0 / (1.0 + np.exp(2.0))))
        self.assertTrue(np.allclose(activated["affinity"], 1.0 / (1.0 + np.exp(-2.0))))

    def test_split_rejects_wrong_flat_width(self) -> None:
        schema = build_cortical_continuity_schema((1.0, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "Expected 20 flat channels"):
            schema.split(np.zeros((19, 2, 2, 2), dtype=np.float32), channel_axis=0)

    def test_schema_refuses_unsafe_mirror_tta(self) -> None:
        value = build_cortical_continuity_schema((1.0, 1.0, 1.0)).to_dict()
        value["directional_affinity_mirror_tta"] = True
        with self.assertRaisesRegex(ValueError, "not implemented"):
            CorticalContinuityHeadSchema.from_dict(value)

    def test_schema_rejects_invalid_spacing(self) -> None:
        for spacing in (
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (float("nan"), 1.0, 1.0),
            (1.0, 1.0),
        ):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                build_cortical_continuity_schema(spacing)

    def test_schema_rejects_unknown_direction_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "direction_set"):
            build_cortical_continuity_schema((1.0, 1.0, 1.0), direction_set="surprise")


if __name__ == "__main__":
    unittest.main()
