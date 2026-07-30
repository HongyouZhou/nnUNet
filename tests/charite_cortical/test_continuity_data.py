from __future__ import annotations

import csv
import gzip
import hashlib
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tools.charite_cortical.continuity_data.audit_dataset import audit_dataset
from tools.charite_cortical.continuity_data.build_dataset import (
    load_source_index,
    main as build_main,
)
from tools.charite_cortical.continuity_data.io import (
    read_nifti_array,
    sha256_file,
)
from tools.charite_cortical.continuity_data.schema import (
    SPLIT_RANDOM_STATE,
    BuildConfig,
    validate_manifest,
)
from tools.charite_cortical.continuity_data.splits import (
    make_stratified_fivefold,
)
from tools.charite_cortical.continuity_data.targets import (
    RELATION_VALID,
    SEMANTIC_VALID,
    derive_case_targets,
)


Shape = tuple[int, int, int]
Coordinate = tuple[int, int, int]
SegmentSpec = tuple[str, int, tuple[Coordinate, ...]]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_nifti(
    path: Path,
    *,
    shape: Shape = (6, 5, 1),
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
) -> None:
    header = bytearray(348)
    struct.pack_into("<I", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    struct.pack_into("<8f", header, 76, 1.0, *spacing, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<h", header, 252, 0)
    struct.pack_into("<h", header, 254, 1)
    for offset, row in (
        (280, (spacing[0], 0.0, 0.0, 0.0)),
        (296, (0.0, spacing[1], 0.0, 0.0)),
        (312, (0.0, 0.0, spacing[2], 0.0)),
    ):
        struct.pack_into("<4f", header, offset, *row)
    header[344:348] = b"n+1\0"
    values = np.arange(np.prod(shape), dtype=np.int16).reshape(shape, order="F")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(header)
            stream.write(b"\0\0\0\0")
            stream.write(values.tobytes(order="F"))


def _extent(coordinates: Sequence[Coordinate]) -> str:
    values = np.asarray(coordinates, dtype=np.int64)
    low = values.min(axis=0)
    high = values.max(axis=0)
    return " ".join(str(int(item)) for item in (*low[[0]], *high[[0]], *low[[1]], *high[[1]], *low[[2]], *high[[2]]))


def _write_nrrd(path: Path, shape: Shape, segments: Sequence[SegmentSpec]) -> None:
    layers = max(layer for _, layer, _ in segments) + 1
    values = np.zeros((layers, *shape), dtype=np.uint8, order="F")
    lines = [
        "NRRD0005",
        "type: uint8",
        "dimension: 4",
        f"sizes: {layers} {shape[0]} {shape[1]} {shape[2]}",
        "space directions: none (1,0,0) (0,1,0) (0,0,1)",
        "kinds: list domain domain domain",
        "encoding: raw",
    ]
    for index, (name, layer, coordinates) in enumerate(segments):
        for coordinate in coordinates:
            values[(layer, *coordinate)] = 1
        lines.extend(
            (
                f"Segment{index}_Name:={name}",
                f"Segment{index}_Layer:={layer}",
                f"Segment{index}_LabelValue:=1",
                f"Segment{index}_Extent:={_extent(coordinates)}",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        ("\n".join(lines) + "\n\n").encode("utf-8")
        + values.tobytes(order="F")
    )


def _base_segments() -> tuple[SegmentSpec, SegmentSpec]:
    # The two masks overlap at (2, 2, 0), but have exclusive cortical
    # voxels one millimetre apart on y=1 and y=3.
    return (
        (
            "seg_piece_10",
            1,
            ((2, 1, 0), (2, 2, 0), (2, 3, 0)),
        ),
        (
            "seg_piece_2",
            0,
            ((1, 1, 0), (2, 2, 0), (1, 3, 0)),
        ),
    )


def _cortical_segments(
    fragments: Iterable[SegmentSpec],
    *,
    omit: frozenset[str] = frozenset(),
) -> tuple[SegmentSpec, ...]:
    return tuple(
        (f"{name}_cortical", layer, coordinates)
        for name, layer, coordinates in fragments
        if name not in omit
    )


def _write_source(root: Path) -> None:
    shape = (6, 5, 1)
    case_ids = ("8", "0", "38", "2", "6", "1", "9", "3", "7", "5")
    manifest_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    for position, case_id in enumerate(case_ids):
        case_root = root / "cases" / case_id
        ct = case_root / "ct.nii.gz"
        fragment = case_root / "fragments.seg.nrrd"
        cortical = case_root / "cortical.seg.nrrd"
        fragments = list(_base_segments())
        omitted: frozenset[str] = frozenset()
        if case_id == "0":
            fragments.append(("seg_piece_99", 2, ((4, 2, 0),)))
            omitted = frozenset({"seg_piece_99"})
        if case_id == "38":
            fragments.append(("seg_tibia_right_8", 2, ((4, 2, 0),)))
        cortical_segments = _cortical_segments(fragments, omit=omitted)
        _write_nifti(ct, shape=shape)
        _write_nrrd(fragment, shape, fragments)
        _write_nrrd(cortical, shape, cortical_segments)
        cohort = "CAN" if position < 5 else "CARO"
        manifest_rows.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "fragment_segments": str(len(fragments)),
                "cortical_segments": str(len(cortical_segments)),
                "ct_file": str(ct.relative_to(root)),
                "fragment_file": str(fragment.relative_to(root)),
                "cortical_file": str(cortical.relative_to(root)),
                "ct_sha256": _digest(ct),
                "fragment_sha256": _digest(fragment),
                "cortical_sha256": _digest(cortical),
            }
        )
        for role, specs in (
            ("fragment", fragments),
            ("cortical", cortical_segments),
        ):
            for name, _, coordinates in specs:
                label_rows.append(
                    {
                        "case_id": case_id,
                        "role": role,
                        "name": name,
                        "voxels_in_dataset": str(len(coordinates)),
                    }
                )

    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    (root / "dataset_summary.json").write_text(
        json.dumps({"cases": len(case_ids)}), encoding="utf-8"
    )
    (root / "validation_report.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ContinuityDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.source = cls.root / "source"
        _write_source(cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_overlap_aware_targets_stable_ids_ignore_and_mm_thresholds(self) -> None:
        source = load_source_index(self.source)
        case = next(item for item in source.cases if item.case_id == "0")
        target = derive_case_targets(
            case.fragment_layout,
            case.cortical_layout,
            (1.0, 1.0, 2.0),
            BuildConfig(),
            distance_backend="numpy",
            chunk_spatial_voxels=7,
        )
        self.assertEqual(
            [(item.local_id, item.fragment_name) for item in target.instance_records],
            [
                (1, "seg_piece_2"),
                (2, "seg_piece_10"),
                (3, "seg_piece_99"),
            ],
        )

        starts = tuple(axis[0] for axis in target.bbox_xyz)

        def local(coordinate: Coordinate) -> Coordinate:
            return tuple(
                value - start for value, start in zip(coordinate, starts)
            )  # type: ignore[return-value]

        overlap = local((2, 2, 0))
        self.assertEqual(int(target.cortical_overlap[overlap]), 1)
        self.assertEqual(int(target.cortical_instances[overlap]), 0)
        self.assertEqual(int(target.fragment_overlap[overlap]), 1)
        self.assertEqual(int(target.fragment_instances[overlap]), 0)
        self.assertEqual(int(target.separator[overlap]), 1)
        self.assertEqual(
            int(target.validity[overlap] & RELATION_VALID), 0
        )

        for coordinate in ((1, 1, 0), (2, 1, 0)):
            self.assertEqual(int(target.separator[local(coordinate)]), 2)
        unknown = local((4, 2, 0))
        self.assertEqual(int(target.fragment_support[unknown]), 1)
        self.assertEqual(int(target.separator[unknown]), 3)
        self.assertEqual(int(target.validity[unknown] & SEMANTIC_VALID), 0)
        empty_inside_crop = local((3, 2, 0))
        self.assertEqual(int(target.fragment_support[empty_inside_crop]), 0)
        self.assertEqual(int(target.separator[empty_inside_crop]), 0)
        self.assertEqual(
            target.audit["fragment_overlap_voxels"], 1
        )
        self.assertTrue(
            np.array_equal(
                target.cortex_union > 0,
                (target.separator == 1) | (target.separator == 2),
            )
        )

        anisotropic = derive_case_targets(
            case.fragment_layout,
            case.cortical_layout,
            (2.0, 1.0, 2.0),
            BuildConfig(),
            distance_backend="numpy",
        )
        self.assertEqual(int(np.count_nonzero(anisotropic.separator == 2)), 0)

    def test_split_is_order_stable_cohort_stratified_and_frozen(self) -> None:
        source = load_source_index(self.source)
        records = [case.split_record() for case in source.cases]
        first = make_stratified_fivefold(records)
        second = make_stratified_fivefold(list(reversed(records)))
        self.assertEqual(first.fold_by_case, second.fold_by_case)
        for fold in range(5):
            validation_cases = [
                case
                for case in source.cases
                if first.fold_by_case[case.case_id] == fold
            ]
            self.assertEqual(
                sorted(case.cohort for case in validation_cases),
                ["CAN", "CARO"],
            )
        with self.assertRaisesRegex(ValueError, "frozen"):
            make_stratified_fivefold(
                records, random_state=SPLIT_RANDOM_STATE + 1
            )

    def test_cli_build_is_reproducible_auditable_and_source_immutable(self) -> None:
        before = _tree_hashes(self.source)
        dry_output = self.root / "dry-run-must-not-exist"
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(
                build_main(
                    [
                        "--source",
                        str(self.source),
                        "--output",
                        str(dry_output),
                    ]
                ),
                0,
            )
        self.assertFalse(dry_output.exists())
        self.assertTrue(json.loads(stdout.getvalue())["writes_source_data"] is False)
        self.assertIn("ignored", stderr.getvalue())
        with self.assertRaisesRegex(SystemExit, "explicit --output"):
            build_main(["--source", str(self.source), "--execute"])

        outputs = (self.root / "build-a", self.root / "build-b")
        for output in outputs:
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    build_main(
                        [
                            "--source",
                            str(self.source),
                            "--output",
                            str(output),
                            "--execute",
                            "--distance-backend",
                            "numpy",
                            "--chunk-spatial-voxels",
                            "7",
                        ]
                    ),
                    0,
                )
        self.assertEqual(before, _tree_hashes(self.source))
        self.assertEqual(_tree_hashes(outputs[0]), _tree_hashes(outputs[1]))

        for directory in (
            "labelsTr",
            "corticalInstancesTr",
            "corticalOverlapTr",
            "fragmentInstancesTr",
            "fragmentOverlapTr",
            "validMasksTr",
            "supportTr",
            "rimContactTr",
            "metadataTr",
        ):
            self.assertTrue((outputs[0] / directory).is_dir())
        dataset_json = json.loads(
            (outputs[0] / "dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(dataset_json["labels"]["ignore"], 3)
        manifest = json.loads(
            (outputs[0] / "continuity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        validate_manifest(manifest)
        self.assertEqual(audit_dataset(outputs[0])["status"], "PASS")

        metadata_38 = json.loads(
            (
                outputs[0]
                / "metadataTr"
                / "charite_38.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            metadata_38["exclude_full_volume_metrics"],
            ["seg_tibia_right_8"],
        )
        synthetic = next(
            item
            for item in metadata_38["instances"]
            if item["fragment_name"] == "seg_tibia_right_8"
        )
        self.assertTrue(synthetic["exclude_full_volume_metrics"])
        self.assertEqual(metadata_38["image_materialization"]["method"], "copy")
        self.assertFalse(
            metadata_38["image_materialization"]["shares_source_storage"]
        )

        case_zero = next(
            item for item in manifest["cases"] if item["case_id"] == "0"
        )
        label = read_nifti_array(
            outputs[0] / case_zero["sidecars"]["labels"]["path"]
        )
        valid = read_nifti_array(
            outputs[0] / case_zero["sidecars"]["valid_mask"]["path"]
        )
        support = read_nifti_array(
            outputs[0] / case_zero["sidecars"]["support"]["path"]
        )
        self.assertTrue(np.any(label == 3))
        self.assertTrue(np.all((valid[label == 3] & SEMANTIC_VALID) == 0))
        self.assertTrue(np.all(support[label == 3] == 1))
        self.assertEqual(int(label[0, 0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
