from pathlib import Path

import pytest

from segmentation_io import resolve_inputs, resolve_output_paths
from segmentation_io import (
    default_output_directory,
    publish_contract_output,
    publish_named_segmentations,
    read_laterality,
    resolve_bone_masks_directory,
)


def test_contract_directory_maps_ct_to_seg(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ct.touch()

    inputs = resolve_inputs(tmp_path)

    assert inputs == [ct.resolve()]
    assert resolve_output_paths(inputs, tmp_path / "out") == [
        (tmp_path / "out" / "seg.nii.gz").resolve()
    ]


def test_batch_inputs_keep_distinct_names(tmp_path: Path) -> None:
    (tmp_path / "case_b.nii.gz").touch()
    (tmp_path / "case_a.nii.gz").touch()

    inputs = resolve_inputs(tmp_path)

    assert [path.name for path in inputs] == ["case_a.nii.gz", "case_b.nii.gz"]
    assert [path.name for path in resolve_output_paths(inputs, tmp_path / "out")] == [
        "case_a_seg.nii.gz",
        "case_b_seg.nii.gz",
    ]


def test_single_uuid_named_upload_still_maps_to_seg(tmp_path: Path) -> None:
    ct = tmp_path / "6a7c532c43e1cea3e697b491_0000.nii.gz"
    ct.touch()

    inputs = resolve_inputs(ct)

    assert resolve_output_paths(inputs, tmp_path) == [
        (tmp_path / "seg.nii.gz").resolve()
    ]
    assert default_output_directory(ct) == tmp_path.resolve()


def test_input_directory_is_the_default_publish_directory(tmp_path: Path) -> None:
    assert default_output_directory(tmp_path) == tmp_path.resolve()


def test_contract_result_is_mirrored_to_fixed_filename(tmp_path: Path) -> None:
    generated = tmp_path / "job-output" / "prediction.nii.gz"
    generated.parent.mkdir()
    generated.write_bytes(b"segmentation")
    upload = tmp_path / "FILE_UPLOAD" / "operation"

    published = publish_contract_output(generated, upload)

    assert published == (upload / "seg.nii.gz").resolve()
    assert published.read_bytes() == b"segmentation"


def test_contract_defines_no_abbc_sidecar_output(tmp_path: Path) -> None:
    ct = tmp_path / "ct.nii.gz"
    ct.touch()

    outputs = resolve_output_paths(resolve_inputs(tmp_path), tmp_path / "out")

    assert [path.name for path in outputs] == ["seg.nii.gz"]
    assert all("abbc" not in path.name.lower() for path in outputs)


def test_named_segmentations_are_published_with_the_aggregate(tmp_path: Path) -> None:
    source = tmp_path / "generated" / "segmentations"
    source.mkdir(parents=True)
    (source / "tibia_L.nii.gz").write_bytes(b"main")
    (source / "tibia_L_fragment_1.nii.gz").write_bytes(b"fragment")

    published = publish_named_segmentations(source, tmp_path / "case")

    assert sorted(path.name for path in published.glob("*.nii.gz")) == [
        "tibia_L.nii.gz",
        "tibia_L_fragment_1.nii.gz",
    ]


def test_postprocessing_inputs_are_resolved_from_case(tmp_path: Path) -> None:
    (tmp_path / "ct.nii.gz").touch()
    (tmp_path / "config.json").write_text('{"side": "right"}')
    masks = tmp_path / "postprocessing"
    masks.mkdir()

    assert read_laterality(tmp_path) == "R"
    assert resolve_bone_masks_directory(tmp_path) == masks.resolve()


def test_non_nifti_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ct.mha"
    path.touch()

    with pytest.raises(ValueError, match="nii.gz"):
        resolve_inputs(path)
