from __future__ import annotations

import json

import numpy as np

from nnunetv2.training.cortical_separator_prior.contract import (
    HU_CODE_OFFSET,
    DensityCalibration,
    decode_hu_code,
    density_score_from_hu,
    encode_hu,
)
from nnunetv2.training.cortical_separator_prior.sampling import (
    sample_separator_prior_center,
)
from tools.charite_cortical.density_prior import pilot_gate


def test_hu_encoding_is_bounded_and_reversible() -> None:
    hu = np.asarray([-3000.0, -1000.2, 0.0, 1200.7, 5000.0])
    code = encode_hu(hu)
    assert code.dtype == np.int16
    np.testing.assert_array_equal(
        decode_hu_code(code.astype(np.int32)),
        np.asarray([-2048, -1000, 0, 1201, 4095]),
    )
    assert np.all(code > 0)
    assert HU_CODE_OFFSET == 2049


def test_density_score_is_monotonic_and_fold_calibrated() -> None:
    score = density_score_from_hu(
        np.asarray([100.0, 500.0, 900.0]), 100.0, 500.0, 900.0
    )
    assert score[0] < score[1] < score[2]
    assert score[1] == 0.5


def test_calibration_rejects_non_monotonic_quantiles() -> None:
    try:
        DensityCalibration(
            fold=0,
            train_cases=("case",),
            q25_hu=800,
            q50_hu=500,
            q75_hu=900,
            splits_sha256="0" * 64,
            plans_sha256="1" * 64,
            source_manifest_sha256="2" * 64,
            voxel_count=1,
        )
    except ValueError as error:
        assert "not monotonic" in str(error)
    else:
        raise AssertionError("non-monotonic calibration was accepted")


def test_density_sampler_separates_low_and_high_surface_records() -> None:
    locations = {
        "separator_prior_contact": np.asarray([[0, 1, 1, 1]], dtype=np.int64),
        "separator_prior_normal_surface": np.asarray(
            [[0, 2, 2, 2], [0, 3, 3, 3]], dtype=np.int64
        ),
        "separator_prior_normal_surface_hu": np.asarray(
            [[0, 2, 2, 2, 100], [0, 3, 3, 3, 900]], dtype=np.int64
        ),
    }
    class FixedRng:
        calls = 0

        @staticmethod
        def choice(value, p=None):
            del p
            FixedRng.calls += 1
            if FixedRng.calls == 1:
                return "surface_low"
            if isinstance(value, int):
                return 0
            return list(value)[0]

    category, coordinate, requested = sample_separator_prior_center(
        locations,
        density_median_hu=500,
        use_density_prior=True,
        rng=FixedRng(),
    )
    assert requested == "surface_low"
    assert category == "surface_low"
    np.testing.assert_array_equal(coordinate, np.asarray([0, 2, 2, 2]))


def test_pilot_gate_applies_all_three_thresholds(tmp_path) -> None:
    control_records = []
    prior_records = []
    for fold in (0, 1):
        control_records.extend(
            [
                {
                    "patient_id": f"multi-{fold}",
                    "fold": fold,
                    "all_child_recovery": False,
                    "intact_case": False,
                    "intact_false_split": False,
                    "cortical_union_dice": 0.90,
                },
                {
                    "patient_id": f"intact-{fold}",
                    "fold": fold,
                    "all_child_recovery": True,
                    "intact_case": True,
                    "intact_false_split": False,
                    "cortical_union_dice": 0.90,
                },
            ]
        )
        prior_records.extend(
            [
                {
                    "patient_id": f"multi-{fold}",
                    "fold": fold,
                    "all_child_recovery": True,
                    "intact_case": False,
                    "intact_false_split": False,
                    "cortical_union_dice": 0.895,
                },
                {
                    "patient_id": f"intact-{fold}",
                    "fold": fold,
                    "all_child_recovery": True,
                    "intact_case": True,
                    "intact_false_split": False,
                    "cortical_union_dice": 0.895,
                },
            ]
        )
    control = tmp_path / "control.json"
    prior = tmp_path / "prior.json"
    output = tmp_path / "gate.json"
    control.write_text(json.dumps({"records": control_records}), encoding="utf-8")
    prior.write_text(json.dumps({"records": prior_records}), encoding="utf-8")
    result = pilot_gate(control, prior, output)
    assert result["full_fivefold_allowed"] is True
    assert output.is_file()
