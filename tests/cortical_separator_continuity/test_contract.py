from copy import deepcopy

import pytest

from nnunetv2.training.cortical_separator_continuity.configure_plans import (
    FORMAL_PLANS_NAME,
    freeze_separator_continuity_plans,
)
from nnunetv2.training.cortical_separator_continuity.contract import (
    CONTINUITY_PLANS_KEY,
    FORMAL_PREPROCESSOR,
    PAIR_OFFSETS_ZYX,
    SAMPLING_WEIGHTS,
    TARGET_CHANNEL_LAYOUT,
    validate_continuity_plans_contract,
)


def _plans():
    return {
        "configurations": {
            "3d_fullres": {
                "spacing": [0.5, 0.5, 0.5],
                "preprocessor_name": FORMAL_PREPROCESSOR,
            }
        }
    }


def test_frozen_contract_records_layout_offsets_losses_surface_and_density():
    identifier = f"{FORMAL_PLANS_NAME}_3d_fullres"
    frozen = freeze_separator_continuity_plans(
        _plans(), data_identifier=identifier
    )
    contract = validate_continuity_plans_contract(frozen, identifier)
    assert contract["target_channel_layout"] == list(TARGET_CHANNEL_LAYOUT)
    assert len(contract["pair_offsets"]) == len(PAIR_OFFSETS_ZYX) == 16
    assert contract["loss"]["continuity_weight"] == 0.1
    assert contract["loss"]["density_weight"] == 0.1
    assert contract["normal_surface"]["kind"] == "union_of_per_instance_inner_boundaries"
    assert contract["density"]["includes_gap_voxels"] is False
    assert contract["sampling"] == SAMPLING_WEIGHTS


def test_contract_rejects_spacing_preprocessor_or_mutation():
    bad_spacing = _plans()
    bad_spacing["configurations"]["3d_fullres"]["spacing"] = [1, 1, 1]
    with pytest.raises(ValueError, match="0.5-mm"):
        freeze_separator_continuity_plans(bad_spacing)

    bad_preprocessor = _plans()
    bad_preprocessor["configurations"]["3d_fullres"]["preprocessor_name"] = "DefaultPreprocessor"
    with pytest.raises(ValueError, match=FORMAL_PREPROCESSOR):
        freeze_separator_continuity_plans(bad_preprocessor)

    frozen = freeze_separator_continuity_plans(_plans())
    changed = deepcopy(frozen)
    changed[CONTINUITY_PLANS_KEY]["loss"]["continuity_weight"] = 0.2
    with pytest.raises(RuntimeError, match="incompatible"):
        validate_continuity_plans_contract(
            changed,
            changed["configurations"]["3d_fullres"]["data_identifier"],
        )
