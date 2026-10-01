"""Exercise real trainer construction before allocating GPU smoke jobs."""

import json

import pytest
import torch

from nnunetv2.training.cortical_separator_continuity.configure_plans import (
    FORMAL_PLANS_NAME,
    freeze_separator_continuity_plans,
)
from nnunetv2.training.cortical_separator_continuity.contract import (
    FORMAL_PREPROCESSOR,
    DensityCalibration,
    sha256_file,
)
from nnunetv2.training.cortical_separator_continuity.trainer import (
    nnUNetTrainerCorticalSeparatorMatchedBase,
    nnUNetTrainerCorticalSeparatorContinuity,
    nnUNetTrainerCorticalSeparatorContinuityDensityPrior,
    nnUNetTrainerCorticalSeparatorMatchedBaseSmoke,
    nnUNetTrainerCorticalSeparatorContinuitySmoke,
    nnUNetTrainerCorticalSeparatorContinuityDensityPriorSmoke,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


@pytest.mark.parametrize(
    "trainer_class,epochs,save_every",
    [
        (nnUNetTrainerCorticalSeparatorMatchedBase, 1000, 50),
        (nnUNetTrainerCorticalSeparatorContinuity, 1000, 50),
        (nnUNetTrainerCorticalSeparatorContinuityDensityPrior, 1000, 50),
        (nnUNetTrainerCorticalSeparatorMatchedBaseSmoke, 2, 1),
        (nnUNetTrainerCorticalSeparatorContinuitySmoke, 2, 1),
        (nnUNetTrainerCorticalSeparatorContinuityDensityPriorSmoke, 2, 1),
    ],
)
def test_real_constructor_records_checkpoint_arguments(
    tmp_path, monkeypatch, trainer_class, epochs, save_every
):
    dataset_name = "Dataset778_ChariteCorticalContinuity"
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.setenv(key, str(tmp_path / key))
    monkeypatch.setenv("nnUNet_wandb_enabled", "false")
    monkeypatch.delenv("CORTICAL_CONTINUITY_CALIBRATION_DIR", raising=False)
    raw = tmp_path / "nnUNet_raw" / dataset_name
    preprocessed = tmp_path / "nnUNet_preprocessed" / dataset_name
    calibration_dir = preprocessed / "separator_continuity_calibration"
    raw.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)
    manifest_path = raw / "continuity_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    splits_path = preprocessed / "splits_final.json"
    splits_path.write_text(
        json.dumps([{"train": ["case_train"], "val": ["case_val"]}]),
        encoding="utf-8",
    )
    plans = freeze_separator_continuity_plans(
        {
            "dataset_name": dataset_name,
            "plans_name": FORMAL_PLANS_NAME,
            "configurations": {
                "3d_fullres": {
                    "spacing": [0.5, 0.5, 0.5],
                    "patch_size": [8, 8, 8],
                    "preprocessor_name": FORMAL_PREPROCESSOR,
                }
            },
        }
    )
    plans_path = preprocessed / f"{FORMAL_PLANS_NAME}.json"
    plans_path.write_text(json.dumps(plans), encoding="utf-8")
    calibration = DensityCalibration(
        fold=0,
        train_cases=("case_train",),
        q25_hu=100,
        q50_hu=500,
        q75_hu=1000,
        splits_sha256=sha256_file(splits_path),
        plans_sha256=sha256_file(plans_path),
        source_manifest_sha256=sha256_file(manifest_path),
        voxel_count=100,
    )
    (calibration_dir / "fold_0.json").write_text(
        json.dumps(calibration.as_dict()), encoding="utf-8"
    )
    plans["continue_training"] = False
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "body": 1, "separator": 2, "ignore": 3},
        "file_ending": ".nii.gz",
    }
    trainer = trainer_class(
        plans=plans,
        configuration="3d_fullres",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )
    assert set(trainer.my_init_kwargs) == {
        "plans", "configuration", "fold", "dataset_json", "device"
    }
    assert trainer.my_init_kwargs["fold"] == 0
    assert trainer.my_init_kwargs["dataset_json"] == dataset_json
    assert trainer.num_epochs == epochs
    assert trainer.save_every == save_every
    assert trainer.network is None
    assert (trainer.density_calibration is not None) == bool(trainer.density_weight)
    if epochs == 2:
        def unexpected_full_volume_export(*args, **kwargs):
            pytest.fail("Performance smoke must not run whole-volume inference")

        monkeypatch.setattr(
            nnUNetTrainer, "perform_actual_validation", unexpected_full_volume_export
        )
        trainer.perform_actual_validation(save_probabilities=True)
