import json

import numpy as np

from tools.charite_cortical.continuity_workflow import (
    PILOT_TASKS,
    arm_fold_from_pilot_task,
    evaluate_instances,
    minimax_watershed_instances,
    smoke_output_folder,
    smoke_status,
)
from tools.charite_cortical.density_prior import pilot_gate


def test_pilot_task_mapping_is_three_arms_by_two_folds():
    assert len(PILOT_TASKS) == 6
    assert arm_fold_from_pilot_task(0) == ("matched_base", 0)
    assert arm_fold_from_pilot_task(1) == ("matched_base", 1)
    assert arm_fold_from_pilot_task(4) == ("continuity_density", 0)
    assert arm_fold_from_pilot_task(5) == ("continuity_density", 1)


def test_minimax_watershed_separates_two_low_barrier_material_basins():
    semantic = np.ones((7, 7, 7), dtype=np.int16)
    semantic[:, :, 3] = 2
    separator_probability = np.full(semantic.shape, 0.05, dtype=np.float32)
    separator_probability[:, :, 3] = 0.95
    instances = minimax_watershed_instances(
        semantic,
        separator_probability,
        threshold=0.5,
        minimum_seed_voxels=2,
    )
    assert len(np.unique(instances[instances > 0])) == 2
    assert np.all(instances > 0)
    assert instances[3, 3, 1] != instances[3, 3, 5]


def test_downstream_metrics_recover_children_and_measure_false_splits():
    gt = np.zeros((4, 4, 4), dtype=np.int16)
    gt[:, :, :2] = 1
    gt[:, :, 2:] = 2
    validity = np.full(gt.shape, 3, dtype=np.int16)
    perfect = evaluate_instances(gt, gt > 0, gt, validity)
    assert perfect["all_child_recovery"] == 1.0
    assert perfect["intact_false_split"] == 0.0
    assert perfect["cortical_union_dice"] == 1.0

    split = gt.copy()
    split[:2, :, :2] = 3
    metrics = evaluate_instances(split, split > 0, gt, validity)
    assert metrics["false_split_child_count"] == 1
    assert metrics["intact_false_split"] == 0.5


def test_gate_uses_child_false_split_rate_when_no_patient_is_intact(tmp_path):
    def metric(recovery):
        return {
            "records": [
                {
                    "patient_id": f"patient-{index}",
                    "fold": index,
                    "intact_case": False,
                    "all_child_recovery": recovery,
                    "intact_false_split": 0.0,
                    "cortical_union_dice": 0.9,
                }
                for index in (0, 1)
            ]
        }

    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "gate.json"
    control.write_text(json.dumps(metric(0.8)), encoding="utf-8")
    candidate.write_text(json.dumps(metric(0.9)), encoding="utf-8")
    result = pilot_gate(control, candidate, output)
    assert result["full_fivefold_allowed"] is True
    assert (
        result["metrics"]["false_split_population"]
        == "all_gt_child_fragments_patient_macro"
    )


def test_smoke_gate_requires_all_three_complete_fast_epochs(tmp_path):
    results = tmp_path / "results"
    for arm, seconds in (
        ("matched_base", 42.0),
        ("continuity", 75.0),
        ("continuity_density", 90.0),
    ):
        fold = smoke_output_folder(results, arm) / "fold_0"
        fold.mkdir(parents=True)
        (fold / "checkpoint_final.pth").write_bytes(b"checkpoint")
        (fold / "training_log_2026_8_5_00_00_00.txt").write_text(
            f"Epoch time: 160.0 s\nEpoch time: {seconds} s\n",
            encoding="utf-8",
        )

    passed = smoke_status(results, max_epoch_seconds=120)
    assert passed["passed"] is True
    assert [record["epoch_seconds"] for record in passed["records"]] == [
        42.0,
        75.0,
        90.0,
    ]

    slow_log = (
        smoke_output_folder(results, "continuity_density")
        / "fold_0"
        / "training_log_2026_8_5_00_00_00.txt"
    )
    slow_log.write_text(
        "Epoch time: 160.0 s\nEpoch time: 121.0 s\n", encoding="utf-8"
    )
    failed = smoke_status(results, max_epoch_seconds=120)
    assert failed["passed"] is False
