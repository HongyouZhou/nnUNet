"""Runtime contract and data operations for the separator-continuity DAG.

The module deliberately keeps orchestration-independent work in Python so the
Slurm files only map array indices and dependencies. Ground-truth instances are
read only by ``postprocess-evaluate``; inference and minimax watershed consume
network predictions alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed


SCHEMA_VERSION = 1
DATASET_NAME = "Dataset778_ChariteCorticalContinuity"
PLANS_NAME = "nnUNetResEncUNetMPlansSeparatorContinuityPrior"
CONFIGURATION = "3d_fullres"
ARMS = ("matched_base", "continuity", "continuity_density")
TRAINERS = {
    "matched_base": "nnUNetTrainerCorticalSeparatorMatchedBase",
    "continuity": "nnUNetTrainerCorticalSeparatorContinuity",
    "continuity_density": "nnUNetTrainerCorticalSeparatorContinuityDensityPrior",
}
SMOKE_TRAINERS = {
    "matched_base": "nnUNetTrainerCorticalSeparatorMatchedBaseSmoke",
    "continuity": "nnUNetTrainerCorticalSeparatorContinuitySmoke",
    "continuity_density": "nnUNetTrainerCorticalSeparatorContinuityDensityPriorSmoke",
}
PILOT_TASKS = tuple(
    (arm, fold) for arm in ARMS for fold in (0, 1)
)
DEFAULT_POSTPROCESS_CONTRACT = {
    "kind": "separator_minimax_watershed",
    "separator_probability_threshold": 0.5,
    "connectivity": 26,
    "minimum_seed_voxels": 10,
    "child_recovery_iou_threshold": 0.5,
    "false_split_gt_fraction_threshold": 0.1,
}


def _write_json_atomic(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def arm_fold_from_pilot_task(task_id: int) -> tuple[str, int]:
    if task_id not in range(len(PILOT_TASKS)):
        raise ValueError(f"Pilot task must be 0..5, got {task_id}")
    return PILOT_TASKS[task_id]


def trainer_for_arm(arm: str) -> str:
    try:
        return TRAINERS[arm]
    except KeyError as error:
        raise ValueError(f"Unknown separator-continuity arm {arm!r}") from error


def training_output_folder(results_root: Path, arm: str) -> Path:
    return (
        results_root.expanduser().resolve()
        / DATASET_NAME
        / f"{trainer_for_arm(arm)}__{PLANS_NAME}__{CONFIGURATION}"
    )


def checkpoint_path(results_root: Path, arm: str, fold: int) -> Path:
    return training_output_folder(results_root, arm) / f"fold_{fold}" / "checkpoint_final.pth"


def smoke_output_folder(results_root: Path, arm: str) -> Path:
    try:
        trainer = SMOKE_TRAINERS[arm]
    except KeyError as error:
        raise ValueError(f"Unknown separator-continuity arm {arm!r}") from error
    return (
        results_root.expanduser().resolve()
        / DATASET_NAME
        / f"{trainer}__{PLANS_NAME}__{CONFIGURATION}"
    )


def smoke_status(
    results_root: Path,
    *,
    max_epoch_seconds: float = 120.0,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Gate the second fold-0 smoke epoch after a compile warm-up epoch."""

    if max_epoch_seconds <= 0:
        raise ValueError("max_epoch_seconds must be positive")
    records = []
    epoch_pattern = re.compile(r"Epoch time: ([0-9]+(?:\.[0-9]+)?) s")
    for task_id, arm in enumerate(ARMS):
        fold_dir = smoke_output_folder(results_root, arm) / "fold_0"
        checkpoint = fold_dir / "checkpoint_final.pth"
        logs = sorted(
            fold_dir.glob("training_log_*.txt"),
            key=lambda path: path.stat().st_mtime,
        )
        epoch_seconds = None
        epoch_count = 0
        log_path = logs[-1] if logs else None
        if log_path is not None:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = epoch_pattern.findall(text)
            epoch_count = len(matches)
            if matches:
                epoch_seconds = float(matches[-1])
        complete = checkpoint.is_file() and epoch_count >= 2
        within_limit = complete and epoch_seconds <= float(max_epoch_seconds)
        records.append(
            {
                "task_id": task_id,
                "arm": arm,
                "fold": 0,
                "trainer": SMOKE_TRAINERS[arm],
                "checkpoint": str(checkpoint),
                "training_log": None if log_path is None else str(log_path),
                "epoch_count": epoch_count,
                "epoch_seconds": epoch_seconds,
                "complete": complete,
                "within_epoch_limit": within_limit,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_performance_smoke",
        "code_revision": code_revision,
        "max_epoch_seconds": float(max_epoch_seconds),
        "passed": all(record["within_epoch_limit"] for record in records),
        "records": records,
    }


def training_status(
    results_root: Path, pairs: Sequence[tuple[str, int]], *, stage: str
) -> dict[str, Any]:
    records = []
    for task_id, (arm, fold) in enumerate(pairs):
        checkpoint = checkpoint_path(results_root, arm, fold)
        records.append(
            {
                "task_id": task_id,
                "arm": arm,
                "fold": fold,
                "checkpoint": str(checkpoint),
                "complete": checkpoint.is_file(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_training_status",
        "stage": stage,
        "complete": all(record["complete"] for record in records),
        "records": records,
        "missing_task_ids": [
            record["task_id"] for record in records if not record["complete"]
        ],
    }


def _preprocessing_status(preprocessed_root: Path) -> dict[str, Any]:
    dataset_dir = preprocessed_root.expanduser().resolve() / DATASET_NAME
    plans_path = dataset_dir / f"{PLANS_NAME}.json"
    plans = _load_json(plans_path)
    data_identifier = plans["configurations"][CONFIGURATION]["data_identifier"]
    target_dir = dataset_dir / data_identifier
    cases = [
        path for path in target_dir.glob("*.b2nd") if not path.name.endswith("_seg.b2nd")
    ]
    properties = list(target_dir.glob("*.pkl"))
    calibrations = list((dataset_dir / "separator_continuity_calibration").glob("fold_*.json"))
    if len(cases) != 68 or len(properties) != 68 or len(calibrations) != 5:
        raise RuntimeError(
            "Separator-continuity preprocessing is incomplete: "
            f"cases={len(cases)} properties={len(properties)} calibrations={len(calibrations)}"
        )
    return {
        "plans": str(plans_path),
        "data_identifier": data_identifier,
        "case_count": len(cases),
        "property_count": len(properties),
        "calibration_count": len(calibrations),
        "status": "complete",
    }


def initialize_run(
    run_dir: Path,
    dataset: Path,
    preprocessed_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    dataset = dataset.expanduser().resolve(strict=True)
    required = ("continuity_manifest.json", "splits_final.json", "dataset.json")
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset contract is incomplete: {missing}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_dag_run",
        "dataset": str(dataset),
        "preprocessed_root": str(preprocessed_root.expanduser().resolve()),
        "results_root": str(results_root.expanduser().resolve()),
        "plans": PLANS_NAME,
        "configuration": CONFIGURATION,
        "arms": list(ARMS),
        "pilot_tasks": [
            {"task_id": index, "arm": arm, "fold": fold}
            for index, (arm, fold) in enumerate(PILOT_TASKS)
        ],
        "postprocess": DEFAULT_POSTPROCESS_CONTRACT,
        "preprocessing": _preprocessing_status(preprocessed_root),
        "forbidden_inference_paths": ["C+A predictor", "Axial19/Dense39 MWS"],
    }
    contract_path = run_dir / "contract.json"
    if contract_path.exists():
        observed = _load_json(contract_path)
        if observed != contract:
            raise RuntimeError(f"Run directory contains a different contract: {run_dir}")
    else:
        _write_json_atomic(contract_path, contract)
    jobs_path = run_dir / "jobs.json"
    if not jobs_path.exists():
        _write_json_atomic(jobs_path, {"schema_version": SCHEMA_VERSION, "jobs": []})
    return contract


def record_job(run_dir: Path, stage: str, job_id: str, **metadata: Any) -> None:
    path = run_dir.expanduser().resolve() / "jobs.json"
    value = _load_json(path)
    record = {"stage": stage, "job_id": str(job_id), **metadata}
    if record not in value["jobs"]:
        value["jobs"].append(record)
        _write_json_atomic(path, value)


def _manifest_cases(dataset: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(dataset / "continuity_manifest.json")
    cases = {str(case["nnunet_id"]): case for case in manifest["cases"]}
    if len(cases) != 68:
        raise RuntimeError(f"Expected 68 unique manifest cases, got {len(cases)}")
    return manifest, cases


def validation_case_ids(dataset: Path, fold: int) -> list[str]:
    splits = _load_json(dataset / "splits_final.json")
    if fold not in range(len(splits)):
        raise ValueError(f"Fold {fold} is outside splits_final.json")
    return [str(value) for value in splits[fold]["val"]]


def stage_fold_inputs(dataset: Path, fold: int, output: Path) -> list[str]:
    dataset = dataset.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    identifiers = validation_case_ids(dataset, fold)
    expected_names = {f"{identifier}_0000.nii.gz" for identifier in identifiers}
    unexpected = [path for path in output.glob("*.nii.gz") if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"Fold input directory contains unexpected files: {unexpected}")
    for identifier in identifiers:
        source = (dataset / "imagesTr" / f"{identifier}_0000.nii.gz").resolve(strict=True)
        target = output / source.name
        if target.is_symlink():
            if target.resolve(strict=True) != source:
                raise RuntimeError(f"Input symlink targets a different image: {target}")
        elif target.exists():
            raise FileExistsError(f"Refusing non-symlink staged input {target}")
        else:
            target.symlink_to(source)
    return identifiers


def _load_probability_channel(npz_path: Path, reference_shape: tuple[int, ...], channel: int) -> np.ndarray:
    with np.load(npz_path) as archive:
        probabilities = archive[
            "probabilities" if "probabilities" in archive.files else archive.files[0]
        ]
    if probabilities.ndim != 4 or channel >= probabilities.shape[0]:
        raise ValueError(f"Invalid probability tensor {probabilities.shape} in {npz_path}")
    value = np.asarray(probabilities[channel], dtype=np.float32)
    if value.shape != reference_shape:
        value = np.transpose(value, (2, 1, 0))
    if value.shape != reference_shape:
        raise ValueError(
            f"Probability shape {probabilities[channel].shape} cannot match {reference_shape}"
        )
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError(f"Separator probabilities are not finite values in [0,1]: {npz_path}")
    return value


def minimax_watershed_instances(
    semantic: np.ndarray,
    separator_probability: np.ndarray,
    *,
    threshold: float = 0.5,
    minimum_seed_voxels: int = 10,
) -> np.ndarray:
    """Assign cortical voxels by the minimum-maximum separator barrier path."""

    semantic = np.asarray(semantic)
    separator_probability = np.asarray(separator_probability, dtype=np.float32)
    if semantic.shape != separator_probability.shape or semantic.ndim != 3:
        raise ValueError("Semantic prediction and separator probability must share a 3-D grid")
    cortical_union = np.isin(semantic, (1, 2))
    if not np.any(cortical_union):
        return np.zeros(semantic.shape, dtype=np.int32)
    connectivity = np.ones((3, 3, 3), dtype=bool)
    raw_markers, marker_count = connected_components(
        cortical_union & (separator_probability < float(threshold)),
        structure=connectivity,
    )
    if marker_count:
        counts = np.bincount(raw_markers.ravel())
        keep = counts >= int(minimum_seed_voxels)
        keep[0] = False
        raw_markers[~keep[raw_markers]] = 0
    markers, _ = connected_components(raw_markers > 0, structure=connectivity)
    union_components, union_count = connected_components(cortical_union, structure=connectivity)
    next_marker = int(markers.max()) + 1
    for component_id in range(1, union_count + 1):
        component = union_components == component_id
        if np.any(markers[component] > 0):
            continue
        coordinates = np.argwhere(component)
        local_values = separator_probability[component]
        seed = coordinates[int(np.argmin(local_values))]
        markers[tuple(seed)] = next_marker
        next_marker += 1
    result = watershed(
        separator_probability,
        markers=markers,
        mask=cortical_union,
        connectivity=connectivity,
        watershed_line=False,
    )
    return np.asarray(result, dtype=np.int32)


def evaluate_instances(
    predicted_instances: np.ndarray,
    predicted_union: np.ndarray,
    gt_instances: np.ndarray,
    validity: np.ndarray,
    *,
    recovery_iou_threshold: float = 0.5,
    false_split_gt_fraction_threshold: float = 0.1,
) -> dict[str, Any]:
    predicted_instances = np.asarray(predicted_instances)
    predicted_union = np.asarray(predicted_union, dtype=bool)
    gt_instances = np.asarray(gt_instances)
    validity = np.asarray(validity, dtype=np.int64)
    if len({value.shape for value in (predicted_instances, predicted_union, gt_instances, validity)}) != 1:
        raise ValueError("Evaluation arrays do not share a grid")
    semantic_valid = (validity & 1) != 0
    relation_valid = (validity & 2) != 0
    gt = np.where(relation_valid, gt_instances, 0)
    pred = np.where(semantic_valid, predicted_instances, 0)
    gt_ids = np.asarray([value for value in np.unique(gt) if value > 0], dtype=np.int64)
    pred_ids = np.asarray([value for value in np.unique(pred) if value > 0], dtype=np.int64)
    if len(gt_ids) == 0:
        raise ValueError("Evaluation case has no relation-valid GT cortical instance")
    intersection = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_sizes = np.asarray([np.count_nonzero(gt == value) for value in gt_ids], dtype=np.int64)
    pred_sizes = np.asarray([np.count_nonzero(pred == value) for value in pred_ids], dtype=np.int64)
    for gt_index, gt_id in enumerate(gt_ids):
        labels, counts = np.unique(pred[gt == gt_id], return_counts=True)
        lookup = {int(label): int(count) for label, count in zip(labels, counts) if label > 0}
        for pred_index, pred_id in enumerate(pred_ids):
            intersection[gt_index, pred_index] = lookup.get(int(pred_id), 0)
    if len(pred_ids):
        union = gt_sizes[:, None] + pred_sizes[None, :] - intersection
        iou = np.divide(
            intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0
        )
        rows, columns = linear_sum_assignment(-iou)
        recovered = np.zeros(len(gt_ids), dtype=bool)
        recovered[rows] = iou[rows, columns] >= float(recovery_iou_threshold)
    else:
        iou = np.zeros((len(gt_ids), 0), dtype=np.float64)
        recovered = np.zeros(len(gt_ids), dtype=bool)
    gt_fraction = np.divide(
        intersection,
        gt_sizes[:, None],
        out=np.zeros_like(intersection, dtype=np.float64),
        where=gt_sizes[:, None] > 0,
    )
    false_split = (
        np.count_nonzero(gt_fraction >= float(false_split_gt_fraction_threshold), axis=1) > 1
    )
    gt_union = gt > 0
    pred_union = predicted_union & semantic_valid
    overlap = int(np.count_nonzero(gt_union & pred_union))
    denominator = int(np.count_nonzero(gt_union) + np.count_nonzero(pred_union))
    dice = 1.0 if denominator == 0 else 2.0 * overlap / denominator
    return {
        "gt_instance_count": int(len(gt_ids)),
        "predicted_instance_count": int(len(pred_ids)),
        "recovered_child_count": int(np.count_nonzero(recovered)),
        "child_recovery_fraction": float(np.mean(recovered)),
        "all_child_recovery": float(np.all(recovered)),
        "false_split_child_count": int(np.count_nonzero(false_split)),
        "intact_false_split": float(np.mean(false_split)),
        "cortical_union_dice": float(dice),
    }


def postprocess_evaluate_fold(
    dataset: Path,
    prediction_dir: Path,
    output_dir: Path,
    arm: str,
    fold: int,
    output_json: Path,
    contract: Mapping[str, Any] = DEFAULT_POSTPROCESS_CONTRACT,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve(strict=True)
    prediction_dir = prediction_dir.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, cases = _manifest_cases(dataset)
    identifiers = validation_case_ids(dataset, fold)
    records = []
    for identifier in identifiers:
        semantic_path = prediction_dir / f"{identifier}.nii.gz"
        probability_path = prediction_dir / f"{identifier}.npz"
        if not semantic_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(f"Missing OOF prediction for {identifier} in {prediction_dir}")
        semantic_image = nib.load(str(semantic_path))
        semantic = np.rint(np.asanyarray(semantic_image.dataobj)).astype(np.int16)
        probability = _load_probability_channel(probability_path, semantic.shape, 2)
        instances = minimax_watershed_instances(
            semantic,
            probability,
            threshold=float(contract["separator_probability_threshold"]),
            minimum_seed_voxels=int(contract["minimum_seed_voxels"]),
        )
        instance_path = output_dir / f"{identifier}_instances.nii.gz"
        output_header = semantic_image.header.copy()
        output_header.set_data_dtype(np.int32)
        nib.save(
            nib.Nifti1Image(
                instances.astype(np.int32), semantic_image.affine, output_header
            ),
            str(instance_path),
        )
        gt = np.rint(
            np.asanyarray(nib.load(str(dataset / "corticalInstancesTr" / f"{identifier}.nii.gz")).dataobj)
        ).astype(np.int32)
        validity = np.rint(
            np.asanyarray(nib.load(str(dataset / "validMasksTr" / f"{identifier}.nii.gz")).dataobj)
        ).astype(np.int16)
        metrics = evaluate_instances(
            instances,
            np.isin(semantic, (1, 2)),
            gt,
            validity,
            recovery_iou_threshold=float(contract["child_recovery_iou_threshold"]),
            false_split_gt_fraction_threshold=float(
                contract["false_split_gt_fraction_threshold"]
            ),
        )
        case = cases[identifier]
        records.append(
            {
                "patient_id": str(case["patient_id"]),
                "case_id": identifier,
                "fold": int(fold),
                "arm": arm,
                "intact_case": bool(metrics["gt_instance_count"] == 1),
                **metrics,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_fold_metrics",
        "arm": arm,
        "fold": int(fold),
        "postprocess_contract": dict(contract),
        "records": records,
    }
    _write_json_atomic(output_json, result)
    return result


def merge_metrics(inputs: Iterable[Path], arm: str, output: Path) -> dict[str, Any]:
    values = [_load_json(path) for path in inputs]
    if not values:
        raise ValueError("At least one fold metric file is required")
    contracts = [value["postprocess_contract"] for value in values]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise RuntimeError("Fold metrics use different postprocess contracts")
    records = [record for value in values for record in value["records"]]
    patient_ids = [str(record["patient_id"]) for record in records]
    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError("Merged OOF metrics contain duplicate patients")
    if any(record["arm"] != arm for record in records):
        raise RuntimeError(f"Metric input contains records outside arm {arm}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_oof_metrics",
        "arm": arm,
        "folds": sorted({int(record["fold"]) for record in records}),
        "patient_count": len(records),
        "postprocess_contract": contracts[0],
        "summary": {
            key: float(np.mean([float(record[key]) for record in records]))
            for key in (
                "all_child_recovery",
                "child_recovery_fraction",
                "intact_false_split",
                "cortical_union_dice",
            )
        },
        "records": sorted(records, key=lambda record: str(record["patient_id"])),
    }
    _write_json_atomic(output, result)
    return result


def final_summary(inputs: Mapping[str, Path], gates_dir: Path, output: Path) -> dict[str, Any]:
    arms = {arm: _load_json(path) for arm, path in inputs.items()}
    gates = {}
    for name in ("continuity-gate.json", "density-gate.json"):
        path = gates_dir / name
        if path.is_file():
            gates[name.removesuffix(".json")] = _load_json(path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "separator_continuity_final_summary",
        "arms": {
            arm: {
                "folds": value["folds"],
                "patient_count": value["patient_count"],
                "summary": value["summary"],
            }
            for arm, value in arms.items()
        },
        "pilot_gates": gates,
    }
    _write_json_atomic(output, result)
    return result


def _pairs_for_status(stage: str, arm: str | None) -> Sequence[tuple[str, int]]:
    if stage == "pilot":
        if arm is not None:
            raise ValueError("Pilot status does not accept --arm")
        return PILOT_TASKS
    if stage == "remaining":
        if arm is None:
            raise ValueError("Remaining status requires --arm")
        trainer_for_arm(arm)
        return tuple((arm, fold) for fold in (2, 3, 4))
    raise ValueError(stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-run")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--dataset", type=Path, required=True)
    init.add_argument("--preprocessed-root", type=Path, required=True)
    init.add_argument("--results-root", type=Path, required=True)
    record = commands.add_parser("record-job")
    record.add_argument("--run-dir", type=Path, required=True)
    record.add_argument("--stage", required=True)
    record.add_argument("--job-id", required=True)
    status = commands.add_parser("training-status")
    status.add_argument("--results-root", type=Path, required=True)
    status.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    status.add_argument("--arm", choices=ARMS)
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--print-missing", action="store_true")
    smoke = commands.add_parser("smoke-status")
    smoke.add_argument("--results-root", type=Path, required=True)
    smoke.add_argument("--max-epoch-seconds", type=float, default=120.0)
    smoke.add_argument("--code-revision")
    smoke.add_argument("--output", type=Path, required=True)
    mapping = commands.add_parser("map-task")
    mapping.add_argument("--task-id", type=int, required=True)
    mapping.add_argument("--field", choices=("arm", "fold", "trainer"), required=True)
    stage = commands.add_parser("stage-fold-inputs")
    stage.add_argument("--dataset", type=Path, required=True)
    stage.add_argument("--fold", type=int, required=True)
    stage.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("postprocess-evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--arm", choices=ARMS, required=True)
    evaluate.add_argument("--fold", type=int, required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)
    merge = commands.add_parser("merge-metrics")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--arm", choices=ARMS, required=True)
    merge.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("final-summary")
    summary.add_argument("--input", action="append", required=True, metavar="ARM=PATH")
    summary.add_argument("--gates-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-run":
        value = initialize_run(
            args.run_dir, args.dataset, args.preprocessed_root, args.results_root
        )
        print(json.dumps(value, sort_keys=True))
    elif args.command == "record-job":
        record_job(args.run_dir, args.stage, args.job_id)
    elif args.command == "training-status":
        pairs = _pairs_for_status(args.stage, args.arm)
        value = training_status(args.results_root, pairs, stage=args.stage)
        _write_json_atomic(args.output, value)
        if args.print_missing:
            print(",".join(str(item) for item in value["missing_task_ids"]))
        else:
            print(json.dumps(value, sort_keys=True))
    elif args.command == "smoke-status":
        value = smoke_status(
            args.results_root,
            max_epoch_seconds=args.max_epoch_seconds,
            code_revision=args.code_revision,
        )
        _write_json_atomic(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0 if value["passed"] else 1
    elif args.command == "map-task":
        arm, fold = arm_fold_from_pilot_task(args.task_id)
        print({"arm": arm, "fold": fold, "trainer": trainer_for_arm(arm)}[args.field])
    elif args.command == "stage-fold-inputs":
        print(len(stage_fold_inputs(args.dataset, args.fold, args.output)))
    elif args.command == "postprocess-evaluate":
        value = postprocess_evaluate_fold(
            args.dataset, args.prediction_dir, args.output_dir, args.arm, args.fold,
            args.output_json,
        )
        print(json.dumps({"patient_count": len(value["records"])}))
    elif args.command == "merge-metrics":
        value = merge_metrics(args.inputs, args.arm, args.output)
        print(json.dumps(value["summary"], sort_keys=True))
    elif args.command == "final-summary":
        inputs = {}
        for item in args.input:
            arm, separator, path = item.partition("=")
            if separator != "=" or arm not in ARMS:
                raise ValueError(f"Expected --input ARM=PATH, got {item!r}")
            inputs[arm] = Path(path)
        value = final_summary(inputs, args.gates_dir, args.output)
        print(json.dumps(value["arms"], sort_keys=True))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
