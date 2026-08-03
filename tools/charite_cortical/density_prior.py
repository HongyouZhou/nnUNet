"""Calibrate and diagnostically audit Dataset778 cortical endpoint density."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nnunetv2.training.cortical_separator_continuity.contract import (
    HU_MAX,
    HU_MIN,
    CONTINUITY_CONTRACT_VERSION,
    SEMANTIC_VALID_BIT,
    DensityCalibration,
    density_score_from_hu,
    sha256_file,
)


AUDIT_BOOTSTRAP_ITERATIONS = 10_000
AUDIT_RANDOM_STATE = 20260802
MAX_AUDIT_VOXELS_PER_CLASS = 20_000


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _nifti_array(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as error:
        raise RuntimeError("nibabel is required for density-prior preparation") from error
    image = nib.load(str(path))
    value = np.asanyarray(image.dataobj)
    if value.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI at {path}, got {value.shape}")
    spacing = tuple(float(item) for item in image.header.get_zooms()[:3])
    return value, spacing


def _case_paths(dataset: Path, identifier: str) -> dict[str, Path]:
    ending = ".nii.gz"
    paths = {
        "image": dataset / "imagesTr" / f"{identifier}_0000{ending}",
        "semantic": dataset / "labelsTr" / f"{identifier}{ending}",
        "support": dataset / "supportTr" / f"{identifier}{ending}",
        "validity": dataset / "validMasksTr" / f"{identifier}{ending}",
        "instances": dataset / "corticalInstancesTr" / f"{identifier}{ending}",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset778 case {identifier} is incomplete: {missing}")
    return paths


def _load_case(dataset: Path, identifier: str):
    paths = _case_paths(dataset, identifier)
    image, spacing = _nifti_array(paths["image"])
    semantic, semantic_spacing = _nifti_array(paths["semantic"])
    support, support_spacing = _nifti_array(paths["support"])
    validity, validity_spacing = _nifti_array(paths["validity"])
    instances, instances_spacing = _nifti_array(paths["instances"])
    if not (
        image.shape
        == semantic.shape
        == support.shape
        == validity.shape
        == instances.shape
    ):
        raise ValueError(f"Dataset778 arrays differ for {identifier}")
    if not (
        np.allclose(spacing, semantic_spacing)
        and np.allclose(spacing, support_spacing)
        and np.allclose(spacing, validity_spacing)
        and np.allclose(spacing, instances_spacing)
    ):
        raise ValueError(f"Dataset778 spacing differs for {identifier}")
    return (
        np.asarray(image, dtype=np.float32),
        np.rint(semantic).astype(np.int16),
        np.rint(support).astype(np.int16),
        np.rint(validity).astype(np.int16),
        np.rint(instances).astype(np.int16),
        spacing,
    )


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    count = int(histogram.sum())
    if count <= 0:
        raise ValueError("Cannot compute an HU quantile from an empty histogram")
    target = quantile * (count - 1)
    index = int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))
    return float(HU_MIN + index)


def calibrate(
    dataset: Path,
    plans: Path,
    output_dir: Path,
) -> list[Path]:
    dataset = dataset.expanduser().resolve(strict=True)
    plans = plans.expanduser().resolve(strict=True)
    splits_path = dataset / "splits_final.json"
    manifest_path = dataset / "continuity_manifest.json"
    splits = _load_json(splits_path)
    if not isinstance(splits, list) or len(splits) != 5:
        raise ValueError("Dataset778 requires exactly five frozen folds")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    bins = HU_MAX - HU_MIN + 1
    for fold, split in enumerate(splits):
        train_cases = tuple(sorted(str(value) for value in split["train"]))
        histogram = np.zeros(bins, dtype=np.int64)
        voxel_count = 0
        for identifier in train_cases:
            image, semantic, _, validity, _, _ = _load_case(dataset, identifier)
            valid = (validity & SEMANTIC_VALID_BIT) != 0
            cortical = valid & np.isin(semantic, (1, 2))
            cortical_hu = image[cortical]
            if not np.all(np.isfinite(cortical_hu)):
                raise ValueError(f"Non-finite cortical HU values in {identifier}")
            hu = np.clip(np.rint(cortical_hu), HU_MIN, HU_MAX).astype(np.int32)
            if hu.size:
                histogram += np.bincount(hu - HU_MIN, minlength=bins)
                voxel_count += int(hu.size)
        calibration = DensityCalibration(
            fold=fold,
            train_cases=train_cases,
            q25_hu=_histogram_quantile(histogram, 0.25),
            q50_hu=_histogram_quantile(histogram, 0.50),
            q75_hu=_histogram_quantile(histogram, 0.75),
            splits_sha256=sha256_file(splits_path),
            plans_sha256=sha256_file(plans),
            source_manifest_sha256=sha256_file(manifest_path),
            voxel_count=voxel_count,
        )
        output = output_dir / f"fold_{fold}.json"
        _write_json_atomic(output, calibration.as_dict())
        outputs.append(output)
    return outputs


def _normal_surface(
    instances: np.ndarray,
    separator: np.ndarray,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ModuleNotFoundError as error:
        raise RuntimeError("SciPy is required for density-prior auditing") from error
    inner = np.zeros(instances.shape, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    for instance_id in (int(value) for value in np.unique(instances) if value > 0):
        mask = instances == instance_id
        inner |= mask & ~binary_erosion(
            mask,
            structure=structure,
            border_value=0,
        )
    if np.any(separator):
        distance = distance_transform_edt(~separator, sampling=spacing)
        away = distance > 2.0
    else:
        away = np.ones(separator.shape, dtype=bool)
    return inner & valid & away


def _sample_values(value: np.ndarray, *, seed: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).ravel()
    if len(flat) <= MAX_AUDIT_VOXELS_PER_CLASS:
        return flat
    rng = np.random.RandomState(seed)
    selected = rng.choice(len(flat), MAX_AUDIT_VOXELS_PER_CLASS, replace=False)
    return flat[selected]


def _stable_seed(identifier: str, role: str) -> int:
    digest = hashlib.sha256(f"{identifier}:{role}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate((positive, negative))
    labels = np.concatenate(
        (np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8))
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0
    ) / (len(positive) * len(negative))


def audit(dataset: Path, calibration_dir: Path, output: Path) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve(strict=True)
    calibration_dir = calibration_dir.expanduser().resolve(strict=True)
    splits = _load_json(dataset / "splits_final.json")
    manifest = _load_json(dataset / "continuity_manifest.json")
    splits_digest = sha256_file(dataset / "splits_final.json")
    manifest_digest = sha256_file(dataset / "continuity_manifest.json")
    patient_by_case = {
        str(case["nnunet_id"]): str(case["patient_id"]) for case in manifest["cases"]
    }
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    calibration_plans_digests: set[str] = set()
    for fold, split in enumerate(splits):
        calibration = DensityCalibration.load(calibration_dir / f"fold_{fold}.json")
        if calibration.fold != fold:
            raise RuntimeError(
                f"Calibration file fold_{fold}.json declares fold {calibration.fold}"
            )
        calibration_plans_digests.add(calibration.plans_sha256)
        if calibration.splits_sha256 != splits_digest:
            raise RuntimeError(f"Calibration fold {fold} splits hash mismatch")
        if calibration.source_manifest_sha256 != manifest_digest:
            raise RuntimeError(f"Calibration fold {fold} manifest hash mismatch")
        expected_train = tuple(sorted(str(value) for value in split["train"]))
        if tuple(sorted(calibration.train_cases)) != expected_train:
            raise RuntimeError(f"Calibration fold {fold} has leaked or missing train cases")
        for identifier_value in split["val"]:
            identifier = str(identifier_value)
            image, semantic, _, validity, instances, spacing = _load_case(dataset, identifier)
            valid = (validity & SEMANTIC_VALID_BIT) != 0
            separator = valid & (semantic == 2)
            surface = _normal_surface(instances, separator, valid, spacing)
            if not np.any(separator) or not np.any(surface):
                excluded.append(
                    {
                        "fold": str(fold),
                        "case_id": identifier,
                        "reason": "missing_separator" if not np.any(separator) else "missing_normal_surface",
                    }
                )
                continue
            positive = density_score_from_hu(
                np.clip(image[separator], HU_MIN, HU_MAX),
                calibration.q25_hu,
                calibration.q50_hu,
                calibration.q75_hu,
            )
            negative = density_score_from_hu(
                np.clip(image[surface], HU_MIN, HU_MAX),
                calibration.q25_hu,
                calibration.q50_hu,
                calibration.q75_hu,
            )
            positive = _sample_values(positive, seed=_stable_seed(identifier, "separator"))
            negative = _sample_values(negative, seed=_stable_seed(identifier, "surface"))
            records.append(
                {
                    "fold": fold,
                    "case_id": identifier,
                    "patient_id": patient_by_case[identifier],
                    "separator_voxels_sampled": len(positive),
                    "normal_surface_voxels_sampled": len(negative),
                    "roc_auc": _roc_auc(positive, negative),
                    "median_density_difference": float(np.median(positive) - np.median(negative)),
                }
            )
    if len(calibration_plans_digests) != 1:
        raise RuntimeError("Fold calibrations were generated from different plans files")
    if not records:
        raise RuntimeError("No eligible validation patient remained for density-prior audit")
    patient_ids = [record["patient_id"] for record in records]
    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError("Density-prior audit expects one Dataset778 case per patient")
    auc = np.asarray([record["roc_auc"] for record in records], dtype=np.float64)
    difference = np.asarray(
        [record["median_density_difference"] for record in records], dtype=np.float64
    )
    rng = np.random.RandomState(AUDIT_RANDOM_STATE)
    bootstrap = np.empty(AUDIT_BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for index in range(AUDIT_BOOTSTRAP_ITERATIONS):
        selected = rng.randint(0, len(auc), size=len(auc))
        bootstrap[index] = float(auc[selected].mean())
    ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    mean_auc = float(auc.mean())
    mean_difference = float(difference.mean())
    allowed = bool(ci_low > 0.5 and mean_difference > 0.0)
    reasons = []
    if ci_low <= 0.5:
        reasons.append(f"patient-macro ROC-AUC 95% CI lower bound {ci_low:.6f} <= 0.5")
    if mean_difference <= 0:
        reasons.append(
            f"patient-macro median density difference {mean_difference:.6f} <= 0"
        )
    result = {
        "schema_version": CONTINUITY_CONTRACT_VERSION,
        "kind": "density_prior_diagnostic_audit",
        "training_allowed": allowed,
        "density_prior_training_allowed": allowed,
        "reasons": reasons,
        "patient_count": len(records),
        "excluded_cases": excluded,
        "patient_macro_roc_auc": {
            "mean": mean_auc,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "bootstrap_iterations": AUDIT_BOOTSTRAP_ITERATIONS,
            "random_state": AUDIT_RANDOM_STATE,
        },
        "patient_macro_median_density_difference": mean_difference,
        "records": records,
    }
    _write_json_atomic(output.expanduser().resolve(), result)
    return result


def pilot_gate(control_path: Path, prior_path: Path, output: Path) -> dict[str, Any]:
    control = _load_json(control_path.expanduser().resolve(strict=True))
    prior = _load_json(prior_path.expanduser().resolve(strict=True))
    if len(control["records"]) != len(
        {str(value["patient_id"]) for value in control["records"]}
    ):
        raise ValueError("Control metric file contains duplicate patients")
    if len(prior["records"]) != len(
        {str(value["patient_id"]) for value in prior["records"]}
    ):
        raise ValueError("Prior metric file contains duplicate patients")
    control_records = {str(value["patient_id"]): value for value in control["records"]}
    prior_records = {str(value["patient_id"]): value for value in prior["records"]}
    if set(control_records) != set(prior_records):
        raise ValueError("Control and prior metric files contain different patients")
    if not control_records:
        raise ValueError("Pilot gate requires at least one paired patient")
    for patient_id in control_records:
        if int(control_records[patient_id]["fold"]) != int(prior_records[patient_id]["fold"]):
            raise ValueError(f"Control/prior fold mismatch for patient {patient_id}")
        if bool(control_records[patient_id]["intact_case"]) != bool(
            prior_records[patient_id]["intact_case"]
        ):
            raise ValueError(f"Control/prior intact_case mismatch for patient {patient_id}")
    folds = {
        int(record["fold"])
        for record in list(control_records.values()) + list(prior_records.values())
    }
    if folds != {0, 1}:
        raise ValueError(f"Pilot gate requires exactly folds 0 and 1, got {sorted(folds)}")

    def mean(records: Mapping[str, Mapping[str, Any]], key: str) -> float:
        return float(np.mean([float(record[key]) for record in records.values()]))

    multi_patient_ids = [
        patient_id
        for patient_id, record in control_records.items()
        if not bool(record["intact_case"])
    ]
    if not multi_patient_ids:
        raise ValueError("Pilot gate requires multi-fragment cases")
    control_recovery = float(
        np.mean(
            [float(control_records[patient_id]["all_child_recovery"]) for patient_id in multi_patient_ids]
        )
    )
    prior_recovery = float(
        np.mean(
            [float(prior_records[patient_id]["all_child_recovery"]) for patient_id in multi_patient_ids]
        )
    )
    recovery_gain = prior_recovery - control_recovery
    control_dice = mean(control_records, "cortical_union_dice")
    prior_dice = mean(prior_records, "cortical_union_dice")
    intact = [record for record in prior_records.values() if bool(record["intact_case"])]
    if not intact:
        raise ValueError("Pilot gate requires intact/single-fragment cases")
    false_split_rate = float(
        np.mean([float(record["intact_false_split"]) for record in intact])
    )
    allowed = bool(
        recovery_gain >= 0.03
        and false_split_rate <= 0.05
        and prior_dice - control_dice >= -0.01
    )
    reasons = []
    if recovery_gain < 0.03:
        reasons.append(f"all-child recovery gain {recovery_gain:.6f} < 0.03")
    if false_split_rate > 0.05:
        reasons.append(f"intact false-split rate {false_split_rate:.6f} > 0.05")
    if prior_dice - control_dice < -0.01:
        reasons.append(
            f"cortical-union Dice delta {prior_dice - control_dice:.6f} < -0.01"
        )
    result = {
        "schema_version": CONTINUITY_CONTRACT_VERSION,
        "kind": "density_prior_pilot_gate",
        "full_fivefold_allowed": allowed,
        "reasons": reasons,
        "patient_count": len(control_records),
        "folds": [0, 1],
        "metrics": {
            "control_all_child_recovery": control_recovery,
            "prior_all_child_recovery": prior_recovery,
            "all_child_recovery_gain": recovery_gain,
            "prior_intact_false_split_rate": false_split_rate,
            "control_cortical_union_dice": control_dice,
            "prior_cortical_union_dice": prior_dice,
            "cortical_union_dice_delta": prior_dice - control_dice,
        },
    }
    _write_json_atomic(output.expanduser().resolve(), result)
    return result


def increment_gate(
    reference_path: Path,
    candidate_path: Path,
    output: Path,
    *,
    comparison_name: str,
) -> dict[str, Any]:
    """Apply the same downstream gate to either planned pilot increment."""

    if comparison_name not in {"continuity_vs_matched_base", "density_vs_continuity"}:
        raise ValueError(f"Unsupported pilot comparison {comparison_name!r}")
    result = pilot_gate(reference_path, candidate_path, output)
    result["kind"] = "separator_continuity_increment_gate"
    result["comparison"] = comparison_name
    _write_json_atomic(output.expanduser().resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--dataset", type=Path, required=True)
    calibrate_parser.add_argument("--plans", type=Path, required=True)
    calibrate_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--dataset", type=Path, required=True)
    audit_parser.add_argument("--calibration-dir", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    pilot_parser = subparsers.add_parser("pilot-gate")
    pilot_parser.add_argument("--control", type=Path, required=True)
    pilot_parser.add_argument("--prior", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="summarize paired OOF semantic/downstream records and apply the pilot gate",
    )
    evaluate_parser.add_argument("--control", type=Path, required=True)
    evaluate_parser.add_argument("--prior", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    increment_parser = subparsers.add_parser(
        "increment-gate",
        help="apply the frozen downstream gate to one named pilot increment",
    )
    increment_parser.add_argument("--reference", type=Path, required=True)
    increment_parser.add_argument("--candidate", type=Path, required=True)
    increment_parser.add_argument(
        "--comparison-name",
        choices=("continuity_vs_matched_base", "density_vs_continuity"),
        required=True,
    )
    increment_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "calibrate":
        outputs = calibrate(args.dataset, args.plans, args.output_dir)
        print(f"Wrote {len(outputs)} fold calibrations to {args.output_dir}")
    elif args.command == "audit":
        result = audit(args.dataset, args.calibration_dir, args.output)
        print(
            json.dumps(
                {
                    "diagnostic_only": True,
                    "training_allowed": result["training_allowed"],
                }
            )
        )
    elif args.command in {"pilot-gate", "evaluate"}:
        result = pilot_gate(args.control, args.prior, args.output)
        print(json.dumps({"full_fivefold_allowed": result["full_fivefold_allowed"]}))
        if not result["full_fivefold_allowed"]:
            return 2
    elif args.command == "increment-gate":
        result = increment_gate(
            args.reference,
            args.candidate,
            args.output,
            comparison_name=args.comparison_name,
        )
        print(json.dumps({"full_fivefold_allowed": result["full_fivefold_allowed"]}))
        if not result["full_fivefold_allowed"]:
            return 2
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
