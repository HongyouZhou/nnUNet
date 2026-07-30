"""Unified, write-safe CLI for the cortical-continuity MVP.

The command keeps the immutable final68 source separate from generated
Dataset778 artifacts. Commands are read-only unless an explicit output is
provided (and, for expensive builders/oracles, ``--execute`` is also set).
Dense inference and post-processing artifacts use a versioned NPZ working-grid
contract; orientation-sensitive NIfTI conversion stays outside this CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.charite_cortical.continuity_data.build_dataset import (
    DEFAULT_SOURCE,
    execute_build,
    load_source_index,
    plan_build,
)
from tools.charite_cortical.continuity_data.io import read_nifti_array
from tools.charite_cortical.continuity_data.schema import BuildConfig
from tools.charite_cortical.continuity_postprocess import (
    AlwaysAbstainScorer,
    ConservativeSplitScorer,
    ContinuityConfig,
    ContinuityInputs,
    SplitFeatures,
    aggregate_patient_macro,
    controlled_event_manifest,
    evaluate_instance_segmentation,
    evaluate_partition_grouping,
    evaluate_split_calibration,
    fit_logistic_split_scorer,
    load_inputs_npz,
    load_scorer,
    refine_instances,
    refine_with_separator,
    run_o1_event,
    run_o2_event,
)
from nnunetv2.training.cortical_continuity.schema import (
    AXIAL_19_DIRECTION_SET,
    DENSE_39_DIRECTION_SET,
    build_cortical_continuity_schema,
)


SCHEMA_VERSION = 1
FROZEN_PROVISIONAL_MODEL = (
    "nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150"
    "__nnUNetResEncUNetMPlans__3d_fullres"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_json_ready(value), indent=2, sort_keys=True))


def _write_json_new(path: Path, value: Any) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _require_new_npz(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("array output must use an explicit .npz suffix")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _load_config(path: Path | None) -> ContinuityConfig:
    return ContinuityConfig() if path is None else ContinuityConfig.load_json(path)


def _validate_run_metadata(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("run metadata has an unsupported schema_version")
    provisional = value.get("provisional_model")
    if (
        not isinstance(provisional, Mapping)
        or provisional.get("name") != FROZEN_PROVISIONAL_MODEL
        or str(provisional.get("fold")) != "all"
        or provisional.get("checkpoint") != "checkpoint_final"
    ):
        raise ValueError("run metadata does not use the frozen provisional baseline")
    artifacts = value.get("artifacts")
    required = {
        "source_manifest",
        "splits",
        "continuity_config",
        "cortical_plans",
        "cortical_checkpoint",
        "provisional_checkpoint",
        "abbc_to_instance_config",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("run metadata artifact set is incomplete")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"run metadata artifact {name} must be an object")
        path = Path(str(artifact.get("path", ""))).expanduser().resolve(strict=True)
        if _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"run metadata artifact {name} SHA-256 mismatch")


def _load_npz_array(
    path: Path,
    *,
    explicit_key: str | None,
    candidate_keys: Sequence[str],
) -> np.ndarray:
    source = path.expanduser().resolve(strict=True)
    if source.suffix == ".npy":
        if explicit_key is not None:
            raise ValueError("--*-key is invalid for a .npy input")
        return np.load(source, allow_pickle=False)
    with np.load(source, allow_pickle=False) as payload:
        key = explicit_key
        if key is None:
            key = next((item for item in candidate_keys if item in payload.files), None)
        if key is None or key not in payload.files:
            raise ValueError(
                f"{source} has no requested array; available keys={sorted(payload.files)}"
            )
        return np.asarray(payload[key])


def _command_build(args: argparse.Namespace) -> int:
    if args.execute and args.output is None:
        raise ValueError("build --execute requires an explicit --output")
    if args.chunk_spatial_voxels <= 0:
        raise ValueError("--chunk-spatial-voxels must be positive")
    source = load_source_index(args.source)
    target_config = BuildConfig(
        separator_mm=args.separator_mm,
        rim_positive_mm=args.rim_positive_mm,
        rim_negative_mm=args.rim_negative_mm,
    )
    plan, split, selected = plan_build(
        source,
        target_config,
        requested_cases=args.cases,
    )
    if not args.execute:
        _print_json(plan)
        return 0
    report = execute_build(
        source,
        selected,
        split,
        target_config,
        args.output,
        image_mode=args.image_mode,
        distance_backend=args.distance_backend,
        chunk_spatial_voxels=args.chunk_spatial_voxels,
    )
    _print_json(report)
    return 0


def _command_splits(args: argparse.Namespace) -> int:
    source = load_source_index(args.source)
    _, split, _ = plan_build(source, BuildConfig())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "random_state": split.audit["random_state"],
        "strategy": split.audit["strategy"],
        "stratify_by": split.audit["stratify_by"],
        "audit": split.audit,
        "splits_final": split.splits_final,
    }
    if args.output is None:
        _print_json(payload)
    else:
        _write_json_new(args.output, payload)
        _print_json({"status": "PASS", "output": str(args.output.resolve())})
    return 0


def _refine_diagnostics(
    *,
    method: str,
    input_path: Path,
    config: ContinuityConfig,
    scorer: Any,
    result: Any,
    run_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for decision in result.decisions:
        status_counts[decision.status] = status_counts.get(decision.status, 0) + 1
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "input": {
            "path": str(input_path.expanduser().resolve()),
            "sha256": _sha256(input_path.expanduser().resolve(strict=True)),
        },
        "config": config.to_dict(),
        "scorer": None if scorer is None else scorer.to_dict(),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "decisions": [decision.to_dict() for decision in result.decisions],
        "run_metadata": run_metadata,
    }


def _command_refine(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    input_path = args.input.expanduser().resolve(strict=True)
    run_metadata: Mapping[str, Any] | None = None
    if args.formal and args.run_metadata is None:
        raise ValueError("formal refinement requires --run-metadata")
    if args.run_metadata is not None:
        parsed = json.loads(args.run_metadata.read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise ValueError("--run-metadata must contain a JSON object")
        run_metadata = parsed
        _validate_run_metadata(run_metadata)

    scorer: Any = None
    if args.method == "ca":
        inputs = load_inputs_npz(input_path)
        if args.scorer is not None:
            scorer = load_scorer(args.scorer)
        elif args.allow_fixed_thresholds:
            scorer = ConservativeSplitScorer.from_config(config)
        else:
            scorer = AlwaysAbstainScorer("calibrated_scorer_required")
        result = refine_instances(inputs, config=config, scorer=scorer)
    else:
        if args.scorer is not None or args.allow_fixed_thresholds:
            raise ValueError("separator refinement does not use a split scorer")
        with np.load(input_path, allow_pickle=False) as payload:
            required = {
                "provisional_instances",
                "three_class_softmax",
                "spacing_zyx",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(
                    f"separator input bundle is missing arrays: {sorted(missing)}"
                )
            result = refine_with_separator(
                payload["provisional_instances"],
                payload["three_class_softmax"],
                payload["spacing_zyx"],
                config=config,
            )

    output = _require_new_npz(args.output)
    diagnostics = args.diagnostics
    if diagnostics is None:
        diagnostics = output.with_name(output.stem + ".diagnostics.json")
    diagnostics = diagnostics.expanduser().resolve()
    if diagnostics.exists():
        raise FileExistsError(
            f"refusing to overwrite existing diagnostics: {diagnostics}"
        )
    np.savez_compressed(
        output,
        full_instances=result.full_instances,
        cortical_instances=result.cortical_instances,
        raw_cortical_clusters=result.raw_cortical_clusters,
    )
    _write_json_new(
        diagnostics,
        _refine_diagnostics(
            method=args.method,
            input_path=input_path,
            config=config,
            scorer=scorer,
            result=result,
            run_metadata=run_metadata,
        ),
    )
    _print_json(
        {
            "status": "PASS",
            "output": str(output),
            "diagnostics": str(diagnostics),
        }
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    prediction = _load_npz_array(
        args.prediction,
        explicit_key=args.prediction_key,
        candidate_keys=("full_instances", "prediction", "arr_0"),
    )
    ground_truth = _load_npz_array(
        args.ground_truth,
        explicit_key=args.ground_truth_key,
        candidate_keys=("ground_truth", "fragment_instances", "arr_0"),
    )
    valid_mask = None
    if args.valid_mask is not None:
        valid_mask = _load_npz_array(
            args.valid_mask,
            explicit_key=args.valid_mask_key,
            candidate_keys=("valid_mask", "fragment_valid", "arr_0"),
        )
    expected = None
    if args.expected_ground_truth_ids:
        expected = {
            int(item)
            for item in args.expected_ground_truth_ids.split(",")
            if item.strip()
        }
    evaluations = {
        str(threshold): evaluate_instance_segmentation(
            prediction,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
            valid_mask=valid_mask,
            iou_threshold=threshold,
            expected_ground_truth_ids=expected,
        )
        for threshold in args.iou_threshold
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prediction_kind": "instance",
        "ground_truth_kind": "instance",
        "prediction_sha256": _sha256(args.prediction.resolve(strict=True)),
        "ground_truth_sha256": _sha256(args.ground_truth.resolve(strict=True)),
        "evaluations": evaluations,
        "partition_grouping": evaluate_partition_grouping(
            prediction,
            ground_truth,
            prediction_kind="instance",
            ground_truth_kind="instance",
            valid_mask=valid_mask,
        ),
    }
    if args.output is None:
        _print_json(payload)
    else:
        _write_json_new(args.output, payload)
        _print_json({"status": "PASS", "output": str(args.output.resolve())})
    return 0


def _command_calibrate(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(records, list):
        raise ValueError(
            "calibration input must be an object with a records array"
        )
    features: list[SplitFeatures] = []
    proposal_correct: list[bool] = []
    intact_negative: list[bool] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"calibration records[{index}] must be an object")
        feature_value = record.get("features")
        if not isinstance(feature_value, Mapping):
            raise ValueError(
                f"calibration records[{index}].features must be an object"
            )
        features.append(SplitFeatures.from_mapping(feature_value))
        for name, destination in (
            ("proposal_correct", proposal_correct),
            ("intact_negative", intact_negative),
        ):
            raw = record.get(name)
            if not isinstance(raw, bool):
                raise ValueError(
                    f"calibration records[{index}].{name} must be boolean"
                )
            destination.append(raw)
    fit = fit_logistic_split_scorer(
        features,
        proposal_correct,
        intact_negative,
        l2_penalty=args.l2_penalty,
        max_intact_false_split_rate=args.max_intact_false_split_rate,
        min_samples=args.min_samples,
        min_intact_samples=args.min_intact_samples,
        max_iterations=args.max_iterations,
    )
    probabilities = [
        fit.scorer.evaluate(feature).probability for feature in features
    ]
    operating_threshold = getattr(fit.scorer, "threshold", None)
    calibration = evaluate_split_calibration(
        probabilities,
        proposal_correct,
        bins=args.calibration_bins,
        operating_threshold=operating_threshold,
        intact_negative=(
            intact_negative if operating_threshold is not None else None
        ),
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing scorer: {output}")
    report = args.report
    if report is None:
        report = output.with_name(output.stem + ".fit.json")
    report = report.expanduser().resolve()
    if report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report}")
    _write_json_new(output, fit.scorer.to_dict())
    _write_json_new(
        report,
        {
            "schema_version": SCHEMA_VERSION,
            "input": {"path": str(source), "sha256": _sha256(source)},
            "status": fit.status,
            "sample_count": fit.sample_count,
            "intact_count": fit.intact_count,
            "accepted_true_positives": fit.accepted_true_positives,
            "intact_false_split_rate": fit.intact_false_split_rate,
            "max_intact_false_split_rate": args.max_intact_false_split_rate,
            "calibration": calibration,
            "scorer_sha256": _sha256(output),
        },
    )
    _print_json(
        {
            "status": fit.status,
            "scorer": str(output),
            "report": str(report),
            "intact_false_split_rate": fit.intact_false_split_rate,
        }
    )
    return 0


def _command_aggregate(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("aggregate input must be an object with a records array")
    result = aggregate_patient_macro(
        records,
        args.metric,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(source), "sha256": _sha256(source)},
        "aggregation": result,
    }
    if args.output is None:
        _print_json(payload)
    else:
        _write_json_new(args.output, payload)
        _print_json({"status": "PASS", "output": str(args.output.resolve())})
    return 0


def _command_provenance(args: argparse.Namespace) -> int:
    artifact_arguments = {
        "source_manifest": args.source_manifest,
        "splits": args.splits,
        "continuity_config": args.continuity_config,
        "cortical_plans": args.plans,
        "cortical_checkpoint": args.cortical_checkpoint,
        "provisional_checkpoint": args.provisional_checkpoint,
        "abbc_to_instance_config": args.abbc_config,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, raw_path in artifact_arguments.items():
        path = raw_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{name} must be a file: {path}")
        artifacts[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    repository = args.repository.expanduser().resolve(strict=True)
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_scope": "development_OOF",
        "provisional_model": {
            "name": FROZEN_PROVISIONAL_MODEL,
            "fold": "all",
            "checkpoint": "checkpoint_final",
        },
        "cortical_model": {
            "folds": args.cortical_folds,
            "checkpoint": args.cortical_checkpoint_name,
        },
        "artifacts": artifacts,
        "code": {
            "repository": str(repository),
            "revision": revision,
            "dirty": dirty,
        },
    }
    _write_json_new(args.output, payload)
    _print_json(
        {
            "status": "PASS",
            "output": str(args.output.resolve()),
            "revision": revision,
            "dirty": dirty,
        }
    )
    return 0


def _command_export_nifti(args: argparse.Namespace) -> int:
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ModuleNotFoundError as error:
        raise ValueError(
            "export-nifti requires nibabel in the HPC runtime"
        ) from error
    result_path = args.result.expanduser().resolve(strict=True)
    with np.load(result_path, allow_pickle=False) as payload:
        required = {"full_instances", "cortical_instances"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"result bundle is missing {sorted(missing)}")
        full = np.asarray(payload["full_instances"])
        cortical = np.asarray(payload["cortical_instances"])
    if (
        full.ndim != 3
        or cortical.shape != full.shape
        or not np.issubdtype(full.dtype, np.integer)
        or not np.issubdtype(cortical.dtype, np.integer)
    ):
        raise ValueError("result instance arrays must be matching 3-D integers")
    if np.any(full < 0) or np.any(cortical < 0):
        raise ValueError("result instance arrays must be non-negative")
    if np.any((cortical > 0) & (full == 0)):
        raise ValueError("cortical instances lie outside full-instance support")

    working_reference_path = args.working_reference.expanduser().resolve(
        strict=True
    )
    working_reference = nib.load(str(working_reference_path))
    if len(working_reference.shape) != 3:
        raise ValueError("working reference must be a scalar 3-D NIfTI")
    xyz_shape = tuple(int(value) for value in full.shape[::-1])
    if tuple(working_reference.shape) != xyz_shape:
        raise ValueError(
            "working reference shape does not match ZYX result grid"
        )
    target = None
    original_reference_path = None
    if args.original_reference is not None:
        if not args.allow_resample_to_original:
            raise ValueError(
                "--original-reference requires --allow-resample-to-original"
            )
        original_reference_path = args.original_reference.expanduser().resolve(
            strict=True
        )
        original_reference = nib.load(str(original_reference_path))
        if len(original_reference.shape) != 3:
            raise ValueError("original reference must be a scalar 3-D NIfTI")
        target = (
            tuple(int(value) for value in original_reference.shape),
            original_reference.affine,
        )

    outputs = {
        "full_instances": args.output_full.expanduser().resolve(),
        "cortical_instances": args.output_cortical.expanduser().resolve(),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("full and cortical outputs must be different files")
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = args.diagnostics
    if diagnostics is None:
        diagnostics = outputs["full_instances"].with_name(
            outputs["full_instances"].name + ".export.json"
        )
    diagnostics = diagnostics.expanduser().resolve()
    if diagnostics.exists():
        raise FileExistsError(f"refusing to overwrite {diagnostics}")

    output_hashes: dict[str, str] = {}
    for role, array in (
        ("full_instances", full),
        ("cortical_instances", cortical),
    ):
        xyz = array.transpose(2, 1, 0).astype(np.uint32, copy=False)
        header = working_reference.header.copy()
        header.set_data_dtype(np.uint32)
        image = nib.Nifti1Image(
            xyz,
            working_reference.affine,
            header=header,
        )
        if target is not None:
            resampled = resample_from_to(
                image,
                target,
                order=0,
                mode="constant",
                cval=0,
            )
            data = np.rint(np.asanyarray(resampled.dataobj)).astype(np.uint32)
            output_image = nib.Nifti1Image(
                data,
                resampled.affine,
                header=resampled.header,
            )
            output_image.header.set_data_dtype(np.uint32)
        else:
            output_image = image
        nib.save(output_image, str(outputs[role]))
        output_hashes[role] = _sha256(outputs[role])

    _write_json_new(
        diagnostics,
        {
            "schema_version": SCHEMA_VERSION,
            "result": {"path": str(result_path), "sha256": _sha256(result_path)},
            "working_reference": {
                "path": str(working_reference_path),
                "sha256": _sha256(working_reference_path),
            },
            "original_reference": (
                None
                if original_reference_path is None
                else {
                    "path": str(original_reference_path),
                    "sha256": _sha256(original_reference_path),
                }
            ),
            "resampling": (
                "none"
                if target is None
                else "nibabel_affine_nearest_neighbor_order0"
            ),
            "coordinate_conversion": "postprocess_ZYX_to_nifti_XYZ",
            "outputs": {
                role: {"path": str(outputs[role]), "sha256": digest}
                for role, digest in output_hashes.items()
            },
        },
    )
    _print_json(
        {
            "status": "PASS",
            "outputs": {key: str(value) for key, value in outputs.items()},
            "diagnostics": str(diagnostics),
        }
    )
    return 0


def _oracle_bundle(path: Path) -> dict[str, np.ndarray]:
    source = path.expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "full_fragment_instances",
            "cortical_instances",
            "spacing_zyx",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"oracle bundle is missing arrays: {sorted(missing)}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _oracle_dataset_case(
    root: Path,
    case_id: str,
    *,
    direction_set: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = root.expanduser().resolve(strict=True)
    manifest_path = dataset / "continuity_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("dataset continuity_manifest.json has no cases array")
    matches = [case for case in cases if str(case.get("case_id")) == case_id]
    if len(matches) != 1:
        raise ValueError(
            f"case {case_id!r} occurs {len(matches)} times in the dataset manifest"
        )
    case = matches[0]
    arrays: dict[str, np.ndarray] = {}
    role_to_key = {
        "fragment_instances": "full_fragment_instances",
        "cortical_instances": "cortical_instances",
        "cortical_overlap": "overlap_mask",
    }
    verified_sidecars: dict[str, str] = {}
    for role, key in role_to_key.items():
        artifact = case["sidecars"][role]
        path = (dataset / artifact["path"]).resolve(strict=True)
        try:
            path.relative_to(dataset)
        except ValueError as error:
            raise ValueError(f"{role} path escapes Dataset778") from error
        observed_hash = _sha256(path)
        if observed_hash != artifact["sha256"]:
            raise ValueError(f"{case_id} {role} SHA-256 mismatch")
        value = read_nifti_array(path).transpose(2, 1, 0)
        arrays[key] = (value > 0) if key == "overlap_mask" else value
        verified_sidecars[role] = observed_hash
    spacing = np.asarray(case["spacing_xyz_mm"][::-1], dtype=np.float64)
    arrays["spacing_zyx"] = spacing
    schema = build_cortical_continuity_schema(
        spacing,
        direction_set=direction_set,
    )
    arrays["affinity_offsets_zyx"] = np.asarray(
        [offset.voxel_offset_zyx for offset in schema.affinity_offsets],
        dtype=np.int16,
    )
    descriptor = {
        "kind": "Dataset778_case",
        "dataset": str(dataset),
        "case_id": case_id,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_content_sha256": manifest.get("content_sha256"),
        "direction_set": direction_set,
        "affinity_channels": len(schema.affinity_offsets),
        "verified_sidecars": verified_sidecars,
    }
    return arrays, descriptor


def _command_oracle(args: argparse.Namespace) -> int:
    if args.input is not None:
        if args.case is not None:
            raise ValueError("--case is only valid with --dataset")
        arrays = _oracle_bundle(args.input)
        input_descriptor = {
            "kind": "NPZ",
            "path": str(args.input.expanduser().resolve(strict=True)),
            "sha256": _sha256(args.input.expanduser().resolve(strict=True)),
        }
    else:
        if args.case is None:
            raise ValueError("--dataset requires --case")
        arrays, input_descriptor = _oracle_dataset_case(
            args.dataset,
            args.case,
            direction_set=args.direction_set,
        )
    overlap = arrays.get("overlap_mask")
    manifest = controlled_event_manifest(
        arrays["full_fragment_instances"],
        arrays["cortical_instances"],
        arrays["spacing_zyx"],
        maximum_surface_distance_mm=args.maximum_surface_distance_mm,
        overlap_mask=overlap,
        maximum_multiway_events=args.maximum_multiway_events,
    )
    selected = [
        event
        for event in manifest.events
        if args.event is None or event.event_id in set(args.event)
    ]
    if args.event is not None:
        missing = set(args.event).difference(event.event_id for event in selected)
        if missing:
            raise ValueError(f"requested oracle events are absent: {sorted(missing)}")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "execute": bool(args.execute),
        "input": input_descriptor,
        "event_counts": {
            kind: sum(event.kind == kind for event in selected)
            for kind in ("binary", "multiway", "intact")
        },
        "coverage_ceiling": {
            "eligible_fragment_ids": list(manifest.eligible_fragment_ids),
            "cortex_missing_fragment_ids": list(
                manifest.cortex_missing_fragment_ids
            ),
        },
    }
    if not args.execute:
        _print_json(plan)
        return 0
    if args.output is None:
        raise ValueError("oracle --execute requires an explicit --output")
    offsets = arrays.get("affinity_offsets_zyx")
    if args.mode == "O2" and offsets is None:
        raise ValueError("O2 oracle bundle requires affinity_offsets_zyx")
    config = _load_config(args.config)
    event_results = []
    summary_rows: list[tuple[str, bool, bool, bool]] = []
    for position, event in enumerate(selected, start=1):
        print(
            f"[{position:03d}/{len(selected):03d}] {args.mode} {event.event_id}",
            file=sys.stderr,
            flush=True,
        )
        if args.mode == "O1":
            result = run_o1_event(
                event,
                arrays["full_fragment_instances"],
                arrays["cortical_instances"],
                arrays["spacing_zyx"],
                overlap_mask=overlap,
                iou_threshold=args.iou_threshold,
            )
        else:
            result = run_o2_event(
                event,
                arrays["full_fragment_instances"],
                arrays["cortical_instances"],
                offsets,
                arrays["spacing_zyx"],
                overlap_mask=overlap,
                config=config,
                iou_threshold=args.iou_threshold,
            )
        event_results.append(
            {
                "event": event,
                "evaluation": result.evaluation,
                "cortical_grouping_evaluation": (
                    result.cortical_grouping_evaluation
                ),
                "decisions": result.decisions,
            }
        )
        grouping = result.cortical_grouping_evaluation
        exact_grouping = bool(
            grouping.prediction_count == grouping.ground_truth_count
            and grouping.all_child_recovery
        )
        summary_rows.append(
            (
                event.kind,
                bool(result.evaluation.all_child_recovery),
                bool(result.evaluation.intact_false_split),
                exact_grouping,
            )
        )
    summary = _oracle_summary(args.mode, summary_rows)
    payload = {
        **plan,
        "execute": True,
        "manifest": manifest,
        "config": config.to_dict(),
        "summary": summary,
        "results": event_results,
    }
    _write_json_new(args.output, payload)
    _print_json(
        {
            "status": "PASS",
            "events": len(event_results),
            "gate_status": summary["gate_status"],
            "output": str(args.output.resolve()),
        }
    )
    return 0


def _oracle_summary(
    mode: str,
    rows: Sequence[tuple[str, bool, bool, bool]],
) -> dict[str, Any]:
    def rate(values: Sequence[bool]) -> float | None:
        return None if not values else float(np.mean(values))

    binary_recovery = [recovered for kind, recovered, _, _ in rows if kind == "binary"]
    multiway_recovery = [
        recovered for kind, recovered, _, _ in rows if kind == "multiway"
    ]
    intact_false_split = [
        false_split for kind, _, false_split, _ in rows if kind == "intact"
    ]
    grouping_exact = [
        exact for kind, _, _, exact in rows if kind in {"binary", "multiway"}
    ]
    metrics = {
        "binary_all_child_recovery": rate(binary_recovery),
        "multiway_all_child_recovery": rate(multiway_recovery),
        "intact_false_split_rate": rate(intact_false_split),
        "exact_k_identity_grouping": rate(grouping_exact),
    }
    checks: dict[str, bool | None] = {
        "intact_false_split_le_0_05": (
            None
            if metrics["intact_false_split_rate"] is None
            else metrics["intact_false_split_rate"] <= 0.05
        )
    }
    if mode == "O1":
        checks.update(
            {
                "binary_recovery_ge_0_90": (
                    None
                    if metrics["binary_all_child_recovery"] is None
                    else metrics["binary_all_child_recovery"] >= 0.90
                ),
                "multiway_recovery_ge_0_75": (
                    None
                    if metrics["multiway_all_child_recovery"] is None
                    else metrics["multiway_all_child_recovery"] >= 0.75
                ),
            }
        )
    else:
        checks["exact_k_identity_ge_0_95"] = (
            None
            if metrics["exact_k_identity_grouping"] is None
            else metrics["exact_k_identity_grouping"] >= 0.95
        )
    evaluated = [value for value in checks.values() if value is not None]
    gate_status = (
        "NOT_EVALUATED"
        if len(evaluated) != len(checks)
        else ("PASS" if all(evaluated) else "NO_GO")
    )
    return {
        "metrics": metrics,
        "gate_checks": checks,
        "gate_status": gate_status,
    }


def _aggregate_oracle_files(
    paths: Sequence[Path],
    *,
    expected_mode: str,
    expected_cases: int,
    expected_direction_set: str | None = None,
) -> dict[str, Any]:
    if not paths:
        raise ValueError(f"no {expected_mode} oracle files were supplied")
    by_case: dict[str, dict[str, Any]] = {}
    file_records: list[dict[str, str]] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("mode") != expected_mode or value.get("execute") is not True:
            raise ValueError(f"{path} is not an executed {expected_mode} result")
        descriptor = value.get("input")
        case_id = (
            str(descriptor.get("case_id"))
            if isinstance(descriptor, Mapping)
            and descriptor.get("kind") == "Dataset778_case"
            else ""
        )
        if not case_id:
            raise ValueError(
                f"{path} lacks a Dataset778 case identity; formal gates require "
                "oracle --dataset/--case inputs"
            )
        if (
            expected_direction_set is not None
            and descriptor.get("direction_set") != expected_direction_set
        ):
            raise ValueError(
                f"{path} direction_set={descriptor.get('direction_set')!r}; "
                f"expected {expected_direction_set!r}"
            )
        if case_id in by_case:
            raise ValueError(f"duplicate {expected_mode} result for case {case_id}")
        results = value.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{path} has no oracle results array")
        manifest_events = value.get("manifest", {}).get("events")
        if not isinstance(manifest_events, list):
            raise ValueError(f"{path} has no controlled-event manifest")
        expected_event_ids = {
            str(event["event_id"]) for event in manifest_events
        }
        observed_event_ids = {
            str(result["event"]["event_id"]) for result in results
        }
        if observed_event_ids != expected_event_ids:
            raise ValueError(
                f"{path} is a partial oracle run; formal gates require every "
                "controlled event"
            )
        by_case[case_id] = value
        file_records.append(
            {"path": str(path), "sha256": _sha256(path), "case_id": case_id}
        )
    if len(by_case) != expected_cases:
        raise ValueError(
            f"{expected_mode} expected {expected_cases} cases, got {len(by_case)}"
        )

    event_rows: dict[str, list[bool]] = {
        "binary_recovery": [],
        "multiway_recovery": [],
        "intact_false_split": [],
        "exact_k_identity": [],
    }
    patient_rows: dict[str, list[float]] = {
        key: [] for key in event_rows
    }
    for case_id in sorted(by_case):
        local: dict[str, list[bool]] = {key: [] for key in event_rows}
        for result in by_case[case_id]["results"]:
            event = result["event"]
            kind = str(event["kind"])
            evaluation = result["evaluation"]
            grouping = result["cortical_grouping_evaluation"]
            if kind == "binary":
                local["binary_recovery"].append(
                    bool(evaluation["all_child_recovery"])
                )
            elif kind == "multiway":
                local["multiway_recovery"].append(
                    bool(evaluation["all_child_recovery"])
                )
            elif kind == "intact":
                local["intact_false_split"].append(
                    bool(evaluation["intact_false_split"])
                )
            if kind in {"binary", "multiway"}:
                local["exact_k_identity"].append(
                    bool(
                        grouping["prediction_count"]
                        == grouping["ground_truth_count"]
                        and grouping["all_child_recovery"]
                    )
                )
        for name, values in local.items():
            event_rows[name].extend(values)
            if values:
                patient_rows[name].append(float(np.mean(values)))
    return {
        "case_count": len(by_case),
        "event_counts": {
            name: len(values) for name, values in event_rows.items()
        },
        "event_micro": {
            name: (None if not values else float(np.mean(values)))
            for name, values in event_rows.items()
        },
        "patient_macro": {
            name: (None if not values else float(np.mean(values)))
            for name, values in patient_rows.items()
        },
        "files": file_records,
    }


def _command_gate(args: argparse.Namespace) -> int:
    o1 = _aggregate_oracle_files(
        args.o1,
        expected_mode="O1",
        expected_cases=args.expected_cases,
    )
    axial = _aggregate_oracle_files(
        args.o2_axial19,
        expected_mode="O2",
        expected_cases=args.expected_cases,
        expected_direction_set=AXIAL_19_DIRECTION_SET,
    )
    dense = (
        None
        if not args.o2_dense39
        else _aggregate_oracle_files(
            args.o2_dense39,
            expected_mode="O2",
            expected_cases=args.expected_cases,
            expected_direction_set=DENSE_39_DIRECTION_SET,
        )
    )
    o1_metrics = o1["patient_macro"]
    axial_metrics = axial["patient_macro"]
    axial_pass = (
        axial_metrics["exact_k_identity"] is not None
        and axial_metrics["exact_k_identity"] >= 0.95
    )
    selected_name = "axial19" if axial_pass else "dense39"
    selected = axial if axial_pass else dense
    reasons: list[str] = []

    def require(metric: Any, predicate: bool, reason: str) -> bool:
        if metric is None or not predicate:
            reasons.append(reason)
            return False
        return True

    o1_binary_ok = require(
        o1_metrics["binary_recovery"],
        o1_metrics["binary_recovery"] is not None
        and o1_metrics["binary_recovery"] >= 0.90,
        "o1_binary_recovery_below_0.90_or_missing",
    )
    o1_multiway_ok = require(
        o1_metrics["multiway_recovery"],
        o1_metrics["multiway_recovery"] is not None
        and o1_metrics["multiway_recovery"] >= 0.75,
        "o1_multiway_recovery_below_0.75_or_missing",
    )
    o1_intact_ok = require(
        o1_metrics["intact_false_split"],
        o1_metrics["intact_false_split"] is not None
        and o1_metrics["intact_false_split"] <= 0.05,
        "o1_intact_false_split_above_0.05_or_missing",
    )
    o2_ok = False
    o2_intact_ok = False
    if selected is None:
        reasons.append("axial19_below_gate_dense39_results_required")
    else:
        selected_metrics = selected["patient_macro"]
        o2_ok = require(
            selected_metrics["exact_k_identity"],
            selected_metrics["exact_k_identity"] is not None
            and selected_metrics["exact_k_identity"] >= 0.95,
            f"{selected_name}_exact_k_identity_below_0.95_or_missing",
        )
        o2_intact_ok = require(
            selected_metrics["intact_false_split"],
            selected_metrics["intact_false_split"] is not None
            and selected_metrics["intact_false_split"] <= 0.05,
            f"{selected_name}_intact_false_split_above_0.05_or_missing",
        )
    propagation_stop = bool(
        o1_metrics["binary_recovery"] is None
        or o1_metrics["binary_recovery"] < 0.70
    )
    neural_training_allowed = bool(
        o1_binary_ok
        and o1_multiway_ok
        and o1_intact_ok
        and o2_ok
        and o2_intact_ok
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_scope": "development_OOF_oracle",
        "aggregation_for_gates": "patient_macro",
        "expected_cases": args.expected_cases,
        "o1": o1,
        "o2_axial19": axial,
        "o2_dense39": dense,
        "selected_o2_direction_set": selected_name,
        "propagation_stop": propagation_stop,
        "neural_training_allowed": neural_training_allowed,
        "reasons": sorted(set(reasons)),
    }
    _write_json_new(args.output, payload)
    _print_json(
        {
            "status": "PASS" if neural_training_allowed else "NO_GO",
            "neural_training_allowed": neural_training_allowed,
            "propagation_stop": propagation_stop,
            "selected_o2_direction_set": selected_name,
            "output": str(args.output.resolve()),
            "reasons": payload["reasons"],
        }
    )
    return 0


def _add_build_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("build", help="build Dataset778 (dry-run by default)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--separator-mm", type=float, default=1.0)
    parser.add_argument("--rim-positive-mm", type=float, default=2.0)
    parser.add_argument("--rim-negative-mm", type=float, default=4.0)
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    parser.add_argument(
        "--distance-backend",
        choices=("auto", "numpy", "scipy"),
        default="auto",
    )
    parser.add_argument("--chunk-spatial-voxels", type=int, default=500_000)
    parser.set_defaults(handler=_command_build)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cortical-continuity data, oracle, refinement, and evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build_parser(subparsers)

    splits = subparsers.add_parser("splits", help="print or save the frozen fivefold")
    splits.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    splits.add_argument("--output", type=Path)
    splits.set_defaults(handler=_command_splits)

    refine = subparsers.add_parser("refine", help="refine a working-grid NPZ bundle")
    refine.add_argument("--input", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--diagnostics", type=Path)
    refine.add_argument("--method", choices=("ca", "separator"), default="ca")
    refine.add_argument("--config", type=Path)
    refine.add_argument("--scorer", type=Path)
    refine.add_argument(
        "--allow-fixed-thresholds",
        action="store_true",
        help="explicit engineering mode; formal C+A runs require a calibrated scorer",
    )
    refine.add_argument("--run-metadata", type=Path)
    refine.add_argument("--formal", action="store_true")
    refine.set_defaults(handler=_command_refine)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate explicit instance arrays"
    )
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--prediction-key")
    evaluate.add_argument("--ground-truth-key")
    evaluate.add_argument("--valid-mask", type=Path)
    evaluate.add_argument("--valid-mask-key")
    evaluate.add_argument(
        "--iou-threshold",
        type=float,
        action="append",
        default=None,
        help="repeatable; defaults to 0.5 and 0.7",
    )
    evaluate.add_argument("--expected-ground-truth-ids")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_command_evaluate)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="fit the OOF split/stop scorer under the intact-risk constraint",
    )
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--report", type=Path)
    calibrate.add_argument("--l2-penalty", type=float, default=1.0)
    calibrate.add_argument(
        "--max-intact-false-split-rate",
        type=float,
        default=0.05,
    )
    calibrate.add_argument("--min-samples", type=int, default=20)
    calibrate.add_argument("--min-intact-samples", type=int, default=10)
    calibrate.add_argument("--max-iterations", type=int, default=100)
    calibrate.add_argument("--calibration-bins", type=int, default=10)
    calibrate.set_defaults(handler=_command_calibrate)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="compute deterministic patient-macro metrics and bootstrap CIs",
    )
    aggregate.add_argument("--input", type=Path, required=True)
    aggregate.add_argument("--metric", action="append", required=True)
    aggregate.add_argument("--bootstrap-iterations", type=int, default=2000)
    aggregate.add_argument("--random-state", type=int, default=20260729)
    aggregate.add_argument("--output", type=Path)
    aggregate.set_defaults(handler=_command_aggregate)

    provenance = subparsers.add_parser(
        "provenance",
        help="hash every artifact required by a formal refinement run",
    )
    provenance.add_argument("--source-manifest", type=Path, required=True)
    provenance.add_argument("--splits", type=Path, required=True)
    provenance.add_argument("--continuity-config", type=Path, required=True)
    provenance.add_argument("--plans", type=Path, required=True)
    provenance.add_argument("--cortical-checkpoint", type=Path, required=True)
    provenance.add_argument("--provisional-checkpoint", type=Path, required=True)
    provenance.add_argument("--abbc-config", type=Path, required=True)
    provenance.add_argument("--repository", type=Path, default=Path.cwd())
    provenance.add_argument("--cortical-folds", default="0,1,2,3,4")
    provenance.add_argument(
        "--cortical-checkpoint-name",
        default="checkpoint_final",
    )
    provenance.add_argument("--output", type=Path, required=True)
    provenance.set_defaults(handler=_command_provenance)

    export_nifti = subparsers.add_parser(
        "export-nifti",
        help="export working-grid instances with explicit reference geometry",
    )
    export_nifti.add_argument("--result", type=Path, required=True)
    export_nifti.add_argument(
        "--working-reference",
        type=Path,
        required=True,
    )
    export_nifti.add_argument("--original-reference", type=Path)
    export_nifti.add_argument(
        "--allow-resample-to-original",
        action="store_true",
    )
    export_nifti.add_argument("--output-full", type=Path, required=True)
    export_nifti.add_argument("--output-cortical", type=Path, required=True)
    export_nifti.add_argument("--diagnostics", type=Path)
    export_nifti.set_defaults(handler=_command_export_nifti)

    oracle = subparsers.add_parser(
        "oracle", help="plan or execute controlled O1/O2 events"
    )
    oracle_source = oracle.add_mutually_exclusive_group(required=True)
    oracle_source.add_argument("--input", type=Path)
    oracle_source.add_argument("--dataset", type=Path)
    oracle.add_argument("--case")
    oracle.add_argument(
        "--direction-set",
        choices=(AXIAL_19_DIRECTION_SET, DENSE_39_DIRECTION_SET),
        default=AXIAL_19_DIRECTION_SET,
    )
    oracle.add_argument("--mode", choices=("O1", "O2"), required=True)
    oracle.add_argument("--execute", action="store_true")
    oracle.add_argument("--output", type=Path)
    oracle.add_argument("--config", type=Path)
    oracle.add_argument("--event", action="append")
    oracle.add_argument("--maximum-surface-distance-mm", type=float, default=2.0)
    oracle.add_argument("--maximum-multiway-events", type=int, default=100)
    oracle.add_argument("--iou-threshold", type=float, default=0.5)
    oracle.set_defaults(handler=_command_oracle)

    gate = subparsers.add_parser(
        "gate",
        help="aggregate 68-case O1/O2 results and freeze the neural no-go gate",
    )
    gate.add_argument("--o1", type=Path, nargs="+", required=True)
    gate.add_argument(
        "--o2-axial19",
        type=Path,
        nargs="+",
        required=True,
    )
    gate.add_argument("--o2-dense39", type=Path, nargs="+")
    gate.add_argument("--expected-cases", type=int, default=68)
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(handler=_command_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "iou_threshold", None) is None:
        args.iou_threshold = [0.5, 0.7]
    try:
        return int(args.handler(args))
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
