"""I/O contract helpers for the repair segmentation service.

The service boundary deliberately uses the original clinical filenames rather
than nnU-Net's ``*_0000`` convention. ABB-C is an internal model
representation and is never part of this boundary:

    ct.nii.gz -> seg.nii.gz

Batch inference remains supported; in that mode each input keeps its stem.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


NIFTI_SUFFIX = ".nii.gz"
CONTRACT_INPUT_NAME = "ct.nii.gz"
CONTRACT_OUTPUT_NAME = "seg.nii.gz"
NAMED_SEGMENTATION_DIR = "segmentations"


def nifti_stem(path: Path) -> str:
    """Return a filename without the compound ``.nii.gz`` suffix."""
    if not path.name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"Expected a {NIFTI_SUFFIX} file, got: {path}")
    return path.name[: -len(NIFTI_SUFFIX)]


def resolve_inputs(input_path: str | Path) -> list[Path]:
    """Resolve a contract CT file or a directory of NIfTI inputs."""
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if not path.name.endswith(NIFTI_SUFFIX):
            raise ValueError(f"Input must be a {NIFTI_SUFFIX} file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    contract_ct = path / CONTRACT_INPUT_NAME
    if contract_ct.is_file():
        return [contract_ct]

    inputs = sorted(path.glob(f"*{NIFTI_SUFFIX}"))
    if not inputs:
        raise FileNotFoundError(f"No {NIFTI_SUFFIX} inputs found in: {path}")
    return inputs


def resolve_output_paths(inputs: list[Path], output_dir: str | Path) -> list[Path]:
    """Map every single-case invocation to the fixed downstream filename."""
    destination = Path(output_dir).expanduser().resolve()
    if len(inputs) == 1:
        return [destination / CONTRACT_OUTPUT_NAME]
    return [destination / f"{nifti_stem(path)}_seg.nii.gz" for path in inputs]


def default_output_directory(input_path: str | Path) -> Path:
    """Return the FILE_UPLOAD container where downstream stages search."""
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_laterality(input_path: str | Path, explicit: str | None = None) -> str:
    """Resolve L/R from a deployment override or the case ``config.json``."""

    candidate = explicit
    case_dir = default_output_directory(input_path)
    if candidate is None:
        config_path = case_dir / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as stream:
                config = json.load(stream)
            if not isinstance(config, dict):
                raise ValueError(f"UI config must be a JSON object: {config_path}")
            values = [
                config.get("side"),
                config.get("laterality"),
                config.get("selectedSide"),
                config.get("selected_side"),
            ]
            ui = config.get("ui")
            if isinstance(ui, dict):
                values.extend((ui.get("side"), ui.get("laterality")))
            candidate = next((value for value in values if value is not None), None)
    if candidate is None:
        candidate = os.environ.get("SEGMENTATION_SIDE", "L")

    normalized = str(candidate).strip().lower()
    if normalized in {"l", "left", "links"}:
        return "L"
    if normalized in {"r", "right", "rechts"}:
        return "R"
    raise ValueError(f"Unsupported segmentation laterality: {candidate!r}")


def resolve_bone_masks_directory(
    input_path: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    """Locate semantic bone masks produced by segmentation post-processing."""

    case_dir = default_output_directory(input_path)
    configured = explicit or os.environ.get("BONE_MASKS_DIR")
    candidates = []
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        case_dir / name
        for name in ("postprocessing", "totalsegmentator", "bone_masks")
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Bone-name post-processing masks are required before publishing "
        f"segmentation output; searched: {rendered}"
    )


def publish_contract_output(output_file: str | Path, contract_dir: str | Path) -> Path:
    """Publish one result under the exact name consumed by downstream stages."""
    source = Path(output_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Segmentation output does not exist: {source}")
    contract_file = Path(contract_dir).expanduser().resolve() / CONTRACT_OUTPUT_NAME
    contract_file.parent.mkdir(parents=True, exist_ok=True)
    if source != contract_file:
        shutil.copy2(source, contract_file)
    if not contract_file.is_file():
        raise RuntimeError(f"Required segmentation was not published: {contract_file}")
    return contract_file


def publish_named_segmentations(
    segmentation_dir: str | Path, contract_dir: str | Path
) -> Path:
    """Publish the KOMO-compatible named binary masks for Repositioning."""

    source = Path(segmentation_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Named segmentation directory does not exist: {source}")
    files = sorted(source.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"Named segmentation directory is empty: {source}")

    destination = (
        Path(contract_dir).expanduser().resolve() / NAMED_SEGMENTATION_DIR
    )
    destination.mkdir(parents=True, exist_ok=True)
    if source != destination:
        for stale in destination.glob("*.nii.gz"):
            stale.unlink()
        for file_path in files:
            shutil.copy2(file_path, destination / file_path.name)
    return destination
