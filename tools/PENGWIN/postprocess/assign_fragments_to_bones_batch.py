#!/usr/bin/env python3
"""Apply fragment-to-bone assignment to a deployment output directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from tools.PENGWIN.postprocess.assign_fragments_to_bones import (
    AssignmentConfig,
    BONE_ORDER,
    build_assignment_report,
    discover_totalsegmentator_masks,
)


def discover_instance_files(instances_root: str | Path) -> list[Path]:
    """Return deployment instance maps in deterministic order."""

    root = Path(instances_root)
    if not root.is_dir():
        raise FileNotFoundError(f"instances root does not exist: {root}")

    instance_files = sorted(root.rglob("*_instance.nii.gz"))
    if not instance_files:
        raise FileNotFoundError(f"no *_instance.nii.gz files found under {root}")
    return instance_files


def _write_json_atomic(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def process_assignment_directories(
    instances_root: str | Path,
    totalseg_root: str | Path,
    *,
    output_filename: str = "instance_classes.json",
    config: AssignmentConfig | None = None,
    chunk_depth: int = 128,
    affine_atol: float = 1e-4,
    require_all_masks: bool = True,
) -> list[Path]:
    """Pair instance case directories with TotalSegmentator case directories."""

    if Path(output_filename).name != output_filename:
        raise ValueError("output_filename must be a filename without directory components")

    instance_root_path = Path(instances_root)
    totalseg_root_path = Path(totalseg_root)
    if not totalseg_root_path.is_dir():
        raise FileNotFoundError(f"TotalSegmentator root does not exist: {totalseg_root_path}")

    written_outputs: list[Path] = []
    for instance_path in discover_instance_files(instance_root_path):
        relative_case_directory = instance_path.parent.relative_to(instance_root_path)
        totalseg_case_directory = totalseg_root_path / relative_case_directory
        discovered_masks = discover_totalsegmentator_masks(totalseg_case_directory)
        missing_masks = [bone for bone in BONE_ORDER if not discovered_masks[bone]]
        if require_all_masks and missing_masks:
            missing = ", ".join(missing_masks)
            raise FileNotFoundError(
                f"case {relative_case_directory}: missing TotalSegmentator masks: {missing} "
                f"in {totalseg_case_directory}"
            )

        report = build_assignment_report(
            instance_path,
            totalseg_case_directory,
            config=config,
            chunk_depth=chunk_depth,
            affine_atol=affine_atol,
        )
        output_path = instance_path.parent / output_filename
        _write_json_atomic(report, output_path)
        written_outputs.append(output_path)
        print(
            f"{relative_case_directory}: assigned "
            f"{report['summary']['assigned_instances']}/"
            f"{report['summary']['total_instances']} fragments; wrote {output_path}"
        )

    return written_outputs


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign all deployment fragment instance maps to bone classes by "
            "pairing case directories under the instance and TotalSegmentator roots."
        )
    )
    parser.add_argument("--instances-root", required=True, type=Path)
    parser.add_argument("--totalseg-root", required=True, type=Path)
    parser.add_argument("--output-filename", default="instance_classes.json")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument("--chunk-depth", type=int, default=128)
    parser.add_argument("--affine-atol", type=float, default=1e-4)
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="Write unknown assignments instead of failing when a bone mask is absent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        outputs = process_assignment_directories(
            args.instances_root,
            args.totalseg_root,
            output_filename=args.output_filename,
            config=AssignmentConfig(
                min_overlap=args.min_overlap,
                min_margin=args.min_margin,
            ),
            chunk_depth=args.chunk_depth,
            affine_atol=args.affine_atol,
            require_all_masks=not args.allow_missing_masks,
        )
    except (OSError, ValueError) as error:
        print(f"fragment-to-bone batch assignment failed: {error}", file=sys.stderr)
        return 1

    print(f"Completed fragment-to-bone assignment for {len(outputs)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
