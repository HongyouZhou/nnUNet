"""Freeze the cortical-continuity schema into an nnU-Net plans JSON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import (
    AXIAL_19_DIRECTION_SET,
    DENSE_39_DIRECTION_SET,
    SCHEMA_PLANS_KEY,
    build_cortical_continuity_schema,
)


FORMAL_TARGET_SPACING_ZYX = (0.5, 0.5, 0.5)
FORMAL_PREPROCESSOR = "ChariteCorticalPreprocessor"


def freeze_cortical_continuity_plans(
    plans: Mapping[str, Any],
    *,
    configuration: str = "3d_fullres",
    direction_set: str = AXIAL_19_DIRECTION_SET,
    data_identifier: str | None = None,
) -> dict[str, Any]:
    """Return a validated copy with a versioned top-level flat-head schema."""

    frozen = deepcopy(dict(plans))
    try:
        configuration_value = frozen["configurations"][configuration]
    except KeyError as exc:
        raise KeyError(f"Plans do not contain configuration {configuration!r}") from exc
    spacing = tuple(float(i) for i in configuration_value["spacing"])
    if len(spacing) != 3 or not np.allclose(
        spacing,
        FORMAL_TARGET_SPACING_ZYX,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Formal cortical continuity plans require target spacing "
            f"{FORMAL_TARGET_SPACING_ZYX} mm z-y-x; got {spacing}"
        )
    if configuration_value.get("preprocessor_name") != FORMAL_PREPROCESSOR:
        raise ValueError(
            f"Configuration {configuration!r} must use {FORMAL_PREPROCESSOR}; "
            f"got {configuration_value.get('preprocessor_name')!r}"
        )
    if data_identifier is None:
        data_identifier = f"nnUNetResEncUNetMPlansContinuity_{configuration}"
    if not data_identifier or "/" in data_identifier or "\\" in data_identifier:
        raise ValueError(
            f"Invalid preprocessing data identifier: {data_identifier!r}"
        )
    # ResEncUNetPlanner reuses nnUNetPlans_3d_fullres for ordinary
    # architecture-only variants. Continuity changes the preprocessor and
    # packed targets, so it must never share the separator output directory.
    configuration_value["data_identifier"] = data_identifier
    schema = build_cortical_continuity_schema(
        spacing,
        direction_set=direction_set,
    )
    existing = frozen.get(SCHEMA_PLANS_KEY)
    if existing is not None and existing != schema.to_dict():
        raise ValueError(
            f"Plans already contain a different {SCHEMA_PLANS_KEY}; "
            "write a separate plans file instead of silently changing a frozen schema"
        )
    frozen[SCHEMA_PLANS_KEY] = schema.to_dict()
    frozen["cortical_continuity_training_schedule"] = {
        "num_epochs": 500,
        "num_iterations_per_epoch": 250,
        "initial_lr": 1e-3,
        "deep_supervision": False,
        "sampling_percent": {
            "contact": 40,
            "instance": 30,
            "support": 20,
            "random": 10,
        },
    }
    return frozen


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
        json.dump(value, stream, indent=2, sort_keys=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate 0.5-mm Charité plans and freeze the C+A head schema"
    )
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument(
        "--direction-set",
        choices=(AXIAL_19_DIRECTION_SET, DENSE_39_DIRECTION_SET),
        default=AXIAL_19_DIRECTION_SET,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a separate plans JSON; default updates --plans after creating a backup",
    )
    parser.add_argument(
        "--data-identifier",
        help=(
            "preprocessed folder identifier; defaults to "
            "<output-plans-stem>_<configuration>"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.plans.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Plans JSON must contain an object")
    if args.output is None:
        output = source
        backup = source.with_name(f"{source.stem}.before_cortical_continuity{source.suffix}")
        if not backup.exists():
            shutil.copy2(source, backup)
        print(f"Preserved original plans at {backup}")
    else:
        output = args.output.expanduser().resolve()
        if output == source:
            raise ValueError("Use the default mode for in-place writing so that a backup is preserved")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite explicit output {output}")
    data_identifier = (
        args.data_identifier
        if args.data_identifier is not None
        else f"{output.stem}_{args.configuration}"
    )
    frozen = freeze_cortical_continuity_plans(
        value,
        configuration=args.configuration,
        direction_set=args.direction_set,
        data_identifier=data_identifier,
    )
    _write_json_atomic(output, frozen)
    print(
        f"Wrote {args.direction_set} cortical-continuity schema "
        f"({frozen[SCHEMA_PLANS_KEY]['heads'][-1]['stop']} logits) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
