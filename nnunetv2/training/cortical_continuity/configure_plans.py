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

from nnunetv2.preprocessing.preprocessors.charite_supervision_roi import (
    CHARITE_ROI_PLANS_KEY,
    roi_plans_contract,
)

from .schema import (
    AXIAL_19_DIRECTION_SET,
    DENSE_39_DIRECTION_SET,
    SCHEMA_PLANS_KEY,
    build_cortical_continuity_schema,
)
from nnunetv2.training.cortical_separator_prior.contract import (
    PRIOR_PLANS_KEY,
    prior_plans_contract,
)


FORMAL_TARGET_SPACING_ZYX = (0.5, 0.5, 0.5)
FORMAL_PREPROCESSOR = "ChariteCorticalPreprocessor"
FORMAL_SEPARATOR_PREPROCESSOR = "ChariteSeparatorPreprocessor"
FORMAL_DENSITY_PRIOR_SEPARATOR_PREPROCESSOR = (
    "ChariteDensityPriorSeparatorPreprocessor"
)


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
    # Every custom target contract gets an independent preprocessed directory.
    # Continuity must never share its packed targets with the separator.
    configuration_value["data_identifier"] = data_identifier
    frozen[CHARITE_ROI_PLANS_KEY] = roi_plans_contract(
        training_source="fragment_support_sidecar",
    )
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


def freeze_charite_separator_plans(
    plans: Mapping[str, Any],
    *,
    configuration: str = "3d_fullres",
    data_identifier: str | None = None,
) -> dict[str, Any]:
    frozen = deepcopy(dict(plans))
    try:
        configuration_value = frozen["configurations"][configuration]
    except KeyError as exc:
        raise KeyError(f"Plans do not contain configuration {configuration!r}") from exc
    spacing = tuple(float(value) for value in configuration_value["spacing"])
    if len(spacing) != 3 or not np.allclose(
        spacing, FORMAL_TARGET_SPACING_ZYX, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "Formal Charite separator plans require target spacing "
            f"{FORMAL_TARGET_SPACING_ZYX} mm z-y-x; got {spacing}"
        )
    if configuration_value.get("preprocessor_name") != FORMAL_SEPARATOR_PREPROCESSOR:
        raise ValueError(
            f"Configuration {configuration!r} must use {FORMAL_SEPARATOR_PREPROCESSOR}; "
            f"got {configuration_value.get('preprocessor_name')!r}"
        )
    if data_identifier is None:
        data_identifier = f"nnUNetResEncUNetMPlansSeparator_{configuration}"
    if not data_identifier or "/" in data_identifier or "\\" in data_identifier:
        raise ValueError(f"Invalid preprocessing data identifier: {data_identifier!r}")
    configuration_value["data_identifier"] = data_identifier
    frozen[CHARITE_ROI_PLANS_KEY] = roi_plans_contract(
        training_source="separator_semantic_nonbackground",
    )
    return frozen


def freeze_charite_density_prior_separator_plans(
    plans: Mapping[str, Any],
    *,
    configuration: str = "3d_fullres",
    data_identifier: str | None = None,
) -> dict[str, Any]:
    """Freeze the paired control/prior target contract into separate plans."""

    frozen = deepcopy(dict(plans))
    try:
        configuration_value = frozen["configurations"][configuration]
    except KeyError as exc:
        raise KeyError(f"Plans do not contain configuration {configuration!r}") from exc
    spacing = tuple(float(value) for value in configuration_value["spacing"])
    if len(spacing) != 3 or not np.allclose(
        spacing, FORMAL_TARGET_SPACING_ZYX, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "Formal density-prior separator plans require target spacing "
            f"{FORMAL_TARGET_SPACING_ZYX} mm z-y-x; got {spacing}"
        )
    if (
        configuration_value.get("preprocessor_name")
        != FORMAL_DENSITY_PRIOR_SEPARATOR_PREPROCESSOR
    ):
        raise ValueError(
            f"Configuration {configuration!r} must use "
            f"{FORMAL_DENSITY_PRIOR_SEPARATOR_PREPROCESSOR}; got "
            f"{configuration_value.get('preprocessor_name')!r}"
        )
    if data_identifier is None:
        data_identifier = f"nnUNetResEncUNetMPlansSeparatorDensityPrior_{configuration}"
    if not data_identifier or "/" in data_identifier or "\\" in data_identifier:
        raise ValueError(f"Invalid preprocessing data identifier: {data_identifier!r}")
    configuration_value["data_identifier"] = data_identifier
    frozen[CHARITE_ROI_PLANS_KEY] = roi_plans_contract(
        training_source="fragment_support_sidecar",
    )
    contract = prior_plans_contract(data_identifier=data_identifier)
    existing = frozen.get(PRIOR_PLANS_KEY)
    if existing is not None and existing != contract:
        raise ValueError(
            f"Plans already contain a different {PRIOR_PLANS_KEY} contract"
        )
    frozen[PRIOR_PLANS_KEY] = contract
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--separator",
        action="store_true",
        help="freeze the ROI-aware separator contract instead of the C+A schema",
    )
    mode.add_argument(
        "--separator-density-prior",
        action="store_true",
        help="freeze the paired density-prior separator target contract",
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
        backup_tag = (
            "charite_separator_density_prior"
            if args.separator_density_prior
            else "charite_separator"
            if args.separator
            else "cortical_continuity"
        )
        backup = source.with_name(f"{source.stem}.before_{backup_tag}{source.suffix}")
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
    if args.separator_density_prior:
        frozen = freeze_charite_density_prior_separator_plans(
            value,
            configuration=args.configuration,
            data_identifier=data_identifier,
        )
    elif args.separator:
        frozen = freeze_charite_separator_plans(
            value,
            configuration=args.configuration,
            data_identifier=data_identifier,
        )
    else:
        frozen = freeze_cortical_continuity_plans(
            value,
            configuration=args.configuration,
            direction_set=args.direction_set,
            data_identifier=data_identifier,
        )
    _write_json_atomic(output, frozen)
    if args.separator_density_prior:
        print(f"Wrote density-prior Charite separator plans to {output}")
    elif args.separator:
        print(f"Wrote ROI-aware Charite separator plans to {output}")
    else:
        print(
            f"Wrote {args.direction_set} cortical-continuity schema "
            f"({frozen[SCHEMA_PLANS_KEY]['heads'][-1]['stop']} logits) to {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
