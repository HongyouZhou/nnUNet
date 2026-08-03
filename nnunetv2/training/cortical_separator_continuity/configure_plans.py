"""Freeze the independent voxel-continuity separator plans contract."""

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

from .contract import (
    CONTINUITY_PLANS_KEY,
    FORMAL_PREPROCESSOR,
    FORMAL_TARGET_SPACING_ZYX,
    continuity_plans_contract,
)


FORMAL_PLANS_NAME = "nnUNetResEncUNetMPlansSeparatorContinuityPrior"


def freeze_separator_continuity_plans(
    plans: Mapping[str, Any],
    *,
    configuration: str = "3d_fullres",
    data_identifier: str | None = None,
) -> dict[str, Any]:
    frozen = deepcopy(dict(plans))
    try:
        configuration_value = frozen["configurations"][configuration]
    except KeyError as error:
        raise KeyError(f"Plans do not contain configuration {configuration!r}") from error
    spacing = tuple(float(value) for value in configuration_value["spacing"])
    if len(spacing) != 3 or not np.allclose(
        spacing, FORMAL_TARGET_SPACING_ZYX, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "Separator continuity plans require frozen 0.5-mm isotropic spacing; "
            f"got {spacing}"
        )
    if configuration_value.get("preprocessor_name") != FORMAL_PREPROCESSOR:
        raise ValueError(
            f"Configuration must use {FORMAL_PREPROCESSOR}; got "
            f"{configuration_value.get('preprocessor_name')!r}"
        )
    if data_identifier is None:
        data_identifier = f"{FORMAL_PLANS_NAME}_{configuration}"
    if not data_identifier or "/" in data_identifier or "\\" in data_identifier:
        raise ValueError(f"Invalid data identifier {data_identifier!r}")
    configuration_value["data_identifier"] = data_identifier
    frozen[CHARITE_ROI_PLANS_KEY] = roi_plans_contract(
        training_source="fragment_support_sidecar"
    )
    contract = continuity_plans_contract(data_identifier=data_identifier)
    existing = frozen.get(CONTINUITY_PLANS_KEY)
    if existing is not None and existing != contract:
        raise ValueError(
            f"Plans already contain a different {CONTINUITY_PLANS_KEY} contract; "
            "use a new plans file"
        )
    frozen[CONTINUITY_PLANS_KEY] = contract
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-identifier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.plans.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Plans JSON must contain an object")
    if args.output is None:
        output = source
        backup = source.with_name(f"{source.stem}.before_separator_continuity{source.suffix}")
        if not backup.exists():
            shutil.copy2(source, backup)
    else:
        output = args.output.expanduser().resolve()
        if output == source:
            raise ValueError("Omit --output for in-place updates with backup")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    data_identifier = args.data_identifier or f"{output.stem}_{args.configuration}"
    frozen = freeze_separator_continuity_plans(
        value,
        configuration=args.configuration,
        data_identifier=data_identifier,
    )
    _write_json_atomic(output, frozen)
    print(f"Wrote separator continuity plans to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
