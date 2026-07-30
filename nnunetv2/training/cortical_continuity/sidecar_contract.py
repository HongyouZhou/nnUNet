"""Pure-NumPy validation helpers for synchronized raw sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def assert_exact_grid(
    reference: np.ndarray,
    reference_properties: dict,
    candidate: np.ndarray,
    candidate_properties: dict,
    *,
    role: str,
    path: str | Path,
) -> None:
    """Fail unless a sidecar has the CT shape and reader-specific geometry."""

    path_text = str(path)
    if tuple(reference.shape[1:]) != tuple(candidate.shape[1:]):
        raise ValueError(
            f"{role} grid shape {candidate.shape[1:]} differs from CT "
            f"{reference.shape[1:]}: {path_text}"
        )
    _assert_geometry_value(
        reference_properties.get("spacing"),
        candidate_properties.get("spacing"),
        role,
        "spacing",
        path_text,
    )

    reference_nibabel = reference_properties.get("nibabel_stuff")
    candidate_nibabel = candidate_properties.get("nibabel_stuff")
    reference_sitk = reference_properties.get("sitk_stuff")
    candidate_sitk = candidate_properties.get("sitk_stuff")
    if isinstance(reference_nibabel, dict) and isinstance(candidate_nibabel, dict):
        for key in ("original_affine", "reoriented_affine"):
            _assert_geometry_value(
                reference_nibabel.get(key),
                candidate_nibabel.get(key),
                role,
                key,
                path_text,
            )
        return
    if isinstance(reference_sitk, dict) and isinstance(candidate_sitk, dict):
        for key in ("spacing", "origin", "direction"):
            _assert_geometry_value(
                reference_sitk.get(key),
                candidate_sitk.get(key),
                role,
                key,
                path_text,
            )
        return
    raise ValueError(
        f"Cannot prove exact CT/{role} grid equality: reader metadata types differ "
        f"or lack nibabel_stuff/sitk_stuff: {path_text}"
    )


def assert_instance_retention(
    source_instances: np.ndarray,
    resampled_instances: np.ndarray,
) -> dict[str, Any]:
    """Hard-fail when 0.5-mm preprocessing removes a cortical instance."""

    source = np.rint(np.asarray(source_instances)).astype(np.int64, copy=False)
    resampled = np.rint(np.asarray(resampled_instances)).astype(
        np.int64,
        copy=False,
    )
    source_ids = {int(value) for value in np.unique(source) if value > 0}
    resampled_ids = {
        int(value) for value in np.unique(resampled) if value > 0
    }
    missing = sorted(source_ids.difference(resampled_ids))
    unexpected = sorted(resampled_ids.difference(source_ids))
    if missing:
        raise ValueError(
            "0.5-mm preprocessing removed cortical instance IDs "
            f"{missing}; thin-cortex retention gate failed"
        )
    if unexpected:
        raise ValueError(
            "resampling introduced unexpected cortical instance IDs "
            f"{unexpected}"
        )
    return {
        "source_instance_count": len(source_ids),
        "resampled_instance_count": len(resampled_ids),
        "missing_instance_ids": missing,
        "status": "PASS",
    }


def _assert_geometry_value(
    reference: Any,
    candidate: Any,
    role: str,
    field: str,
    path: str,
) -> None:
    if reference is None or candidate is None:
        raise ValueError(f"Missing {field} metadata while checking {role}: {path}")
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape or not np.allclose(
        reference_array,
        candidate_array,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{role} {field} differs from CT grid: {path}")
