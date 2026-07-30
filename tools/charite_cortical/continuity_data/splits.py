"""Frozen cohort-stratified patient-level five-fold split."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import N_FOLDS, SPLIT_RANDOM_STATE


@dataclass(frozen=True)
class SplitResult:
    fold_by_case: Mapping[str, int]
    splits_final: tuple[Mapping[str, list[str]], ...]
    audit: Mapping[str, Any]


def fragment_burden_bin(count: int) -> str:
    if count <= 4:
        return "low_1_4"
    if count <= 9:
        return "medium_5_9"
    return "high_10_plus"


def make_stratified_fivefold(
    cases: Sequence[Mapping[str, Any]],
    *,
    random_state: int = SPLIT_RANDOM_STATE,
    n_folds: int = N_FOLDS,
) -> SplitResult:
    """Match ``sklearn.model_selection.StratifiedKFold`` exactly.

    Cases are first ordered by ``case_id`` so the result is independent of CSV
    row order.  The stratification label is the source cohort, as frozen for
    Dataset778.  The small compatibility implementation below follows
    scikit-learn's allocation algorithm and lets the data audit run in a
    dependency-light environment as well.
    """

    if random_state != SPLIT_RANDOM_STATE:
        raise ValueError(f"random_state is frozen at {SPLIT_RANDOM_STATE}")
    if n_folds != N_FOLDS:
        raise ValueError(f"Dataset778 uses exactly {N_FOLDS} folds")
    if len(cases) < n_folds:
        raise ValueError("number of cases is smaller than number of folds")

    ordered_cases = sorted(cases, key=lambda item: str(item["case_id"]))
    case_ids: set[str] = set()
    nnunet_ids: set[str] = set()
    group_ids: set[str] = set()
    for case in ordered_cases:
        case_id = str(case["case_id"])
        nnunet_id = str(case["nnunet_id"])
        if case_id in case_ids or nnunet_id in nnunet_ids:
            raise ValueError("case_id and nnunet_id must be unique")
        group_id = str(case["group_id"])
        if group_id in group_ids:
            raise ValueError(
                "frozen Dataset778 StratifiedKFold requires one case per patient/group"
            )
        case_ids.add(case_id)
        nnunet_ids.add(nnunet_id)
        group_ids.add(group_id)

    cohorts = np.asarray([str(case["cohort"]) for case in ordered_cases], dtype=object)
    classes, first_indices, inverse = np.unique(
        cohorts, return_index=True, return_inverse=True
    )
    # sklearn encodes classes by order of first appearance, not lexical order.
    _, class_permutation = np.unique(first_indices, return_inverse=True)
    encoded = class_permutation[inverse]
    class_counts = np.bincount(encoded)
    if np.min(class_counts) < n_folds:
        counts = dict(zip(classes.tolist(), class_counts.tolist()))
        raise ValueError(
            f"every cohort needs at least {n_folds} cases for stratification: {counts}"
        )
    encoded_order = np.sort(encoded)
    allocation = np.asarray(
        [
            np.bincount(encoded_order[fold::n_folds], minlength=len(classes))
            for fold in range(n_folds)
        ]
    )
    rng = np.random.RandomState(random_state)
    test_folds = np.empty(len(ordered_cases), dtype=np.int64)
    for class_index in range(len(classes)):
        folds_for_class = np.arange(n_folds).repeat(allocation[:, class_index])
        rng.shuffle(folds_for_class)
        test_folds[encoded == class_index] = folds_for_class

    fold_by_case = {
        str(case["case_id"]): int(fold)
        for case, fold in zip(ordered_cases, test_folds, strict=True)
    }
    all_nnunet_ids = sorted(str(case["nnunet_id"]) for case in ordered_cases)
    split_records: list[Mapping[str, list[str]]] = []
    for fold in range(n_folds):
        validation = sorted(
            str(case["nnunet_id"])
            for case, selected_fold in zip(ordered_cases, test_folds, strict=True)
            if int(selected_fold) == fold
        )
        validation_set = set(validation)
        split_records.append(
            {
                "train": [
                    identifier
                    for identifier in all_nnunet_ids
                    if identifier not in validation_set
                ],
                "val": validation,
            }
        )

    if set(fold_by_case) != case_ids:
        raise RuntimeError("split assignment omitted one or more cases")
    audit_folds: list[dict[str, Any]] = []
    case_by_id = {str(case["case_id"]): case for case in ordered_cases}
    for fold in range(n_folds):
        selected_cases = [
            case_by_id[case_id]
            for case_id, selected_fold in fold_by_case.items()
            if selected_fold == fold
        ]
        audit_folds.append(
            {
                "fold": fold,
                "cases": len(selected_cases),
                "groups": len(
                    {str(case["group_id"]) for case in selected_cases}
                ),
                "cohort_counts": dict(
                    sorted(Counter(str(case["cohort"]) for case in selected_cases).items())
                ),
                "fragment_burden_counts": dict(
                    sorted(
                        Counter(
                            fragment_burden_bin(int(case["fragment_count"]))
                            for case in selected_cases
                        ).items()
                    )
                ),
                "thick_z_cases": sum(
                    float(case["spacing_xyz_mm"][2]) >= 1.0
                    for case in selected_cases
                ),
                "fragment_without_cortical_cases": sum(
                    int(case.get("fragment_without_cortical_count", 0)) > 0
                    for case in selected_cases
                ),
            }
        )
    return SplitResult(
        fold_by_case=fold_by_case,
        splits_final=tuple(split_records),
        audit={
            "strategy": "sklearn.model_selection.StratifiedKFold",
            "stratify_by": "cohort",
            "shuffle": True,
            "case_order": "case_id_lexical",
            "random_state": random_state,
            "n_folds": n_folds,
            "folds": audit_folds,
        },
    )
