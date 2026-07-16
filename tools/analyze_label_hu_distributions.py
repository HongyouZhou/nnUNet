#!/usr/bin/env python3
"""Compare ABBC-label HU distributions across Dataset777 data sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import levene, mannwhitneyu, rankdata, wasserstein_distance


LABELS = (1, 2, 3)


def _seed(case: str, suffix: str) -> int:
    digest = hashlib.sha256(f"{case}:{suffix}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _sample(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= limit:
        return values.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    return values[rng.choice(values.size, limit, replace=False)].astype(np.float32, copy=False)


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    q = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
    }


def _foreground_crop(seg: np.ndarray, margin: np.ndarray) -> tuple[slice, ...]:
    foreground = seg > 0
    slices = []
    for axis in range(seg.ndim):
        other_axes = tuple(i for i in range(seg.ndim) if i != axis)
        occupied = np.flatnonzero(np.any(foreground, axis=other_axes))
        if occupied.size == 0:
            return tuple(slice(0, n) for n in seg.shape)
        start = max(0, int(occupied[0]) - int(margin[axis]))
        stop = min(seg.shape[axis], int(occupied[-1]) + int(margin[axis]) + 1)
        slices.append(slice(start, stop))
    return tuple(slices)


def _ellipsoid(spacing: np.ndarray, radius_mm: float) -> np.ndarray:
    radii = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    distance = np.zeros(tuple(2 * radii + 1), dtype=np.float32)
    for grid, voxel_spacing in zip(grids, spacing):
        distance += (grid * voxel_spacing / radius_mm) ** 2
    return distance <= 1.0


def _overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    combined = np.concatenate((a, b))
    lo, hi = np.quantile(combined, (0.005, 0.995))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    ha, edges = np.histogram(a, bins=128, range=(lo, hi))
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(1, ha.sum())
    pb = hb / max(1, hb.sum())
    return float(np.minimum(pa, pb).sum())


def _binary_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """AUC with average ranks, including correct handling of tied HU values."""
    scores = np.concatenate((positive_scores, negative_scores))
    ranks = rankdata(scores, method="average")
    n_positive = positive_scores.size
    n_negative = negative_scores.size
    rank_sum_positive = ranks[:n_positive].sum()
    u = rank_sum_positive - n_positive * (n_positive + 1) / 2
    return float(u / (n_positive * n_negative))


def _process_case(
    case: str,
    image_path: str,
    label_path: str,
    sample_limit: int,
    local_radius_mm: float,
) -> dict:
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    seg = np.asarray(label_nii.dataobj, dtype=np.uint8)
    if image.shape != seg.shape:
        raise ValueError(f"{case}: image shape {image.shape} != label shape {seg.shape}")

    spacing = np.asarray(image_nii.header.get_zooms()[:3], dtype=np.float32)
    source = case.split("_", 1)[0]
    samples = {}
    label_stats = {}
    for label in LABELS:
        values = image[seg == label]
        label_stats[str(label)] = _stats(values)
        samples[label] = _sample(values, sample_limit, _seed(case, f"label-{label}"))

    bone_reference = np.concatenate((samples[1], samples[2]))
    center = float(np.median(bone_reference))
    q25, q75 = np.quantile(bone_reference, (0.25, 0.75))
    scale = float(max(q75 - q25, 1.0))

    margin = np.ceil(local_radius_mm / spacing).astype(int) + 1
    crop = _foreground_crop(seg, margin)
    cropped_seg = seg[crop]
    cropped_image = image[crop]
    label3 = cropped_seg == 3
    local_control_mask = binary_dilation(
        label3,
        structure=_ellipsoid(spacing, local_radius_mm),
    ) & ((cropped_seg == 1) | (cropped_seg == 2))
    local_control = _sample(
        cropped_image[local_control_mask],
        sample_limit,
        _seed(case, "local-control"),
    )
    local_label3 = samples[3]

    local = {"control_n": int(local_control.size)}
    if local_control.size and local_label3.size:
        n = min(local_control.size, local_label3.size, sample_limit)
        rng = np.random.default_rng(_seed(case, "balanced-local"))
        a = local_label3 if local_label3.size == n else local_label3[rng.choice(local_label3.size, n, replace=False)]
        b = local_control if local_control.size == n else local_control[rng.choice(local_control.size, n, replace=False)]
        auc_low = _binary_auc(-a, -b)
        a_z = (a - center) / scale
        b_z = (b - center) / scale
        local.update({
            "balanced_n_per_class": int(n),
            "label3_median_hu": float(np.median(a)),
            "control_median_hu": float(np.median(b)),
            "median_delta_hu": float(np.median(a) - np.median(b)),
            "median_delta_bone_iqr": float(np.median(a_z) - np.median(b_z)),
            "auc_label3_is_lower_hu": auc_low,
            "auc_separability": max(auc_low, 1.0 - auc_low),
            "wasserstein_hu": float(wasserstein_distance(a, b)),
            "wasserstein_bone_iqr": float(wasserstein_distance(a_z, b_z)),
            "overlap_coefficient": _overlap_coefficient(a_z, b_z),
        })

    del image, seg
    return {
        "case": case,
        "source": source,
        "spacing": spacing.tolist(),
        "bone_reference_median_hu": center,
        "bone_reference_iqr_hu": scale,
        "labels": label_stats,
        "local_label3_vs_label12": local,
    }


def _describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _metric_values(cases: list[dict], path: tuple[str, ...]) -> list[float]:
    output = []
    for case in cases:
        value = case
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        if isinstance(value, (int, float)) and np.isfinite(value):
            output.append(float(value))
    return output


def _source_summary(cases: list[dict]) -> dict:
    metrics = {
        "label1_median_hu": ("labels", "1", "median"),
        "label2_median_hu": ("labels", "2", "median"),
        "label3_median_hu": ("labels", "3", "median"),
        "label3_voxels": ("labels", "3", "n"),
        "local_median_delta_hu": ("local_label3_vs_label12", "median_delta_hu"),
        "local_median_delta_bone_iqr": ("local_label3_vs_label12", "median_delta_bone_iqr"),
        "local_auc_label3_is_lower_hu": ("local_label3_vs_label12", "auc_label3_is_lower_hu"),
        "local_auc_separability": ("local_label3_vs_label12", "auc_separability"),
        "local_wasserstein_bone_iqr": ("local_label3_vs_label12", "wasserstein_bone_iqr"),
        "local_overlap_coefficient": ("local_label3_vs_label12", "overlap_coefficient"),
    }
    return {name: _describe(_metric_values(cases, path)) for name, path in metrics.items()}


def _source_tests(grouped: dict[str, list[dict]]) -> dict:
    if "charite" not in grouped or "pengwin" not in grouped:
        return {}
    paths = {
        "label3_median_hu": ("labels", "3", "median"),
        "local_median_delta_bone_iqr": ("local_label3_vs_label12", "median_delta_bone_iqr"),
        "local_auc_label3_is_lower_hu": ("local_label3_vs_label12", "auc_label3_is_lower_hu"),
        "local_auc_separability": ("local_label3_vs_label12", "auc_separability"),
        "local_overlap_coefficient": ("local_label3_vs_label12", "overlap_coefficient"),
    }
    tests = {}
    for name, path in paths.items():
        charite = _metric_values(grouped["charite"], path)
        pengwin = _metric_values(grouped["pengwin"], path)
        if not charite or not pengwin:
            continue
        u, p = mannwhitneyu(charite, pengwin, alternative="two-sided")
        lv, lp = levene(charite, pengwin, center="median")
        tests[name] = {
            "mann_whitney_u": float(u),
            "mann_whitney_p": float(p),
            "rank_biserial_charite_minus_pengwin": float(2 * u / (len(charite) * len(pengwin)) - 1),
            "levene_statistic": float(lv),
            "levene_p": float(lp),
        }
    return tests


def _write_csv(path: Path, cases: list[dict]) -> None:
    fields = [
        "case", "source", "spacing", "bone_reference_median_hu", "bone_reference_iqr_hu",
        "label1_n", "label1_median_hu", "label2_n", "label2_median_hu",
        "label3_n", "label3_median_hu", "local_control_n", "local_median_delta_hu",
        "local_median_delta_bone_iqr", "local_auc_label3_is_lower_hu",
        "local_auc_separability", "local_wasserstein_bone_iqr", "local_overlap_coefficient",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            local = case["local_label3_vs_label12"]
            row = {
                "case": case["case"],
                "source": case["source"],
                "spacing": "x".join(f"{x:.5g}" for x in case["spacing"]),
                "bone_reference_median_hu": case["bone_reference_median_hu"],
                "bone_reference_iqr_hu": case["bone_reference_iqr_hu"],
            }
            for label in LABELS:
                row[f"label{label}_n"] = case["labels"][str(label)].get("n")
                row[f"label{label}_median_hu"] = case["labels"][str(label)].get("median")
            for key in (
                "control_n", "median_delta_hu", "median_delta_bone_iqr",
                "auc_label3_is_lower_hu", "auc_separability",
                "wasserstein_bone_iqr", "overlap_coefficient",
            ):
                row[f"local_{key}"] = local.get(key)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=100_000)
    parser.add_argument("--local-radius-mm", type=float, default=3.0)
    args = parser.parse_args()

    image_dir = args.dataset / "imagesTr"
    label_dir = args.dataset / "labelsTr"
    jobs = []
    for image_path in sorted(image_dir.glob("*_0000.nii.gz")):
        case = image_path.name.removesuffix("_0000.nii.gz")
        label_path = label_dir / f"{case}.nii.gz"
        if label_path.is_file():
            jobs.append((case, str(image_path), str(label_path)))
    if not jobs:
        raise RuntimeError(f"No paired training cases found under {args.dataset}")

    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_case,
                case,
                image_path,
                label_path,
                args.sample_limit,
                args.local_radius_mm,
            ): case
            for case, image_path, label_path in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            result = future.result()
            cases.append(result)
            print(f"[{index}/{len(futures)}] {case}", flush=True)

    cases.sort(key=lambda item: item["case"])
    grouped: dict[str, list[dict]] = {}
    for case in cases:
        grouped.setdefault(case["source"], []).append(case)
    report = {
        "dataset": str(args.dataset),
        "local_radius_mm": args.local_radius_mm,
        "sample_limit_per_class_per_case": args.sample_limit,
        "num_cases": len(cases),
        "sources": {source: _source_summary(items) for source, items in grouped.items()},
        "source_tests": _source_tests(grouped),
        "cases": cases,
    }
    (args.output / "hu_distribution_report.json").write_text(json.dumps(report, indent=2))
    _write_csv(args.output / "hu_distribution_per_case.csv", cases)
    print(json.dumps({"num_cases": len(cases), "sources": report["sources"], "source_tests": report["source_tests"]}, indent=2))


if __name__ == "__main__":
    main()
