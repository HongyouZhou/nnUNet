"""Development-only diagnosis of cortical instance failures on frozen OOF outputs.

GT ownership is used to measure coverage and seed mixing, never to produce a
deployable prediction. Touching annotations are a candidate list for inspection,
not proof that a particular CT contains an anatomical bridge.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.segmentation import watershed

from tools.charite_cortical.continuity_workflow import (
    _load_probability_channel,
    _write_json_atomic,
    evaluate_instances,
    validation_case_ids,
)


TASKS = tuple(("matched_base", fold) for fold in range(5)) + (
    ("continuity", 0), ("continuity", 1),
    ("continuity_density", 0), ("continuity_density", 1),
)
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


def separator_markers(union, probability, threshold=0.5, minimum_seed_voxels=10):
    """Reproduce the frozen postprocessor's markers, including fallback seeds."""
    markers, _ = label(union & (probability < threshold), structure=CONNECTIVITY)
    counts = np.bincount(markers.ravel())
    keep = counts >= minimum_seed_voxels
    keep[0] = False
    markers[~keep[markers]] = 0
    markers, _ = label(markers > 0, structure=CONNECTIVITY)
    components, count = label(union, structure=CONNECTIVITY)
    next_id = int(markers.max()) + 1
    fallback_count = 0
    for component_id in range(1, count + 1):
        component = components == component_id
        if np.any(markers[component] > 0):
            continue
        indices = np.flatnonzero(component)
        seed = indices[int(np.argmin(probability.ravel()[indices]))]
        markers.ravel()[seed] = next_id
        next_id += 1
        fallback_count += 1
    return markers, components, fallback_count


def _overlap_table(gt, labels, gt_ids):
    table = {}
    for gt_id in gt_ids:
        values, counts = np.unique(labels[gt == gt_id], return_counts=True)
        table[int(gt_id)] = {
            int(value): int(count) for value, count in zip(values, counts) if value > 0
        }
    return table


def _mixed_regions(table, sizes, minimum_fraction=0.1):
    regions = {}
    for gt_id, overlaps in table.items():
        for region_id, overlap in overlaps.items():
            if overlap / sizes[gt_id] >= minimum_fraction:
                regions.setdefault(region_id, []).append(gt_id)
    return {region: owners for region, owners in regions.items() if len(owners) > 1}


def touching_pairs(gt):
    """Count annotation contacts using the same 26-neighbour convention."""
    contacts = {}
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        first = tuple(slice(max(0, -d), min(n, n - d)) for n, d in zip(gt.shape, offset))
        second = tuple(slice(max(0, d), min(n, n + d)) for n, d in zip(gt.shape, offset))
        a, b = gt[first], gt[second]
        different = (a > 0) & (b > 0) & (a != b)
        if not different.any():
            continue
        pairs = np.sort(np.column_stack((a[different], b[different])), axis=1)
        pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for pair, count in zip(pairs, counts):
            key = (int(pair[0]), int(pair[1]))
            contacts[key] = contacts.get(key, 0) + int(count)
    return contacts


def diagnose_arrays(semantic, probability, predicted, gt_instances, validity):
    semantic_valid = (validity & 1) != 0
    gt = np.where((validity & 2) != 0, gt_instances, 0)
    union = np.isin(semantic, (1, 2))
    if not np.array_equal(predicted > 0, union):
        raise ValueError("Saved postprocessing does not exactly cover the predicted cortical union")
    ids = [int(value) for value in np.unique(gt) if value > 0]
    sizes = {value: int(np.count_nonzero(gt == value)) for value in ids}
    markers, components, fallback_count = separator_markers(union, probability)
    seed_table = _overlap_table(gt, markers, ids)
    component_table = _overlap_table(gt, components, ids)
    instance_table = _overlap_table(gt, np.where(semantic_valid, predicted, 0), ids)
    mixed_seeds = _mixed_regions(seed_table, sizes)
    mixed_components = _mixed_regions(component_table, sizes)
    mixed_instances = _mixed_regions(instance_table, sizes)
    seed_blocked = {value for owners in mixed_seeds.values() for value in owners}
    pred_sizes = np.bincount(predicted[semantic_valid].ravel())
    children = []
    for gt_id in ids:
        overlaps = instance_table[gt_id]
        best_iou = max(
            (count / (sizes[gt_id] + int(pred_sizes[pred_id]) - count)
             for pred_id, count in overlaps.items()), default=0.0,
        )
        coverage = sum(component_table[gt_id].values()) / sizes[gt_id]
        seed_coverage = sum(seed_table[gt_id].values()) / sizes[gt_id]
        children.append({
            "gt_id": gt_id, "gt_voxels": sizes[gt_id],
            "predicted_union_coverage": coverage,
            "coverage_below_recovery_threshold": coverage < 0.5,
            "best_individual_iou": best_iou,
            "seed_coverage": seed_coverage,
            "substantial_seed_count": sum(count / sizes[gt_id] >= 0.1 for count in seed_table[gt_id].values()),
            "in_mixed_seed": gt_id in seed_blocked,
            "seed_overlaps": seed_table[gt_id],
        })
    contacts = touching_pairs(gt)
    pairs = []
    for pair, count in sorted(contacts.items(), key=lambda item: -item[1]):
        shared_markers = [region for region, owners in mixed_seeds.items() if all(value in owners for value in pair)]
        pairs.append({"gt_ids": list(pair), "contact_edges_26": count, "shared_substantial_markers": shared_markers})
    metrics = evaluate_instances(predicted, union, gt_instances, validity)
    return {
        "metrics": metrics, "children": children,
        "marker_count": int(markers.max()), "union_component_count": int(components.max()),
        "fallback_seed_count": fallback_count,
        "mixed_marker_count": len(mixed_seeds), "mixed_markers": mixed_seeds,
        "mixed_union_components": mixed_components, "mixed_output_instances": mixed_instances,
        "touching_gt_pairs": pairs,
        "coverage_upper_bound_recovery_fraction": float(np.mean([not c["coverage_below_recovery_threshold"] for c in children])),
        "coverage_limited_child_count": sum(c["coverage_below_recovery_threshold"] for c in children),
        "mixed_seed_child_count": len(seed_blocked),
        "adequately_covered_mixed_seed_child_count": sum(c["in_mixed_seed"] and not c["coverage_below_recovery_threshold"] for c in children),
        "touching_mixed_seed_pair_count": sum(bool(pair["shared_substantial_markers"]) for pair in pairs),
    }


def _load_aligned(path, reference, dtype):
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
        raise ValueError(f"Image does not share the native prediction grid: {path}")
    return np.rint(np.asanyarray(image.dataobj)).astype(dtype)


def load_case(dataset, run_dir, arm, fold, case):
    artifact = run_dir / "artifacts" / arm / f"fold_{fold}"
    reference = nib.load(str(artifact / "predictions" / f"{case}.nii.gz"))
    arrays = {
        "semantic": np.rint(np.asanyarray(reference.dataobj)).astype(np.int16),
        "probability": _load_probability_channel(artifact / "predictions" / f"{case}.npz", reference, 2),
        "predicted": _load_aligned(artifact / "instances" / f"{case}_instances.nii.gz", reference, np.int32),
        "gt_instances": _load_aligned(dataset / "corticalInstancesTr" / f"{case}.nii.gz", reference, np.int32),
        "validity": _load_aligned(dataset / "validMasksTr" / f"{case}.nii.gz", reference, np.int16),
    }
    occupied = (arrays["semantic"] > 0) | (arrays["gt_instances"] > 0)
    bounds = []
    for axis in range(3):
        extent = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append(slice(max(0, int(extent[0]) - 1), min(occupied.shape[axis], int(extent[-1]) + 2)))
    crop = tuple(bounds)
    arrays = {name: value[crop].copy() for name, value in arrays.items()}
    return reference, crop, arrays


def oracle_seed_partition(semantic, probability, gt_instances, validity, seed_fraction=0.1):
    """GT-assisted counterfactual; tests separate markers, not a new algorithm."""
    if not 0 < seed_fraction <= 1:
        raise ValueError("Seed fraction must be in (0,1]")
    union = np.isin(semantic, (1, 2))
    gt = np.where((validity & 2) != 0, gt_instances, 0)
    markers = np.zeros(gt.shape, dtype=np.int32)
    for gt_id in np.unique(gt):
        if gt_id == 0:
            continue
        indices = np.flatnonzero((gt == gt_id) & union)
        if not len(indices):
            continue
        count = max(1, int(np.ceil(len(indices) * seed_fraction)))
        selected = np.argpartition(probability.ravel()[indices], count - 1)[:count]
        markers.ravel()[indices[selected]] = gt_id
    components, count = label(union, structure=CONNECTIVITY)
    next_id = int(gt.max()) + 1
    for component_id in range(1, count + 1):
        component = components == component_id
        if np.any(markers[component] > 0):
            continue
        indices = np.flatnonzero(component)
        seed = indices[int(np.argmin(probability.ravel()[indices]))]
        markers.ravel()[seed] = next_id
        next_id += 1
    result = watershed(probability, markers=markers, mask=union, connectivity=CONNECTIVITY, watershed_line=False)
    return markers, np.asarray(result, dtype=np.int32)


def inspect_case(dataset, run_dir, output, arm, fold, case):
    """Export three-plane CT evidence plus a labelled oracle-seed ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    reference, crop, arrays = load_case(dataset, run_dir, arm, fold, case)
    diagnosis = json.loads((output / arm / f"fold_{fold}" / f"{case}.json").read_text())
    markers, _, _ = separator_markers(np.isin(arrays["semantic"], (1, 2)), arrays["probability"])
    oracle_markers, oracle = oracle_seed_partition(arrays["semantic"], arrays["probability"], arrays["gt_instances"], arrays["validity"])
    oracle_metrics = evaluate_instances(oracle, np.isin(arrays["semantic"], (1, 2)), arrays["gt_instances"], arrays["validity"])
    candidate = next((pair for pair in diagnosis["touching_gt_pairs"] if pair["shared_substantial_markers"]), None)
    if candidate is None:
        raise ValueError(f"No touching mixed-seed pair for {case}; choose a different diagnostic case")
    first, second = candidate["gt_ids"]
    interface = binary_dilation(arrays["gt_instances"] == first, structure=CONNECTIVITY) & (arrays["gt_instances"] == second)
    interface &= np.isin(markers, candidate["shared_substantial_markers"])
    points = np.argwhere(interface)
    middle = np.median(points, axis=0)
    center = points[np.argmin(np.sum((points - middle) ** 2, axis=1))]
    image = nib.load(str(dataset / "imagesTr" / f"{case}_0000.nii.gz"))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
        raise ValueError("CT does not share the native prediction grid")
    spacing = np.asarray(reference.header.get_zooms()[:3])
    radius = np.ceil(25 / spacing).astype(int)
    local = tuple(slice(max(0, int(c - r)), min(n, int(c + r + 1))) for c, r, n in zip(center, radius, arrays["semantic"].shape))
    native = tuple(slice(s.start + offset.start, s.stop + offset.start) for s, offset in zip(local, crop))
    ct = np.asarray(image.dataobj[native], dtype=np.float32)
    fields = {name: value[local] for name, value in arrays.items()}
    fields.update(markers=markers[local], oracle=oracle[local], oracle_markers=oracle_markers[local])
    local_center = center - np.array([s.start for s in local])
    figure, axes = plt.subplots(4, 3, figsize=(12, 14))
    row_names = ("GT ownership", "Frozen separator probability", "Frozen markers + output", "GT-assisted markers + output")
    orientations = nib.aff2axcodes(reference.affine)
    for axis in range(3):
        plane = lambda value: np.take(value, int(local_center[axis]), axis=axis).T
        other = [a for a in range(3) if a != axis]
        extent = (0, ct.shape[other[0]] * spacing[other[0]], 0, ct.shape[other[1]] * spacing[other[1]])
        for row in range(4):
            panel = axes[row, axis]
            panel.imshow(plane(ct), cmap="gray", vmin=-200, vmax=1500, origin="lower", extent=extent)
            if row == 1:
                values = np.ma.masked_where(~np.isin(plane(fields["semantic"]), (1, 2)), plane(fields["probability"]))
                panel.imshow(values, cmap="magma", vmin=0, vmax=1, alpha=0.7, origin="lower", extent=extent)
            if row in (2, 3):
                name = "predicted" if row == 2 else "oracle"
                values = np.ma.masked_where(plane(fields[name]) == 0, plane(fields[name]) % 20)
                panel.imshow(values, cmap="tab20", vmin=0, vmax=20, alpha=0.65, origin="lower", extent=extent)
                name = "markers" if row == 2 else "oracle_markers"
                seed_plane = plane(fields[name])
                if (seed_plane > 0).any() and (seed_plane == 0).any():
                    panel.contour(seed_plane > 0, levels=[0.5], colors="yellow", linewidths=0.6, origin="lower", extent=extent)
            for gt_id, color in ((first, "cyan"), (second, "lime")):
                mask = plane(fields["gt_instances"]) == gt_id
                if mask.any() and (~mask).any():
                    panel.contour(mask, levels=[0.5], colors=color, linewidths=0.9, origin="lower", extent=extent)
            panel.set_title(f"{row_names[row]} | native axis {axis} ({orientations[axis]})")
            panel.set_xlabel("mm")
            panel.set_ylabel("mm")
    figure.suptitle(f"{case} / {arm}: GT {first}=cyan, {second}=green; seed outline=yellow\nGT-assisted experiment is diagnostic only; frozen recovery {diagnosis['metrics']['child_recovery_fraction']:.1%} -> oracle seeds {oracle_metrics['child_recovery_fraction']:.1%}")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    destination = output / "inspections" / arm / case
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / "ct_seed_diagnosis.png", dpi=160)
    plt.close(figure)
    record = {
        "case_id": case, "arm": arm, "fold": fold, "development_only": True,
        "counterfactual": "GT ownership supplies separate markers using the lowest-probability 10% of each covered GT instance; cortical union and separator field stay frozen.",
        "frozen_metrics": diagnosis["metrics"], "oracle_seed_metrics": oracle_metrics,
        "selected_pair": candidate, "native_voxel": (center + np.array([s.start for s in crop])).tolist(),
        "world_mm": nib.affines.apply_affine(reference.affine, center + np.array([s.start for s in crop])).tolist(),
        "ct_crop_start": [s.start for s in native], "ct_crop_shape": list(ct.shape),
    }
    transform = np.eye(4)
    transform[:3, 3] = [s.start for s in native]
    affine = reference.affine @ transform
    for name in ("gt_instances", "predicted", "markers", "oracle", "oracle_markers"):
        nib.save(nib.Nifti1Image(fields[name].astype(np.int32), affine), destination / f"{name}_roi.nii.gz")
    nib.save(nib.Nifti1Image(ct, affine), destination / "ct_roi.nii.gz")
    _write_json_atomic(destination / "inspection.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def diagnose_fold(dataset, run_dir, output, arm, fold):
    cases = validation_case_ids(dataset, fold)
    source = json.loads((run_dir / "artifacts" / arm / f"fold_{fold}" / "metrics.json").read_text())
    expected = {record["case_id"]: record for record in source["records"]}
    for index, case in enumerate(cases):
        _, crop, arrays = load_case(dataset, run_dir, arm, fold, case)
        record = diagnose_arrays(**arrays)
        for key, value in record["metrics"].items():
            if not np.isclose(value, expected[case][key], rtol=0, atol=1e-10):
                raise ValueError(f"Diagnostic metrics changed: {case}/{key}")
        record.update(case_id=case, arm=arm, fold=fold, crop_start=[s.start for s in crop])
        _write_json_atomic(output / arm / f"fold_{fold}" / f"{case}.json", record)
        print(f"[DIAG] {arm}/{fold} {index+1}/{len(cases)} {case} mixed={record['mixed_marker_count']} coverage_limited={record['coverage_limited_child_count']}", flush=True)
    _write_json_atomic(output / arm / f"fold_{fold}" / "complete.json", {"cases": cases})


def summarize(output):
    summaries = {}
    representatives = []

    def aggregate(records):
        children = [child for record in records for child in record["children"]]
        return {
            "case_count": len(records), "gt_child_count": len(children),
            "cases_with_mixed_seeds": sum(record["mixed_marker_count"] > 0 for record in records),
            "cases_with_touching_mixed_seeds": sum(record["touching_mixed_seed_pair_count"] > 0 for record in records),
            "coverage_limited_children": sum(record["coverage_limited_child_count"] for record in records),
            "children_in_mixed_seeds": sum(record["mixed_seed_child_count"] for record in records),
            "adequately_covered_children_in_mixed_seeds": sum(record["adequately_covered_mixed_seed_child_count"] for record in records),
            "actual_recovery_patient_macro": float(np.mean([record["metrics"]["child_recovery_fraction"] for record in records])),
            "coverage_upper_bound_patient_macro": float(np.mean([record["coverage_upper_bound_recovery_fraction"] for record in records])),
        }

    for arm in ("matched_base", "continuity", "continuity_density"):
        records = []
        for task_arm, fold in TASKS:
            if task_arm != arm:
                continue
            path = output / arm / f"fold_{fold}"
            complete = json.loads((path / "complete.json").read_text())
            records.extend(json.loads((path / f"{case}.json").read_text()) for case in complete["cases"])
        summaries[arm] = aggregate(records)
        if arm == "matched_base":
            summaries["matched_base_pilot"] = aggregate([record for record in records if record["fold"] in (0, 1)])
            ranked = sorted(records, key=lambda r: (r["touching_mixed_seed_pair_count"] > 0, r["adequately_covered_mixed_seed_child_count"]), reverse=True)
            representatives = [{"case_id": r["case_id"], "fold": r["fold"], "mixed_seed_child_count": r["mixed_seed_child_count"], "touching_mixed_seed_pair_count": r["touching_mixed_seed_pair_count"]} for r in ranked[:6]]
    result = {
        "kind": "cortical_continuity_failure_diagnosis", "schema_version": 1,
        "contract": {
            "development_only": True, "gt_used_for_diagnosis": True,
            "substantial_overlap_fraction_of_gt": 0.1, "recovery_iou_threshold": 0.5,
            "coverage_upper_bound": "Optimistic assignment of correctly covered GT voxels; excludes all prediction false positives. Not a deployable result.",
            "touching_annotations": "26-neighbour contacts, requiring visual CT inspection before anatomical interpretation.",
            "failure_flags_overlap": True,
        },
        "arms": summaries, "representatives": representatives,
    }
    _write_json_atomic(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--inspect-case")
    parser.add_argument("--arm", default="matched_base")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.output), indent=2))
    elif args.inspect_case:
        inspect_case(args.dataset, args.run_dir, args.output, args.arm, args.fold, args.inspect_case)
    else:
        arm, fold = TASKS[args.task_id]
        diagnose_fold(args.dataset, args.run_dir, args.output, arm, fold)


if __name__ == "__main__":
    main()
