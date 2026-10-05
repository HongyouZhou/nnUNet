"""Selected-case development validation of automatic cortical ownership.

Prediction is finished before any GT files or diagnostic selections are read.
Evaluation and inspection may use annotations; neither can change prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from tools.charite_cortical.continuity_workflow import _load_probability_channel, _write_json_atomic, evaluate_instances
from tools.charite_cortical.geometry_ownership import GeometryConfig, geometry_ownership


CASES = (("charite_1434", 0), ("charite_18", 1), ("charite_103", 1))


def _check_grid(image, reference, name):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
        raise ValueError(f"{name} does not share the native prediction grid")
    basis = reference.affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    normalized = basis / spacing
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-5):
        raise ValueError("Sheared image grids need resampling before physical geometry")


def prediction_crop(semantic, spacing, config):
    """Crop uses only prediction support, with a physical CT context margin."""
    occupied = np.isin(semantic, (1, 2))
    padding = np.ceil(4 * (config.derivative_mm + config.context_tensor_mm) / spacing).astype(int) + 2
    crop = []
    for axis in range(3):
        extent = np.flatnonzero(occupied.any(axis=tuple(a for a in range(3) if a != axis)))
        if not len(extent):
            return tuple(slice(0, n) for n in semantic.shape)
        crop.append(slice(max(0, int(extent[0] - padding[axis])), min(semantic.shape[axis], int(extent[-1] + padding[axis] + 1))))
    return tuple(crop)


def sparse_evaluation(predicted, union, gt, validity):
    """Exact frozen metrics, omitting only background irrelevant to all counts."""
    support = ((gt > 0) & ((validity & 2) != 0)) | ((predicted > 0) & ((validity & 1) != 0))
    return evaluate_instances(predicted[support], union[support], gt[support], validity[support])


def audit_graph(graph_path, gt, validity):
    """Evaluate geometric constraints after inference; never supplies labels.

    Accept only patches with >=80% of relation-valid owned voxels assigned to
    one GT child, and >=50% of all patch voxels carrying valid ownership. Report
    excluded edges to avoid presenting this conditional audit as full coverage.
    """
    with np.load(graph_path) as graph:
        positions, patches = graph["positions"], graph["voxel_patch"]
        edges, repulsive, partition = graph["edges"], graph["repulsive"], graph["partition"]
        owners = gt.ravel()[positions]
        valid = ((validity.ravel()[positions] & 2) != 0) & (owners > 0)
        count = len(partition)
        sizes = np.bincount(patches, minlength=count)
        owned = np.bincount(patches[valid], minlength=count)
        combinations, counts = np.unique(np.column_stack((patches[valid], owners[valid])), axis=0, return_counts=True)
        dominant = np.zeros(count, np.int32)
        largest = np.zeros(count, np.int64)
        for (patch, owner), overlap in zip(combinations, counts):
            if overlap > largest[patch]:
                dominant[patch] = owner; largest[patch] = overlap
        purity = largest / np.maximum(owned, 1)
        retained = (purity >= 0.8) & (owned / sizes >= 0.5)
        a, b = edges.T
        usable = retained[a] & retained[b]
        different = dominant[a] != dominant[b]
        same_repulsive = usable & ~different & repulsive
        different_repulsive = usable & different & repulsive
        resolved = partition[a] != partition[b]
        child_ids, child_counts = np.unique(dominant[a[same_repulsive]], return_counts=True)
        angle = graph["fine_angles_deg"]
        excess = graph["excess_deg"]
        def quantiles(mask, values):
            return np.quantile(values[mask], [0.5, 0.9, 0.99]).tolist() if mask.any() else []
        denominator = int(np.count_nonzero(repulsive & usable))
        return dict(
            evaluation_only=True, minimum_owned_patch_fraction=0.5, minimum_owned_label_purity=0.8,
            total_edges=len(edges), audited_edges=int(usable.sum()), excluded_edges=int((~usable).sum()),
            total_repulsive_edges=int(repulsive.sum()), audited_repulsive_edges=denominator,
            same_gt_repulsive_edges=int(same_repulsive.sum()), different_gt_repulsive_edges=int(different_repulsive.sum()),
            different_gt_fraction_of_audited_repulsion=float(different_repulsive.sum() / denominator) if denominator else None,
            same_gt_edges=int((usable & ~different).sum()), different_gt_edges=int((usable & different).sum()),
            same_gt_edges_cut=int((usable & ~different & resolved).sum()), different_gt_edges_cut=int((usable & different & resolved).sum()),
            same_gt_repulsion_by_child={str(int(i)): int(n) for i, n in zip(child_ids, child_counts)},
            same_gt_angle_quantiles=quantiles(usable & ~different, angle),
            different_gt_angle_quantiles=quantiles(usable & different, angle),
            same_gt_excess_quantiles=quantiles(usable & ~different, excess),
            different_gt_excess_quantiles=quantiles(usable & different, excess),
            interpretation="Contact edges are correlated; this conditional development diagnostic is not an independent ownership-classifier score.",
        )


def ownership_evaluation(predicted, gt, validity, pairs, spacing, config=GeometryConfig()):
    """Pair separation plus same-GT regions far from other annotated pieces."""
    gt = np.where((validity & 2) != 0, gt, 0)
    pred = np.where((validity & 1) != 0, predicted, 0)
    children = []
    for gt_id in np.unique(gt[gt > 0]):
        values, counts = np.unique(pred[gt == gt_id], return_counts=True)
        overlaps = {int(v): int(c) for v, c in zip(values, counts) if v > 0}
        size = int(np.count_nonzero(gt == gt_id))
        dominant = max(overlaps, key=overlaps.get) if overlaps else 0
        children.append(dict(gt_id=int(gt_id), gt_voxels=size, dominant_prediction=dominant,
                             dominant_gt_fraction=overlaps.get(dominant, 0) / size,
                             substantial_predictions=[p for p, n in overlaps.items() if n / size >= 0.1]))
    lookup = {c["gt_id"]: c for c in children}
    pair_records = []
    for pair in pairs:
        if not pair["shared_substantial_markers"]:
            continue
        a, b = (lookup[value] for value in pair["gt_ids"])
        pair_records.append(dict(
            gt_ids=pair["gt_ids"], contact_edges_26=pair["contact_edges_26"],
            dominant_predictions=[a["dominant_prediction"], b["dominant_prediction"]],
            dominant_gt_fractions=[a["dominant_gt_fraction"], b["dominant_gt_fraction"]],
            separated=bool(a["dominant_prediction"] != b["dominant_prediction"] and
                           min(a["dominant_gt_fraction"], b["dominant_gt_fraction"]) >= 0.5),
        ))
    controls = []
    for child in sorted(children, key=lambda c: -c["gt_voxels"])[:3]:
        points = np.argwhere((gt == child["gt_id"]) & (pred > 0))
        others = np.argwhere((gt > 0) & (gt != child["gt_id"]))
        if len(points) < 100 or not len(others):
            continue
        radius = 8.0
        # A far-away point often lies at the scan end. Exclude that confound:
        # the entire control sphere and CT tensor context must fit in the scan.
        scan_margin = radius + 4 * (config.derivative_mm + config.context_tensor_mm)
        boundary_distance = np.minimum(points * spacing, (np.asarray(gt.shape) - 1 - points) * spacing)
        interior = points[np.all(boundary_distance >= scan_margin, axis=1)]
        if not len(interior):
            continue
        candidates = interior[np.linspace(0, len(interior) - 1, min(1000, len(interior))).astype(int)]
        distance, _ = cKDTree(others * spacing).query(candidates * spacing)
        index = int(np.argmax(distance))
        if distance[index] < 10:
            continue
        center = candidates[index]
        local = np.linalg.norm((points - center) * spacing, axis=1) <= radius
        labels, counts = np.unique(pred[tuple(points[local].T)], return_counts=True)
        substantial = labels[counts / counts.sum() >= 0.1]
        controls.append(dict(gt_id=child["gt_id"], native_center=center.tolist(), radius_mm=radius,
                             nearest_other_instance_mm=float(distance[index]), covered_voxels=int(local.sum()),
                             minimum_scan_boundary_distance_mm=float(np.min(boundary_distance[np.all(points == center, axis=1)])),
                             substantial_predictions=substantial.tolist(), false_split=bool(len(substantial) > 1),
                             interpretation="same-annotation region; proxy control, not clinically verified intact cortex"))
    return dict(children=children, touching_mixed_marker_pairs=pair_records, same_instance_controls=controls)


def inspection_figure(output, reference, crop, ct, base, prediction, gt, inspection, controls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    origin = np.array([s.start for s in crop])
    centers = [("selected contact", np.array(inspection["native_voxel"]), inspection["selected_pair"]["gt_ids"])]
    centers += [(f"same-instance control GT {c['gt_id']}", np.array(c["native_center"]), [c["gt_id"]]) for c in controls[:1]]
    for name, native_center, ids in centers:
        center = native_center - origin
        if np.any(center < 0) or np.any(center >= ct.shape):
            continue
        radius = np.ceil(22 / spacing).astype(int)
        region = tuple(slice(max(0, int(c - r)), min(n, int(c + r + 1))) for c, r, n in zip(center, radius, ct.shape))
        local_center = center - np.array([s.start for s in region])
        fields = [base[region], prediction[region], gt[region]]
        figure, axes = plt.subplots(3, 3, figsize=(12, 11))
        for axis in range(3):
            plane = lambda arr: np.take(arr, int(local_center[axis]), axis=axis).T
            others = [a for a in range(3) if a != axis]
            shape = ct[region].shape
            extent = (0, shape[others[0]] * spacing[others[0]], 0, shape[others[1]] * spacing[others[1]])
            for row, values in enumerate(fields):
                panel = axes[row, axis]
                panel.imshow(plane(ct[region]), cmap="gray", vmin=-200, vmax=1500, origin="lower", extent=extent)
                overlay = np.ma.masked_where(plane(values) == 0, plane(values) % 20)
                panel.imshow(overlay, cmap="tab20", vmin=0, vmax=20, alpha=0.65, origin="lower", extent=extent)
                for index, gt_id in enumerate(ids):
                    mask = plane(gt[region]) == gt_id
                    if mask.any() and (~mask).any():
                        panel.contour(mask, levels=[0.5], colors=["cyan", "lime"][index % 2], linewidths=0.8, origin="lower", extent=extent)
                panel.set_title(f"{('Frozen instances', 'Automatic CT geometry', 'GT evaluation only')[row]} / native axis {axis}")
                panel.set_xlabel("mm"); panel.set_ylabel("mm")
        figure.suptitle(f"{inspection['case_id']}: {name}; exploratory automatic ownership")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(output / ("contact.png" if name == "selected contact" else "same-instance-control.png"), dpi=150)
        plt.close(figure)


def run_case(dataset, run_dir, diagnosis, output, case, fold, config=GeometryConfig()):
    destination = output / case
    destination.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "artifacts" / "matched_base" / f"fold_{fold}"
    reference = nib.load(str(artifact / "predictions" / f"{case}.nii.gz"))
    semantic = np.rint(np.asanyarray(reference.dataobj)).astype(np.int16)
    spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    ct_image = nib.load(str(dataset / "imagesTr" / f"{case}_0000.nii.gz"))
    _check_grid(ct_image, reference, "CT")
    crop = prediction_crop(semantic, spacing, config)
    ct = np.asarray(ct_image.dataobj[crop], np.float32)
    probability = _load_probability_channel(artifact / "predictions" / f"{case}.npz", reference, 2)[crop].copy()
    prediction, baseline, geometry = geometry_ownership(semantic[crop], probability, ct, spacing, config,
                                                       graph_output=destination / "prediction-graph.npz")
    transform = np.eye(4); transform[:3, 3] = [s.start for s in crop]
    nib.save(nib.Nifti1Image(prediction, reference.affine @ transform), destination / "geometry_instances_crop.nii.gz")
    _write_json_atomic(destination / "prediction-contract.json", dict(
        case_id=case, fold=fold, crop_start=[s.start for s in crop], crop_shape=list(prediction.shape),
        prediction_inputs=[str(artifact / "predictions" / f"{case}.nii.gz"), str(artifact / "predictions" / f"{case}.npz"), str(dataset / "imagesTr" / f"{case}_0000.nii.gz")],
        annotations_read_before_prediction=False, geometry=geometry,
    ))
    print(f"[GEOMETRY] {case}: prediction saved; loading GT for evaluation now", flush=True)
    # Only after inference and its saved output: load evaluation annotations.
    gt_image = nib.load(str(dataset / "corticalInstancesTr" / f"{case}.nii.gz"))
    valid_image = nib.load(str(dataset / "validMasksTr" / f"{case}.nii.gz"))
    _check_grid(gt_image, reference, "GT"); _check_grid(valid_image, reference, "validity")
    gt = np.rint(np.asanyarray(gt_image.dataobj)).astype(np.int32)
    validity = np.rint(np.asanyarray(valid_image.dataobj)).astype(np.int16)
    full_prediction = np.zeros(semantic.shape, np.int32); full_prediction[crop] = prediction
    full_base = np.zeros(semantic.shape, np.int32); full_base[crop] = baseline
    baseline_metrics = sparse_evaluation(full_base, semantic > 0, gt, validity)
    metrics = sparse_evaluation(full_prediction, semantic > 0, gt, validity)
    diagnostic = json.loads((diagnosis / "matched_base" / f"fold_{fold}" / f"{case}.json").read_text())
    for key, value in baseline_metrics.items():
        if not np.isclose(value, diagnostic["metrics"][key], atol=1e-10, rtol=0):
            raise AssertionError(f"Frozen baseline changed: {case}/{key}")
    if not np.isclose(metrics["cortical_union_dice"], baseline_metrics["cortical_union_dice"], atol=1e-10, rtol=0):
        raise AssertionError("Geometry changed union Dice")
    ownership = ownership_evaluation(full_prediction, gt, validity, diagnostic["touching_gt_pairs"], spacing, config)
    baseline_ownership = ownership_evaluation(full_base, gt, validity, diagnostic["touching_gt_pairs"], spacing, config)
    inspection = json.loads((diagnosis / "inspections" / "matched_base" / case / "inspection.json").read_text())
    inspection_figure(destination, reference, crop, ct, baseline, prediction, gt[crop], inspection, ownership["same_instance_controls"])
    record = dict(case_id=case, fold=fold, development_only=True, geometry=geometry,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  baseline_metrics=baseline_metrics, geometry_metrics=metrics, baseline_ownership=baseline_ownership,
                  geometry_ownership=ownership, selected_pair=inspection["selected_pair"]["gt_ids"],
                  inference_annotation_access=False,
                  graph_audit=audit_graph(destination / "prediction-graph.npz", gt[crop], validity[crop]))
    _write_json_atomic(destination / "result.json", record)
    print(json.dumps(dict(case=case, baseline=baseline_metrics, geometry=metrics, graph=geometry)), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, choices=range(len(CASES)))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        records = [json.loads((args.output / case / "result.json").read_text()) for case, _ in CASES]
        summary = dict(development_only=True, algorithm="ct_sheet_signed_graph_v1", config=asdict(GeometryConfig()), records=records,
                       total_gt_children=sum(r["geometry_metrics"]["gt_instance_count"] for r in records),
                       baseline_recovered=sum(r["baseline_metrics"]["recovered_child_count"] for r in records),
                       geometry_recovered=sum(r["geometry_metrics"]["recovered_child_count"] for r in records),
                       baseline_false_splits=sum(r["baseline_metrics"]["false_split_child_count"] for r in records),
                       geometry_false_splits=sum(r["geometry_metrics"]["false_split_child_count"] for r in records))
        _write_json_atomic(args.output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    elif args.task_id is not None:
        case, fold = CASES[args.task_id]
        run_case(args.dataset, args.run_dir, args.diagnosis, args.output, case, fold)
    else:
        parser.error("Choose --task-id or --summary")


if __name__ == "__main__":
    main()
