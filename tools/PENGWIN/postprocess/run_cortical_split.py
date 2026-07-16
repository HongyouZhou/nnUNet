"""Run automatic cortical splitting or interactive two-seed instance splitting.

Automatic mode processes the initial instance map using ABBC-derived seeds.
Manual mode is enabled by supplying an instance ID and two seed-mask NIfTIs;
it changes only that selected instance. Manual mode defaults to core-first
ownership and retains full-volume min-cut as a legacy comparison mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    SplitConfig,
    cortical_anchored_split,
    split_instance_core_first,
    split_instance_with_seeds,
)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abbc-pred", "--abbc_pred", dest="abbc_pred", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument(
        "--prob-label-3",
        "--prob_label_3",
        dest="prob_label_3",
        help="Optional softmax-channel-3 NIfTI used to route the cut",
    )
    ap.add_argument("--outdir", required=True)

    manual = ap.add_argument_group("manual two-seed mode")
    manual.add_argument("--instance-id", "--instance_id", dest="instance_id", type=int)
    manual.add_argument(
        "--source-seed",
        "--source_seed",
        dest="source_seed",
        help="Binary NIfTI scribble inside the first desired fragment",
    )
    manual.add_argument(
        "--sink-seed",
        "--sink_seed",
        dest="sink_seed",
        help="Binary NIfTI scribble inside the second desired fragment",
    )
    manual.add_argument(
        "--manual-split-mode",
        choices=("core-first", "full-mincut"),
        default="core-first",
        help="Core-first fixes fragment cores before assigning peripheral voxels",
    )

    params = ap.add_argument_group("split parameters")
    params.add_argument(
        "--concavity-min-relative",
        "--concavity_min_relative",
        "--concavity_min",
        dest="concavity_min_relative",
        type=float,
        default=0.20,
    )
    params.add_argument("--endpoint-min-distance", "--endpoint_min_distance", type=int, default=4)
    params.add_argument("--pair-max-distance", "--pair_max_distance", type=int, default=60)
    params.add_argument("--pair-max-normal-dot", "--pair_max_normal_dot", type=float, default=0.0)
    params.add_argument(
        "--w-prob3",
        "--w_prob3",
        "--cost_label3_weight",
        dest="w_prob3",
        type=float,
        default=4.0,
    )
    params.add_argument("--w-prob1", "--w_prob1", type=float, default=6.0)
    params.add_argument("--w-prob2", "--w_prob2", type=float, default=1.0)
    params.add_argument("--w-min", "--w_min", type=float, default=0.05)
    params.add_argument("--min-cortical-voxels", "--min_cortical_voxels", type=int, default=200)
    params.add_argument("--min-instance-voxels", "--min_instance_voxels", type=int, default=500)
    params.add_argument("--min-split-piece-size", "--min_split_piece_size", type=int, default=50)
    params.add_argument("--prob3-seed-threshold", "--prob3_seed_threshold", type=float, default=0.20)
    params.add_argument("--prob3-seed-min-size", "--prob3_seed_min_size", type=int, default=10)
    params.add_argument(
        "--core-anchor-erosion-mm",
        type=float,
        default=1.0,
        help="Physical erosion used to create robust interactive Label-2 cores",
    )
    params.add_argument(
        "--core-anchor-min-voxels",
        type=int,
        default=20,
        help="Minimum number of voxels in each selected core anchor",
    )
    params.add_argument(
        "--core-anchor-distinct-max-distance-mm",
        type=float,
        default=15.0,
        help="Maximum projection distance used to enforce two distinct core components",
    )
    params.add_argument(
        "--core-first-max-graph-voxels",
        type=int,
        default=5_000_000,
        help="Use core-seeded watershed above this bbox size; zero always uses watershed",
    )

    params.set_defaults(use_prob3_seeds=True)
    params.add_argument(
        "--also-use-pred-label3-as-seeds",
        "--also_use_pred_label3_as_seeds",
        dest="use_prob3_seeds",
        action="store_true",
        help="Accepted for compatibility; label-3 seeds are enabled by default",
    )
    params.add_argument(
        "--no-prob3-seeds",
        dest="use_prob3_seeds",
        action="store_false",
        help="Disable automatic seeds derived from label-3 probabilities",
    )
    return ap


def _load_image(path: str, name: str) -> sitk.Image:
    image = sitk.ReadImage(path)
    if image.GetDimension() != 3:
        raise SystemExit(f"{name} must be a 3D image, got {image.GetDimension()}D")
    return image


def _assert_same_grid(reference: sitk.Image, other: sitk.Image, name: str) -> None:
    if reference.GetSize() != other.GetSize():
        raise SystemExit(
            f"{name} size {other.GetSize()} != reference size {reference.GetSize()}"
        )
    fields = (
        ("spacing", reference.GetSpacing(), other.GetSpacing()),
        ("origin", reference.GetOrigin(), other.GetOrigin()),
        ("direction", reference.GetDirection(), other.GetDirection()),
    )
    for field, ref_value, other_value in fields:
        if not np.allclose(ref_value, other_value):
            raise SystemExit(f"{name} does not share the reference {field}")


def _load_aligned_mask(path: str, reference: sitk.Image, name: str) -> np.ndarray:
    image = _load_image(path, name)
    _assert_same_grid(reference, image, name)
    return sitk.GetArrayFromImage(image) != 0


def _write_like(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def main() -> None:
    args = _parser().parse_args()
    manual_values = (args.instance_id, args.source_seed, args.sink_seed)
    manual_mode = any(value is not None for value in manual_values)
    if manual_mode and not all(value is not None for value in manual_values):
        raise SystemExit(
            "manual mode requires --instance-id, --source-seed, and --sink-seed together"
        )

    try:
        import maxflow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("PyMaxflow is required for cortical splitting") from exc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_image = _load_image(args.abbc_pred, "abbc_pred")
    inst_image = _load_image(args.instances, "instances")
    _assert_same_grid(pred_image, inst_image, "instances")
    abbc_pred = sitk.GetArrayFromImage(pred_image)
    instances = sitk.GetArrayFromImage(inst_image)
    if not np.issubdtype(abbc_pred.dtype, np.integer):
        raise SystemExit(f"abbc_pred must use an integer pixel type, got {abbc_pred.dtype}")
    if not np.issubdtype(instances.dtype, np.integer):
        raise SystemExit(f"instances must use an integer pixel type, got {instances.dtype}")

    prob_l3 = None
    if args.prob_label_3 is not None:
        prob_image = _load_image(args.prob_label_3, "prob_label_3")
        _assert_same_grid(pred_image, prob_image, "prob_label_3")
        prob_l3 = sitk.GetArrayFromImage(prob_image).astype(np.float32)

    cfg = SplitConfig(
        min_cortical_voxels=args.min_cortical_voxels,
        min_instance_voxels=args.min_instance_voxels,
        concavity_min_relative=args.concavity_min_relative,
        endpoint_min_distance=args.endpoint_min_distance,
        prob3_seed_threshold=args.prob3_seed_threshold,
        prob3_seed_min_size=args.prob3_seed_min_size,
        core_anchor_erosion_mm=args.core_anchor_erosion_mm,
        core_anchor_min_voxels=args.core_anchor_min_voxels,
        core_anchor_distinct_max_distance_mm=args.core_anchor_distinct_max_distance_mm,
        core_first_max_graph_voxels=args.core_first_max_graph_voxels,
        pair_max_distance=args.pair_max_distance,
        pair_max_normal_dot=args.pair_max_normal_dot,
        w_prob3=args.w_prob3,
        w_prob1=args.w_prob1,
        w_prob2=args.w_prob2,
        w_min=args.w_min,
        min_split_piece_size=args.min_split_piece_size,
    )

    print(f"Volume shape: {abbc_pred.shape}")
    print(f"Initial instances: {np.count_nonzero(np.unique(instances))}")
    mode = f"manual {args.manual_split_mode}" if manual_mode else "automatic"
    print(f"Mode: {mode}")

    if manual_mode:
        source_seed = _load_aligned_mask(args.source_seed, pred_image, "source_seed")
        sink_seed = _load_aligned_mask(args.sink_seed, pred_image, "sink_seed")
        if args.manual_split_mode == "core-first":
            instances_out, label3_added, diag = split_instance_core_first(
                abbc_pred=abbc_pred,
                instances=instances,
                instance_id=args.instance_id,
                source_interaction_mask=source_seed,
                sink_interaction_mask=sink_seed,
                prob_label_3=prob_l3,
                spacing_zyx=pred_image.GetSpacing()[::-1],
                cfg=cfg,
            )
        else:
            instances_out, label3_added, diag = split_instance_with_seeds(
                abbc_pred=abbc_pred,
                instances=instances,
                instance_id=args.instance_id,
                source_seed_mask=source_seed,
                sink_seed_mask=sink_seed,
                prob_label_3=prob_l3,
                cfg=cfg,
            )
    else:
        instances_out, label3_added, diag = cortical_anchored_split(
            abbc_pred=abbc_pred,
            instances=instances,
            prob_label_3=prob_l3,
            use_prob3_seeds=args.use_prob3_seeds,
            cfg=cfg,
        )

    n_before = np.count_nonzero(np.unique(instances))
    n_after = np.count_nonzero(np.unique(instances_out))
    print(f"Instances: {n_before} -> {n_after}")
    print(f"Cut-boundary voxels: {int(label3_added.sum())}")

    out_inst = outdir / "instances_split.nii.gz"
    out_l3 = outdir / "label3_added.nii.gz"
    out_diag = outdir / "diagnostics.json"
    _write_like(instances_out.astype(np.uint32), inst_image, out_inst)
    _write_like(label3_added.astype(np.uint8), inst_image, out_l3)
    with out_diag.open("w", encoding="utf-8") as handle:
        json.dump(diag, handle, indent=2)

    print(f"Wrote {out_inst}")
    print(f"Wrote {out_l3}")
    print(f"Wrote {out_diag}")


if __name__ == "__main__":
    main()
