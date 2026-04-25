"""CLI: run cortical_anchored_split on a real ABBC prediction file.

Inputs
------
--abbc_pred  : NIfTI of network argmax (0=bg, 1=boundary, 2=core, 3=border)
--instances  : NIfTI of the abbc2instance output (initial instance map)
--prob_label_3 : (optional) NIfTI of softmax channel 3 (used in cut routing)
--outdir

Outputs
-------
<outdir>/instances_split.nii.gz   — refined instance map
<outdir>/label3_added.nii.gz       — voxels stamped as new label 3 by the splitter
<outdir>/diagnostics.json          — per-instance counts (endpoints, splits)

Run on a held-out failing case to validate the geometric assumption (cortical
truncation == fracture endpoint). If the splitter does not split anything on a
known-merged instance, log the endpoints + concavity field for visual inspection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.PENGWIN.postprocess.cortical_anchored_split import (
    SplitConfig,
    cortical_anchored_split,
    detect_endpoints,
)
from tools.PENGWIN.utils.utils import MedVol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abbc_pred", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--prob_label_3", default=None,
                    help="Optional softmax-channel-3 NIfTI for cut routing")
    ap.add_argument("--outdir", required=True)
    # Hyperparameters with sensible defaults; tune per case
    ap.add_argument("--concavity_min", type=float, default=0.15)
    ap.add_argument("--endpoint_min_distance", type=int, default=4)
    ap.add_argument("--pair_max_distance", type=int, default=40)
    ap.add_argument("--pair_max_normal_dot", type=float, default=-0.2)
    ap.add_argument("--cost_label3_weight", type=float, default=1.0)
    ap.add_argument("--cost_distance_weight", type=float, default=0.3)
    ap.add_argument("--cut_thickness", type=int, default=1)
    ap.add_argument("--min_cortical_voxels", type=int, default=200)
    ap.add_argument("--also_use_pred_label3_as_seeds", action="store_true",
                    help="Bootstrap endpoints off the model's partial label-3 prediction")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    pred_mv = MedVol(args.abbc_pred)
    inst_mv = MedVol(args.instances)
    abbc_pred = pred_mv.array.astype(np.int32)
    instances = inst_mv.array.astype(np.int32)

    if abbc_pred.shape != instances.shape:
        raise SystemExit(
            f"shape mismatch: abbc_pred={abbc_pred.shape}, instances={instances.shape}"
        )

    prob_l3 = None
    if args.prob_label_3 is not None:
        p3_mv = MedVol(args.prob_label_3)
        prob_l3 = p3_mv.array.astype(np.float32)
        if prob_l3.shape != abbc_pred.shape:
            raise SystemExit(
                f"prob_label_3 shape {prob_l3.shape} != pred {abbc_pred.shape}"
            )

    cfg = SplitConfig(
        concavity_min=args.concavity_min,
        endpoint_min_distance=args.endpoint_min_distance,
        pair_max_distance=args.pair_max_distance,
        pair_max_normal_dot=args.pair_max_normal_dot,
        cost_label3_weight=args.cost_label3_weight,
        cost_distance_weight=args.cost_distance_weight,
        cut_thickness=args.cut_thickness,
        min_cortical_voxels=args.min_cortical_voxels,
    )

    print(f"Volume shape: {abbc_pred.shape}")
    print(f"Initial instances: {len(np.unique(instances)) - 1}")
    print(f"Cortical voxels: {(abbc_pred == 1).sum()}")
    print(f"Border voxels (model label 3): {(abbc_pred == 3).sum()}")

    # Optional: per-instance endpoint dump for debugging
    print("\n[dry-run] per-instance endpoint preview (top 3 instances by size):")
    sizes = {int(i): int((instances == i).sum()) for i in np.unique(instances) if i > 0}
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:3]
    for inst_id, sz in top:
        cortical_only = (instances == inst_id) & (abbc_pred == 1)
        seed_mask = None
        if args.also_use_pred_label3_as_seeds:
            seed_mask = (instances == inst_id) & (abbc_pred == 3)
        eps = detect_endpoints(cortical_only, cfg, extra_seed_mask=seed_mask)
        print(f"  inst={inst_id}  size={sz}  cortical={int(cortical_only.sum())}  endpoints={len(eps)}")

    print("\nRunning cortical_anchored_split ...")
    instances_out, l3_added, diag = cortical_anchored_split(
        abbc_pred=abbc_pred,
        instances=instances,
        prob_label_3=prob_l3,
        cfg=cfg,
    )

    n_before = len(np.unique(instances)) - 1
    n_after = len(np.unique(instances_out)) - 1
    print(f"\n{n_before} → {n_after} instances ({n_after - n_before} new from splits)")
    print(f"label-3 voxels added: {int(l3_added.sum())}")

    out_inst = os.path.join(args.outdir, "instances_split.nii.gz")
    out_l3 = os.path.join(args.outdir, "label3_added.nii.gz")
    out_diag = os.path.join(args.outdir, "diagnostics.json")

    MedVol(instances_out.astype(np.uint16), copy=inst_mv).save(out_inst)
    MedVol(l3_added.astype(np.uint8), copy=inst_mv).save(out_l3)
    with open(out_diag, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out_inst}")
    print(f"  {out_l3}")
    print(f"  {out_diag}")


if __name__ == "__main__":
    main()
