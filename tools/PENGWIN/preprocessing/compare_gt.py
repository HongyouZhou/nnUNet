# Sanity-check helper: run old (morphology shell-overlap) and new (surface) ABBC GT
# generators on a single instance label file, then save both as NIfTI so they can be
# diff'd visually in ITK-Snap / Slicer / napari before committing to a full retrain.
#
# Usage:
#     PYTHONPATH=. python tools/PENGWIN/preprocessing/compare_gt.py \
#         --label  $PROJECT_HOME/dev/data/nnUNet_raw/Dataset777_Merged/labelsTr_raw/charite_103.nii.gz \
#         --outdir /tmp/abbc_compare
#
# Reads:  one instance-label NIfTI.
# Writes: <outdir>/<stem>_abbc_surface.nii.gz, <stem>_abbc_morphology.nii.gz,
#         <stem>_diff.nii.gz   (label-3 disagreement: 1=only morph, 2=only surface, 3=both)
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from tools.PENGWIN.preprocessing.preprocessing import (
    get_fracture_regions_morphology,
    get_fracture_regions_surface,
)
from tools.PENGWIN.utils.utils import MedVol


def _abbc_inplace(instances, *, d_threshold, d2_threshold, border_dilation,
                  core_exclusion_buffer, backend):
    """Re-implement _abbc here so we can switch backends per call without env-var hacks."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    from tools.PENGWIN.utils.bounding_boxes import (
        bounding_box_to_slice,
        get_bbox_from_mask,
    )

    instance_labels = np.unique(instances)
    instance_labels = instance_labels[instance_labels > 0]

    abbc_labels = np.zeros_like(instances, dtype=np.float32)
    for instance_label in instance_labels:
        object_mask = instances == instance_label
        bbox = get_bbox_from_mask(object_mask)
        slicer = bounding_box_to_slice(bbox)
        distance = distance_transform_edt(object_mask[slicer])
        distance = gaussian_filter(distance, sigma=2)
        gs = []
        for dim in range(instances.ndim):
            if distance.shape[dim] > 1:
                gs.append(np.gradient(np.gradient(distance, axis=dim), axis=dim))
            else:
                gs.append(np.zeros_like(distance))
        d = np.sum(np.stack(gs, axis=0), axis=0)
        inside_mask = -d > d_threshold
        inside_mask[distance > d2_threshold] = True
        abbc_labels[slicer][object_mask[slicer]] = 1
        abbc_labels[slicer][inside_mask] = 2

    if backend == "surface":
        gfr = get_fracture_regions_surface
    elif backend == "morphology":
        gfr = get_fracture_regions_morphology
    else:
        raise ValueError(backend)

    fractures_expanded = gfr(instances, border_dilation + core_exclusion_buffer)
    abbc_labels[(fractures_expanded > 0) & (abbc_labels == 2)] = 1
    fractures = gfr(instances, border_dilation)
    abbc_labels[fractures > 0] = 3
    return abbc_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Instance label NIfTI (raw, not ABBC)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--d_threshold", type=float, default=0.10)
    ap.add_argument("--d2_threshold", type=float, default=4.0)
    ap.add_argument("--surface_border_dilation", type=int, default=1)
    ap.add_argument("--surface_core_exclusion", type=int, default=6)
    ap.add_argument("--morph_diskradius", type=int, default=4)
    ap.add_argument("--morph_core_exclusion", type=int, default=8)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.label).name.replace(".nii.gz", "")

    instances_mv = MedVol(args.label)
    instances = instances_mv.array.astype(np.int32)

    print(f"Loaded {args.label}  shape={instances.shape}  "
          f"unique labels (head)={np.unique(instances)[:10]}")

    abbc_surf = _abbc_inplace(
        instances,
        d_threshold=args.d_threshold,
        d2_threshold=args.d2_threshold,
        border_dilation=args.surface_border_dilation,
        core_exclusion_buffer=args.surface_core_exclusion,
        backend="surface",
    )
    abbc_morph = _abbc_inplace(
        instances,
        d_threshold=args.d_threshold,
        d2_threshold=args.d2_threshold,
        border_dilation=args.morph_diskradius,
        core_exclusion_buffer=args.morph_core_exclusion,
        backend="morphology",
    )

    diff = np.zeros_like(abbc_surf, dtype=np.uint8)
    morph_b = abbc_morph == 3
    surf_b = abbc_surf == 3
    diff[morph_b & ~surf_b] = 1
    diff[~morph_b & surf_b] = 2
    diff[morph_b & surf_b] = 3

    surf_path = os.path.join(args.outdir, f"{stem}_abbc_surface.nii.gz")
    morph_path = os.path.join(args.outdir, f"{stem}_abbc_morphology.nii.gz")
    diff_path = os.path.join(args.outdir, f"{stem}_diff.nii.gz")

    MedVol(abbc_surf.astype(np.float32), copy=instances_mv).save(surf_path)
    MedVol(abbc_morph.astype(np.float32), copy=instances_mv).save(morph_path)
    MedVol(diff.astype(np.float32), copy=instances_mv).save(diff_path)

    n_surf = int(surf_b.sum())
    n_morph = int(morph_b.sum())
    n_only_morph = int((morph_b & ~surf_b).sum())
    n_only_surf = int((~morph_b & surf_b).sum())
    n_both = int((morph_b & surf_b).sum())
    print()
    print(f"Label-3 (border) voxel counts:")
    print(f"  surface:    {n_surf:>10d}")
    print(f"  morphology: {n_morph:>10d}")
    print(f"  only morph: {n_only_morph:>10d}  (likely over-detection in non-touching neighbors)")
    print(f"  only surf : {n_only_surf:>10d}  (touching pairs the morph backend missed)")
    print(f"  both      : {n_both:>10d}")
    print()
    print(f"Wrote: {surf_path}")
    print(f"       {morph_path}")
    print(f"       {diff_path}  (1=morph-only, 2=surface-only, 3=both)")


if __name__ == "__main__":
    main()
