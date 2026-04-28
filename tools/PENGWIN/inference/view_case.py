#!/usr/bin/env python3
"""Open one Charite case in napari with all relevant layers loaded.

Expected layout under --root:
    <root>/cases/<id>/                 raw delivered case
        ct.nii.gz                      original CT (also on disk under predict/imagesTs/)
        segmentations/tibia*.nii.gz    per-fragment GT masks (where available)
        *_instance_smooth_edt.nii.gz   prior pipeline output (where available)
    <root>/predict/
        imagesTs/case_<id>_0000.nii.gz
        results/abbc/case_<id>.nii.gz
        results/instances/watershed/case_<id>/case_<id>_instance.nii.gz
        results/cortical_split/case_<id>/instances_split.nii.gz
        gt/<src>_<id>/{<src>_<id>.nii.gz,<src>_<id>_0000.nii.gz}   training-set GT

Bypass napari's plugin readers (no .nii.gz support in this env) by loading via
nibabel and calling add_image / add_labels explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import napari


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="case id, e.g. 95 / 1681 / 4522")
    ap.add_argument("--root", type=Path,
                    default=Path("/home/hongyou/dev/data/charite"),
                    help="parent dir holding cases/ and predict/")
    args = ap.parse_args()

    c = args.case
    OLD = args.root / "cases" / c
    NEW = args.root / "predict"

    ct_path   = NEW / "imagesTs" / f"case_{c}_0000.nii.gz"
    abbc_path = NEW / "results" / "abbc" / f"case_{c}.nii.gz"
    inst_path = NEW / "results" / "instances" / "watershed" / f"case_{c}" / f"case_{c}_instance.nii.gz"
    split_path = NEW / "results" / "cortical_split" / f"case_{c}" / "instances_split.nii.gz"
    l3_added = NEW / "results" / "cortical_split" / f"case_{c}" / "label3_added.nii.gz"

    if not ct_path.exists():
        raise FileNotFoundError(ct_path)

    v = napari.Viewer()
    v.add_image(load(ct_path), name=f"CT {c}", colormap="gray",
                contrast_limits=(-200, 1500))

    if abbc_path.exists():
        v.add_labels(load(abbc_path).astype(np.int32), name="abbc (new)")
    if inst_path.exists():
        v.add_labels(load(inst_path).astype(np.int32), name="instance ws (new)")
    if split_path.exists():
        v.add_labels(load(split_path).astype(np.int32), name="instance after cortical_split")
    if l3_added.exists():
        v.add_labels(load(l3_added).astype(np.uint8), name="label3 added by cortical_split")

    for old_inst in OLD.glob("*instance_smooth_edt*.nii.gz"):
        v.add_labels(load(old_inst).astype(np.int32), name=f"instance (old) {old_inst.name}")

    seg_dir = OLD / "segmentations"
    if seg_dir.is_dir():
        for f in sorted(seg_dir.glob("tibia*.nii.gz")):
            v.add_labels((load(f) > 0).astype(np.uint8), name=f"seg/{f.name}")

    # GT abbc from training set, if previously rsynced into <root>/predict/gt/<src>_<id>/
    for gt_dir in (NEW / "gt").glob(f"*_{c}"):
        for f in sorted(gt_dir.glob("*.nii.gz")):
            arr = load(f)
            uniq = np.unique(arr)
            if f.name.endswith("_0000.nii.gz"):
                print(f"[view_case] image  : {f.name}  shape={arr.shape}  range=({arr.min():.0f},{arr.max():.0f})")
                v.add_image(arr, name=f"gt train CT ({f.name})", colormap="gray",
                            contrast_limits=(-200, 1500))
            else:
                lbl = arr.astype(np.int32)
                print(f"[view_case] labels : {f.name}  shape={arr.shape}  unique={uniq[:8].tolist()}")
                v.add_labels(lbl, name=f"gt abbc ({f.name})", opacity=0.6)

    napari.run()


if __name__ == "__main__":
    main()
