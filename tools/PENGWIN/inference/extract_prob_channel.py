#!/usr/bin/env python3
"""Extract one softmax channel from nnUNetv2's .npz probability dumps.

nnUNetv2 with --save_probabilities writes <case>.npz alongside <case>.nii.gz in
the prediction folder. The npz holds a (C, Z, Y, X) array under key
``probabilities``. We pull a single channel out and save it as a NIfTI sharing
the affine/header of the matching <case>.nii.gz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abbc_dir", required=True, type=Path,
                    help="Directory holding <case>.nii.gz and <case>.npz pairs.")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Where to write per-case <case>_prob<channel>.nii.gz")
    ap.add_argument("--channel", type=int, default=3,
                    help="Softmax channel to extract (default 3 = fracture surface).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.abbc_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz under {args.abbc_dir}")

    for npz in npz_files:
        case = npz.stem
        ref = args.abbc_dir / f"{case}.nii.gz"
        if not ref.exists():
            print(f"[skip] no reference nifti for {case}")
            continue

        ref_img = nib.load(str(ref))
        with np.load(npz) as f:
            keys = list(f.keys())
            if "probabilities" in keys:
                probs = f["probabilities"]
            else:
                probs = f[keys[0]]
        if probs.ndim != 4:
            raise SystemExit(f"{npz}: expected (C,Z,Y,X), got {probs.shape}")
        if args.channel >= probs.shape[0]:
            raise SystemExit(f"{npz}: channel {args.channel} out of range "
                             f"(C={probs.shape[0]})")

        # nnUNetv2 stores (C, Z, Y, X); the matching nifti is (X, Y, Z).
        # Bring channel into (X, Y, Z) by transposing the spatial axes.
        ch = np.asarray(probs[args.channel], dtype=np.float32)
        if ch.shape != ref_img.shape:
            ch = np.transpose(ch, (2, 1, 0))
        if ch.shape != ref_img.shape:
            raise SystemExit(
                f"{npz}: prob shape {probs[args.channel].shape} not reconcilable "
                f"with ref {ref_img.shape}"
            )

        out = args.out_dir / f"{case}_prob{args.channel}.nii.gz"
        nib.save(nib.Nifti1Image(ch, ref_img.affine, ref_img.header), str(out))
        print(f"[ok] {case}: channel {args.channel} -> {out}")


if __name__ == "__main__":
    main()
