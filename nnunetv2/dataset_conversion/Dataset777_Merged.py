import glob
import os
import re
import shutil
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------------------------
TARGET_DATASET_ID = 777
TARGET_DATASET_NAME = f"Dataset{TARGET_DATASET_ID:03d}_Merged"

# Using PROJECT_HOME for paths as requested
PROJECT_HOME = os.environ.get('PROJECT_HOME', '/home/hongyou')  # Fallback just in case

# Dataset989 Paths
# BASE_CHARITE: "hongyou/data/04_08_charite" (relative to PROJECT_HOME based on previous context, but user said $PROJECT_HOME/data/...)
BASE_CHARITE = join(PROJECT_HOME, "data", "04_08_charite")
BASE_SEG = join(PROJECT_HOME, "data", "07_12_seg")

# Dataset999 Paths (User said absolute path provided was fine: /ssdArray/hongyou/data/PENGWIN)
BASE_PENGWIN = "/ssdArray/hongyou/data/PENGWIN"

# ---------------------------------------------------------------------------
# DATASET 989 LOGIC (Adapted)
# ---------------------------------------------------------------------------
BONES = ["tibia", "fibula", "femur", "patella", "fabella"]
FRAGMENT_LIMITS = {
    "tibia": 20,
    "fibula": 10,
    "femur": 5,
    "patella": 5,
    "fabella": 5,
}

SEG_NAME_RE = re.compile(r"^seg_(tibia|fibula|femur|patella|fabella)_(left|right)_(\d+)$")
SEG_FRAGMENT_RE = re.compile(r"^seg_fragment_(left|right)_(\d+)$")
SEG_GENERIC_RE = re.compile(r"^segment(?:_[0-9]+)*(?:_[0-9]+)?$")

def _build_charite_labels_dict():
    labels = {"background": 0}
    next_idx = 1
    for bone in BONES:
        for i in range(1, FRAGMENT_LIMITS[bone] + 1):
            labels[f"{bone}_fragment_{i}"] = next_idx
            next_idx += 1
    return labels, next_idx - 1  # Return valid max index

def _pick_first_existing(paths):
    for p in paths:
        if isfile(p):
            return p
    return None

def _find_charite_case_paths(case):
    case_dir = join(BASE_CHARITE, case, "axial")
    ct_path = join(case_dir, f"{case}_0000_input.nii.gz")
    # Priority to 'bearbeitet' as per user request
    seg_candidates = [
        join(case_dir, "bearbeitet", "seg_v1.nii.gz.seg.nrrd"),
        join(case_dir, "seg_v1.nii.gz.seg.nrrd"),
    ]
    seg_path = _pick_first_existing(seg_candidates)
    if not isfile(ct_path) or seg_path is None:
        return None, None, None
    out_case_id = f"charite_{case}" # Rename
    return (ct_path, seg_path, out_case_id)

def _find_seg_case_paths(case):
    case_dir = join(BASE_SEG, case)
    ct_candidates = sorted(glob.glob(join(case_dir, "*_img_data.nii.gz")))
    ct_path = ct_candidates[0] if ct_candidates else None
    seg_candidates = sorted(glob.glob(join(case_dir, "*.seg.nrrd")))
    seg_path = seg_candidates[0] if seg_candidates else None
    if ct_path is None or seg_path is None:
        return None, None, None
    out_case_id = f"charite_{case}" # Rename to share namespace
    return (ct_path, seg_path, out_case_id)

def _next_available_fragment(bone, used_fragments):
    existing = used_fragments.get(bone, set())
    for k in range(1, FRAGMENT_LIMITS[bone] + 1):
        if k not in existing:
            return k
    return None

def _load_charite_seg_mask(seg_img, ct_img, labels_dict):
    """Return numpy mask aligned to ct_img with mapped label values."""
    seg_array_out = sitk.GetArrayFromImage(ct_img)
    seg_array_out[:] = 0

    used_fragments = {bone: set() for bone in BONES}
    is_vector = seg_img.GetNumberOfComponentsPerPixel() > 1
    
    seg_scalar_arr = None
    if not is_vector:
        # Resample if scalar and needed (simplified from original)
        seg_img_rs = sitk.Resample(
            seg_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg_img.GetPixelID()
        )
        seg_scalar_arr = sitk.GetArrayFromImage(seg_img_rs)
        # Handle shape mismatch with cropping/padding (same as original)
        if seg_scalar_arr.shape != seg_array_out.shape:
            zt, yt, xt = seg_array_out.shape
            zs, ys, xs = seg_scalar_arr.shape
            seg_scalar_arr = seg_scalar_arr[: min(zs, zt), : min(ys, yt), : min(xs, xt)]
            if seg_scalar_arr.shape != seg_array_out.shape:
                pad_z = max(0, zt - seg_scalar_arr.shape[0])
                pad_y = max(0, yt - seg_scalar_arr.shape[1])
                pad_x = max(0, xt - seg_scalar_arr.shape[2])
                seg_scalar_arr = np.pad(
                    seg_scalar_arr,
                    ((0, pad_z), (0, pad_y), (0, pad_x)),
                    constant_values=0,
                )

    i = 0
    # Loop through metadata segments
    while seg_img.HasMetaDataKey(f"Segment{i}_Name"):
        seg_name_raw = seg_img.GetMetaData(f"Segment{i}_Name")
        seg_name = seg_name_raw.strip().lower()

        bone = None
        frag_idx = None

        m = SEG_NAME_RE.match(seg_name)
        if m:
            bone, _side, idx_str = m.groups()
            frag_idx = int(idx_str)
        else:
            m2 = SEG_FRAGMENT_RE.match(seg_name)
            if m2:
                bone = "tibia"
                frag_idx = int(m2.groups()[1])
            elif SEG_GENERIC_RE.match(seg_name):
                bone = "tibia"
                frag_idx = _next_available_fragment(bone, used_fragments)
            else:
                # print(f"  [SKIP] Unknown: {seg_name_raw}")
                i += 1
                continue

        if frag_idx is None or frag_idx > FRAGMENT_LIMITS[bone]:
            i += 1
            continue

        used_fragments.setdefault(bone, set()).add(frag_idx)
        key = f"{bone}_fragment_{frag_idx}"
        label_val = labels_dict.get(key)
        if label_val is None:
            i += 1
            continue

        if is_vector:
            comp_img = sitk.VectorIndexSelectionCast(seg_img, i, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(comp_img) > 0
        else:
            lv_key = f"Segment{i}_LabelValue"
            if seg_img.HasMetaDataKey(lv_key):
                lv = int(seg_img.GetMetaData(lv_key))
                mask = seg_scalar_arr == lv
            else:
                i += 1
                continue

        write_mask = mask & (seg_array_out == 0)
        seg_array_out[write_mask] = label_val
        i += 1

    out_seg = sitk.GetImageFromArray(seg_array_out.astype(np.uint8))
    out_seg.CopyInformation(ct_img)
    return out_seg

def _process_charite_case(ct_path, seg_path, out_case_id, images_tr, labels_tr, labels_dict):
    print(f"Processing Charite: {out_case_id}")
    ct_img = sitk.ReadImage(ct_path)
    sitk.WriteImage(ct_img, join(images_tr, f"{out_case_id}_0000.nii.gz"))

    seg_img = sitk.ReadImage(seg_path)
    out_seg = _load_charite_seg_mask(seg_img, ct_img, labels_dict)
    sitk.WriteImage(out_seg, join(labels_tr, f"{out_case_id}.nii.gz"))

# ---------------------------------------------------------------------------
# DATASET 999 LOGIC (Adapted & Shifted)
# ---------------------------------------------------------------------------
def _build_pengwin_labels_dict(shift_offset):
    # Original logic:
    # 1..10 SA
    # 11..20 LH
    # 21..30 RH
    labels = {}
    for i in range(1, 11):
        labels[f'SA_{i}'] = i + shift_offset
    for i in range(11, 21):
        labels[f'LH_{i}'] = i + shift_offset
    for i in range(21, 31):
        labels[f'RH_{i}'] = i + shift_offset
    return labels

def _process_pengwin_case(case_filename, shift_offset, images_tr, labels_tr):
    # case_filename like "case_01.mha"
    case_name = case_filename.replace('.mha', '')
    out_case_id = f"pengwin_{case_name}"
    print(f"Processing Pengwin: {out_case_id}")

    # Paths
    ct_in = join(BASE_PENGWIN, 'images', case_filename)
    seg_in = join(BASE_PENGWIN, 'labels', case_filename)
    
    # CT
    im = sitk.ReadImage(ct_in)
    sitk.WriteImage(im, join(images_tr, f"{out_case_id}_0000.nii.gz"))

    # Seg - Needs shift
    seg = sitk.ReadImage(seg_in)
    seg_arr = sitk.GetArrayFromImage(seg)
    
    # Shift labels: only values > 0
    # Caution: Assuming input uses 1..30 range.
    # Prevent overflow if uint8, but 30+45=75 fits in uint8.
    mask = seg_arr > 0
    seg_arr[mask] = seg_arr[mask] + shift_offset
    
    new_seg = sitk.GetImageFromArray(seg_arr)
    new_seg.CopyInformation(seg)
    sitk.WriteImage(new_seg, join(labels_tr, f"{out_case_id}.nii.gz"))

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_out = join(nnUNet_raw, TARGET_DATASET_NAME)
    imagesTr = join(base_out, "imagesTr")
    labelsTr = join(base_out, "labelsTr")
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # 1. Build Merged Labels
    charite_labels, charite_max_idx = _build_charite_labels_dict()
    pengwin_labels = _build_pengwin_labels_dict(shift_offset=charite_max_idx)
    
    merged_labels = charite_labels.copy() # background: 0 + charite labels
    # Verify no collision (shouldn't happen with logic)
    for k, v in pengwin_labels.items():
        if k in merged_labels:
            print(f"WARNING: Label name collision {k}!")
        if v in merged_labels.values():
             print(f"WARNING: Label value collision {v}!")
        merged_labels[k] = v

    exported = 0

    # 2. Process Charite
    cases_charite = sorted(subdirs(BASE_CHARITE, prefix="", join=False))
    cases_seg = sorted(subdirs(BASE_SEG, prefix="", join=False))

    for case in cases_charite:
        ct_path, seg_path, out_case_id = _find_charite_case_paths(case)
        if ct_path and seg_path:
            _process_charite_case(ct_path, seg_path, out_case_id, imagesTr, labelsTr, charite_labels)
            exported += 1
        else:
            print(f"[SKIP Charite] {case}: missing files")

    for case in cases_seg:
        ct_path, seg_path, out_case_id = _find_seg_case_paths(case)
        if ct_path and seg_path:
            _process_charite_case(ct_path, seg_path, out_case_id, imagesTr, labelsTr, charite_labels)
            exported += 1
        else:
             print(f"[SKIP Seg] {case}: missing files")

    # 3. Process PENGWIN
    if isdir(join(BASE_PENGWIN, 'images')):
        cases_pengwin = subfiles(join(BASE_PENGWIN, 'images'), join=False, prefix='')
        for case in cases_pengwin:
             _process_pengwin_case(case, charite_max_idx, imagesTr, labelsTr)
             exported += 1
    else:
        print(f"[SKIP Pengwin] Directory not found: {BASE_PENGWIN}")


    # 4. Generate JSON
    generate_dataset_json(
        base_out,
        {0: "CT"},
        merged_labels,
        exported,
        ".nii.gz",
        None,
        TARGET_DATASET_NAME,
        overwrite_image_reader_writer="NibabelIOWithReorient",
        reference="Charite merged with PENGWIN",
        license="private"
    )
    print(f"Finished! Exported {exported} cases to {TARGET_DATASET_NAME}.")
