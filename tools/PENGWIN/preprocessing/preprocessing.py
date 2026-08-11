# Copyright 2024, German Cancer Research Center (DKFZ) and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tools.PENGWIN.utils.utils import load_filenames, MedVol
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    convolve,
    gaussian_filter,
    binary_dilation as nd_binary_dilation,
    binary_erosion,
    maximum_filter,
    minimum_filter,
)
from skimage.morphology import disk, ball, binary_dilation
from tqdm import tqdm
from tools.PENGWIN.utils.bounding_boxes import *
from pathlib import Path
import os
from tools.PENGWIN.utils.parallel import tqdmp
import shutil

_MASK = np.array([1.0, 0.0, -1.0])


def get_fracture_regions_surface(mask, dilation_radius=0):
    """
    Inter-fragment surface via neighborhood-label disagreement (find_boundaries-style),
    optionally dilated to give the surface thickness.

    Why this instead of morphology shell-overlap:
    The previous ``get_fracture_regions`` computed (dilation ^ erosion) per instance
    and counted overlaps. That fails at distance = 0 (touching fragments): the erosion
    eats the thin contact region and the shell never overlaps. ``find_boundaries`` /
    neighborhood-disagreement always identifies the contact face — even at distance = 0.

    Implementation: a foreground voxel sits on the inter-fragment surface iff its
    3x3x3 neighborhood spans more than one non-zero label. Background neighbors are
    masked out via sentinels so fg-bg boundaries are not mistaken for inter-fragment.

    Args:
        mask: instance label array (0 = background, >0 = instance ids).
        dilation_radius: ball() radius for thickening the 1-vx surface.
            0 → keep 1-vx surface (good for regression targets).
            1–2 → thicken for class-based supervision (typical for ABBC label 3).

    Returns:
        uint8 array, 1 on inter-fragment surface (and its dilation), else 0.
    """
    fg = mask > 0
    if not np.any(fg):
        return np.zeros(mask.shape, dtype=np.uint8)

    # Use int32 sentinels well inside the type range. scipy's max/minimum_filter has
    # been observed to misbehave at values near np.iinfo(int64).min/max — keep
    # sentinels modest and the dtype small.
    label_max = int(mask.max())
    LO = np.int32(-1)                       # smaller than any valid fg label
    HI = np.int32(label_max + 1)            # larger than any valid fg label

    mask32 = mask.astype(np.int32)
    arr_for_max = np.where(fg, mask32, LO)
    nb_max = maximum_filter(arr_for_max, size=3)
    arr_for_min = np.where(fg, mask32, HI)
    nb_min = minimum_filter(arr_for_min, size=3)

    # Both extremes must come from foreground neighbors (i.e. not the sentinels).
    # On a fg voxel with at least two distinct fg labels in its 3x3x3 neighborhood,
    # nb_max > nb_min and both are valid label values (1 .. label_max).
    inter_fragment = (
        fg
        & (nb_max != nb_min)
        & (nb_max >= 1)
        & (nb_min >= 1)
        & (nb_min <= label_max)
    )

    if dilation_radius > 0:
        inter_fragment = nd_binary_dilation(
            inter_fragment, structure=ball(dilation_radius)
        )

    return inter_fragment.astype(np.uint8)


def get_fracture_regions_morphology(mask, diskradius=6):
    """Legacy shell-overlap detector. Kept for A/B comparison; superseded by
    ``get_fracture_regions_surface`` because it misses distance=0 contacts."""
    morphological_element = ball(diskradius // 2)
    unique_labels = np.unique(mask).astype(int)
    instance_ids = unique_labels[unique_labels > 0]

    overlap_map = np.zeros(mask.shape, dtype=np.uint8)
    for instance_id in instance_ids:
        instance_mask_bool = mask == instance_id
        bbox = get_bbox_from_mask(instance_mask_bool)
        bbox_padded = pad_bbox(bbox, diskradius, array_shape=mask.shape)
        slicer = bounding_box_to_slice(bbox_padded)
        submask = instance_mask_bool[slicer]
        instance_big = binary_dilation(submask, morphological_element)
        instance_small = binary_erosion(submask, morphological_element)
        boundary = instance_big ^ instance_small
        overlap_map[slicer] += boundary.astype(np.uint8)

    return (overlap_map > 1).astype(np.uint8)


# Default: surface-based detector. Set FRACTURE_REGION_BACKEND="morphology" to revert.
def get_fracture_regions(mask, dilation_radius_or_diskradius):
    backend = os.environ.get("FRACTURE_REGION_BACKEND", "surface").lower()
    if backend == "morphology":
        return get_fracture_regions_morphology(
            mask, diskradius=int(dilation_radius_or_diskradius)
        )
    return get_fracture_regions_surface(
        mask, dilation_radius=int(dilation_radius_or_diskradius)
    )


def partial(img, axis, mode="nearest"):
    """Return partial derivative of *img* w.r.t. *axis* direction"""
    d = len(img.shape)
    shm = np.identity(d) * 2 + 1
    m = np.reshape(_MASK, shm[axis].astype(int)) / 2.0
    return convolve(img, m, mode=mode)


def normalized_distance(object_mask, subset_mask):
    distance_to_object = distance_transform_edt(object_mask)
    distance_to_subset = distance_transform_edt(~subset_mask)
    normalized_distance = np.zeros_like(distance_to_object, dtype=np.float32)
    inside_object = object_mask & ~subset_mask
    normalized_distance[inside_object] = distance_to_object[inside_object] / (
        distance_to_subset[inside_object] + distance_to_object[inside_object]
    )
    normalized_distance[~object_mask] = 0
    normalized_distance[subset_mask] = 1
    return normalized_distance


def _abbc(
    name,
    datapath,
    outputpath,
    d_threshold,
    d2_threshold,
    border_dilation,
    core_exclusion_buffer=5,
):
    """Convert one instance label map to ABBC (boundary=1, core=2, border=3).

    Parameters
    ----------
    border_dilation : int
        Dilation radius for the inter-fragment surface that becomes label 3.
        With the surface-based backend, 1 gives a ~3-vx-thick band; 2 gives ~5-vx.
    core_exclusion_buffer : int
        Additional dilation on top of ``border_dilation`` used to demote core (2)
        back to boundary (1) in the neighborhood of every fracture, so cores from
        adjacent fragments don't fuse during training.
    """
    instances_mv = MedVol(os.path.join(datapath, f"{name}.nii.gz"))
    instances = instances_mv.array
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

    # Pass 1: demote cores back to boundary inside the *expanded* fracture neighborhood,
    # so adjacent cores stay separated during training.
    fractures_expanded = get_fracture_regions(
        instances, border_dilation + core_exclusion_buffer
    )
    abbc_labels[(fractures_expanded > 0) & (abbc_labels == 2)] = 1

    # Pass 2: stamp the actual inter-fragment surface (dilated) as label 3.
    fractures = get_fracture_regions(instances, border_dilation)
    abbc_labels[fractures > 0] = 3

    MedVol(abbc_labels, copy=instances_mv).save(
        os.path.join(outputpath, f"{name}.nii.gz")
    )


if __name__ == "__main__":
    labelsTr_path = os.path.join(
        os.environ["PROJECT_HOME"], "dev/data/nnUNet_raw/Dataset777_Merged/labelsTr"
    )
    labelsTr_raw_path = os.path.join(
        os.environ["PROJECT_HOME"], "dev/data/nnUNet_raw/Dataset777_Merged/labelsTr_raw"
    )

    if not os.path.exists(labelsTr_raw_path):
        shutil.move(labelsTr_path, labelsTr_raw_path)
        os.makedirs(labelsTr_path, exist_ok=True)

    datapath = labelsTr_raw_path
    base_outputpath = labelsTr_path
    processes = 6
    names = load_filenames(datapath, extension=".nii.gz")
    # Surface-based backend semantics (different from old morphology shell-overlap):
    #   border_dilation       = dilation radius of the 1-vx inter-fragment surface (label 3)
    #   core_exclusion_buffer = extra dilation when demoting nearby cores back to boundary
    # Tuned to give roughly the same band thickness as the old (f_radius=4,
    # core_exclusion_buffer=8) recipe: ~3-vx label-3 band, ~13-vx core exclusion zone.
    d_threshold = 0.10
    d2_threshold = 4
    border_dilation = 1
    core_exclusion_buffer = 6

    outputpath = base_outputpath

    tqdmp(
        _abbc,
        names,
        processes,
        desc="abbc",
        datapath=datapath,
        outputpath=outputpath,
        d_threshold=d_threshold,
        d2_threshold=d2_threshold,
        border_dilation=border_dilation,
        core_exclusion_buffer=core_exclusion_buffer,
    )
