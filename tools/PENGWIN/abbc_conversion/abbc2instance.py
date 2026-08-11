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


import copy
import timeit
from typing import Optional, Tuple, Type

import cc3d
import numpy as np
import numpy_indexed as npi
import skfmm
from scipy.ndimage import label as nd_label
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import cube, dilation

from tools.PENGWIN.utils.bounding_boxes import *
from tools.PENGWIN.utils.parallel import tqdmp


def abbc2instance(abbc: np.ndarray, core_label: int = 2, boundary_label: int = 1, border_label: int = 3,
                  processes: Optional[int] = None, progressbar: bool = False, dtype: Type = np.uint16) -> Tuple[
    np.ndarray, int]:
    """
    Convert the adaptive boundary border core segmentation of an entire image into an instance segmentation.

    Args:
        abbc (np.ndarray): The abbc segmentation of the entire image.
        core_label (Optional[int], default=2): The core label of the adaptive boundary border segmentation.
        boundary_label (Optional[int], default=1): The boundary label of the adaptive boundary border segmentation.
        border_label (Optional[int], default=3): The border label of the adaptive boundary border segmentation.
        processes (Optional[int], default=None): Number of processes to use. If None, it uses a single process.
        progressbar (bool, default=False): Whether to show progress bar.
        dtype (Type, default=np.uint16): The data type for the output segmentation.

    Returns:
        Tuple[np.ndarray, int]: The instance segmentation of the entire image, Number of instances.
    """

    abbc_array = np.array(abbc)

    # component_seg = cc3d.connected_components(abbc_array > 0)
    component_seg = cc3d.connected_components((abbc_array == core_label) | (abbc_array == boundary_label))
    component_seg = component_seg.astype(dtype)
    instances = np.zeros_like(abbc, dtype=dtype)
    num_instances = 0
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(component_seg)["bounding_boxes"])}
    del props[0]

    border_core_component2instance = border_core_component2instance_fmm

    border_core_patches = []
    for index, (label, bbox) in enumerate(props.items()):
        filter_mask = component_seg[bbox] == label
        border_core_patch = copy.deepcopy(abbc[bbox])
        border_core_patch[filter_mask != 1] = 0
        border_core_patches.append(border_core_patch)

    instances_patches = tqdmp(border_core_component2instance, border_core_patches, processes,
                              desc="Border-Core2Instance", disable=not progressbar, core_label=core_label,
                              border_label=border_label, boundary_label=boundary_label)

    for index, (label, bbox) in enumerate(props.items()):
        instances_patch = instances_patches[index].astype(dtype)
        instances_patch[instances_patch > 0] += num_instances
        num_instances = max(num_instances, int(np.max(instances_patch)))
        patch_labels = np.unique(instances_patch)
        patch_labels = patch_labels[patch_labels > 0]
        for patch_label in patch_labels:
            instances[bbox][instances_patch == patch_label] = patch_label

    return instances, num_instances


def abbc_refine_border(mask, instances, processes, dtype: Type = np.uint16):
    """
    Convert the border-core segmentation of an entire image into an instance segmentation.

    Args:
        border_core (np.ndarray): The border-core segmentation of the entire image.
        processes (Optional[int], default=None): Number of processes to use. If None, it uses a single process.
        progressbar (bool, default=False): Whether to show progress bar.
        dtype (Type, default=np.uint16): The data type for the output segmentation.

    Returns:
        Tuple[np.ndarray, int]: The instance segmentation of the entire image, Number of instances.
    """

    # border_core_array = np.array(mask)
    component_seg = cc3d.connected_components(mask > 0)
    component_seg = component_seg.astype(dtype)
    instances_final = np.zeros_like(instances, dtype=dtype)
    num_instances = 0
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(component_seg)["bounding_boxes"])}
    del props[0]

    instance_patches = []
    for index, (label, bbox) in enumerate(props.items()):
        # volume = np.prod([s.stop - s.start for s in bbox])
        if np.max(instances[component_seg==label]) > 0:  # otherwise no core... might need to add new thingy if its big.
            filter_mask = component_seg[bbox] == label
            border_core_patch = copy.deepcopy(instances[bbox])
            border_core_patch[~filter_mask] = 0
            border_core_patch[np.logical_and(filter_mask, border_core_patch == 0)] = 1000
            instance_patches.append(border_core_patch)
        else:
            print("here!")

    instances_patches = tqdmp(refine_instances_dilation, instance_patches, processes,
                              desc="refine_instances_dilation")

    real_id = 0
    for index, (label, bbox) in enumerate(props.items()):
        # volume = np.prod([s.stop - s.start for s in bbox])
        if np.max(instances[component_seg==label]) > 0:  # otherwise no core... might need to add new thingy if its big.
            instances_patch = instances_patches[real_id].astype(dtype)
            real_id += 1
            # instances_patch[instances_patch > 0] += num_instances
            # num_instances = max(num_instances, int(np.max(instances_patch)))
            # patch_labels = np.unique(instances_patch)
            # patch_labels = patch_labels[patch_labels > 0]
            instances[bbox][instances_patch > 1] = instances_patch[instances_patch > 1]

    return instances


def refine_instances_dilation(patch: np.ndarray):
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a morphological dilation operation.

    This method starts with the core instances and progressively dilates them until all border voxels are covered, hence performing the instance segmentation.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    border = patch == 1000
    core_instances = copy.deepcopy(patch)
    core_instances[core_instances == 1000] = 0
    instances = patch
    while np.sum(border) > 0:
        ball_here = cube(3)

        dilated = dilation(core_instances, ball_here)
        dilated[patch == 0] = 0
        diff = (core_instances == 0) & (dilated != core_instances)
        instances[diff & border] = dilated[diff & border]
        border[diff] = 0
        core_instances = dilated

    return instances


def border_core_component2instance_fmm(patch: np.ndarray, core_label: int, border_label: int,
                                       boundary_label: int) -> np.ndarray:
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a morphological dilation operation.

    This method starts with the core instances and progressively dilates them until all border voxels are covered, hence performing the instance segmentation.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    patch_orig = patch.copy()
    patch[patch == border_label] = 0 #the boundary label might not be symmetric around the border/fracture. thus do not fill it at this step but later on!
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    if num_instances == 0:
        return patch
    patch, core_instances, num_instances = remove_small_cores_simple(patch, core_instances, core_label, boundary_label, min_size=50)
    # patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, boundary_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label,
                             output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch
    instances = copy.deepcopy(core_instances)

    minimum_distance = np.zeros_like(patch, dtype=np.single) + np.inf

    mask = patch == 0
    for instance in range(1, num_instances + 1):
        border = np.ones_like(instances, dtype=bool)
        border[core_instances == instance] = False

        phi = np.ma.MaskedArray(border, mask)
        dist = skfmm.distance(phi)
        boringdist = dist.data
        if isinstance(dist, np.ma.MaskedArray):
            boringdist[dist.mask] = np.inf
        instances[np.logical_and(boringdist < minimum_distance, core_instances == 0)] = instance
        minimum_distance = np.minimum(dist, minimum_distance)
    if num_instances > 1:
        # print('here!')
        patch = patch_orig
        core_instances = copy.deepcopy(instances)
        # patch[patch == border_label] =
        mask = patch == 0
        minimum_distance = np.zeros_like(patch, dtype=np.single) + np.inf
        for instance in range(1, num_instances + 1):
            border = np.ones_like(instances, dtype=bool)
            border[core_instances == instance] = False

            phi = np.ma.MaskedArray(border, mask)
            dist = skfmm.distance(phi)
            boringdist = dist.data
            boringdist[dist.mask] = np.inf
            instances[np.logical_and(boringdist < minimum_distance, core_instances == 0)] = instance
            minimum_distance = np.minimum(dist, minimum_distance)

    return instances


def add_mask_to_closest_instance(instances, mask):
    bbox = get_bbox_from_mask(mask)
    slicer = bounding_box_to_slice(bbox)

    core_instances = instances
    instances = copy.deepcopy(core_instances)

    minimum_distance = np.zeros_like(instances[slicer], dtype=np.single) + np.inf

    # mask = mask
    unique_instances = np.unique(instances[slicer])
    unique_instances = unique_instances[unique_instances > 0]
    for instance in unique_instances:
        border = np.ones_like(instances[slicer], dtype=bool)
        border[core_instances[slicer] == instance] = False

        phi = np.ma.MaskedArray(border[slicer], mask[slicer])
        dist = skfmm.distance(phi, narrow=5)
        boringdist = dist.data
        if isinstance(dist, np.ma.MaskedArray):
            boringdist[dist.mask] = np.inf
        instances[slicer][np.logical_and(boringdist < minimum_distance, core_instances == 0)] = instance
        minimum_distance = np.minimum(dist, minimum_distance)

    return instances


def border_core_component2instance_dilation(patch: np.ndarray, core_label: int, border_label: int,
                                            boundary_label: int) -> np.ndarray:
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a morphological dilation operation.

    This method starts with the core instances and progressively dilates them until all border voxels are covered, hence performing the instance segmentation.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    patch[patch == border_label] = boundary_label
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    if num_instances == 0:
        return patch
    patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label,
                             output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch
    instances = copy.deepcopy(core_instances)
    border = patch == boundary_label
    while np.sum(border) > 0:
        ball_here = cube(3)

        dilated = dilation(core_instances, ball_here)
        dilated[patch == 0] = 0
        diff = (core_instances == 0) & (dilated != core_instances)
        instances[diff & border] = dilated[diff & border]
        border[diff] = 0
        core_instances = dilated

    return instances


def remove_small_cores(
        patch: np.ndarray,
        core_instances: np.ndarray,
        core_label: int,
        border_label: int,
        min_distance: float = 5,  # Original: 1 (But bad for the challenge)
        min_ratio_threshold: float = 0.95,
        max_distance: float = 3,
        max_ratio_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Removes small cores in a patch based on the distance transform of the core label.

    Args:
        patch (np.ndarray): An entire connected component patch of the border-core segmentation.
        core_instances (np.ndarray): The labeled core instances in the patch.
        core_label (int): The label for cores.
        border_label (int): The label for borders.
        min_distance (float, default=1): The minimum distance for removal.
        min_ratio_threshold (float, default=0.95): The minimum ratio threshold for removal.
        max_distance (float, default=3): The maximum distance for removal.
        max_ratio_threshold (float, default=0.0): The maximum ratio threshold for removal.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: The updated patch after removing small cores, the updated core instances, and the number of cores.
    """

    distances = distance_transform_edt(patch == core_label)
    core_ids = np.unique(core_instances)

    core_ids_to_remove = []
    for core_id in core_ids:
        core_distances = distances[core_instances == core_id]
        num_min_distances = np.count_nonzero(core_distances <= min_distance)
        num_max_distances = np.count_nonzero(core_distances >= max_distance)
        num_core_voxels = np.count_nonzero(core_instances == core_id)
        min_ratio = num_min_distances / num_core_voxels
        max_ratio = num_max_distances / num_core_voxels
        if (min_ratio_threshold is None or min_ratio >= min_ratio_threshold) and (
                max_ratio_threshold is None or max_ratio <= max_ratio_threshold):
            core_ids_to_remove.append(core_id)

    num_cores = len(core_ids) - len(core_ids_to_remove)

    if len(core_ids_to_remove) > 0:
        target_values = np.zeros_like(core_ids_to_remove, dtype=int)
        shape = patch.shape
        core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
        core_instances = core_instances.reshape(shape)

        patch[(patch == core_label) & (core_instances == 0)] = border_label

    return patch, core_instances, num_cores


def remove_small_cores_simple(
        patch: np.ndarray,
        core_instances: np.ndarray,
        core_label: int,
        border_label: int,
        min_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Removes small cores in a patch based on the distance transform of the core label.

    Args:
        patch (np.ndarray): An entire connected component patch of the border-core segmentation.
        core_instances (np.ndarray): The labeled core instances in the patch.
        core_label (int): The label for cores.
        border_label (int): The label for borders.
        min_distance (float, default=1): The minimum distance for removal.
        min_ratio_threshold (float, default=0.95): The minimum ratio threshold for removal.
        max_distance (float, default=3): The maximum distance for removal.
        max_ratio_threshold (float, default=0.0): The maximum ratio threshold for removal.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: The updated patch after removing small cores, the updated core instances, and the number of cores.
    """

    # distances = distance_transform_edt(patch == core_label)
    core_ids = np.unique(core_instances)

    core_ids_to_remove = []
    for core_id in core_ids:
        # core_distances = distances[core_instances == core_id]
        num_core_voxels = np.count_nonzero(core_instances == core_id)
        if (num_core_voxels < min_size):
            core_ids_to_remove.append(core_id)

    num_cores = len(core_ids) - len(core_ids_to_remove)

    if len(core_ids_to_remove) > 0:
        target_values = np.zeros_like(core_ids_to_remove, dtype=int)
        shape = patch.shape
        core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
        core_instances = core_instances.reshape(shape)

        patch[(patch == core_label) & (core_instances == 0)] = border_label

    return patch, core_instances, num_cores
