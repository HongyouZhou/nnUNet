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



try:
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator="project-root", pythonpath=True)
except:
    pass
import copy
import itertools
import timeit

import cc3d
import skfmm
from scipy.ndimage import binary_dilation, binary_erosion, convolve, distance_transform_edt, gaussian_filter, label
from scipy.spatial import distance
from skimage.measure import regionprops
from skimage.morphology import ball, binary_dilation
from tools.PENGWIN.utils.parallel import tqdmp

from tools.PENGWIN.utils.bounding_boxes import *

_MASK = np.array([1.0, 0.0, -1.0])


#
# def get_fracture_regions(mask, diskradius=6):
#     morphological_element = ball(diskradius // 2)
#
#     gt_groups = [[1, 10], [11, 20], [21, 30]]
#     unique_labels = np.unique(mask).astype(int)
#     # unique_labels = int(unique_labels)
#     groups = []
#     for group in gt_groups:
#         groups.append(
#             [x for x in unique_labels if x in list(range(group[0], group[1]))]
#         )
#     slow_and_stupid_mask_array = np.zeros(mask.shape, dtype=np.uint8)
#     # fracture_indices = np.empty(
#     #     [0, foreground_indices.shape[1]], dtype=foreground_indices.dtype
#     # )
#     for group in groups:
#         if len(group) > 1:
#             # Apply closing to each nucleus to avoid artifacts
#             bbox = get_bbox_from_mask(
#                 np.logical_and(mask >= group[0], mask <= group[-1])
#             )
#             bbox_padded = pad_bbox(bbox, diskradius, array_shape=mask.shape)
#             slicer = bounding_box_to_slice(bbox_padded)
#             submask = mask[slicer]
#             local_slow_and_stupid = np.zeros(submask.shape, dtype=np.uint8)
#
#             for instance_id in group:
#                 instance = submask == instance_id
#                 instance_big = binary_dilation(
#                     instance, morphological_element
#                 )
#                 instance_small = binary_erosion(
#                     instance, morphological_element
#                 )
#                 boundary = instance_small ^ instance_big
#                 local_slow_and_stupid[boundary] += 1
#
#             indices = np.argwhere(
#                 local_slow_and_stupid > 1
#             )  # regions where two instance touch
#             for dim in range(indices.shape[1]):
#                 indices[:, dim] += bbox_padded[dim][0]
#             # slow_and_stupid_mask_array[local_slow_and_stupid > 1] += 1
#             np.put(
#                 slow_and_stupid_mask_array,
#                 np.ravel_multi_index(indices.transpose(), dims=mask.shape),
#                 1,
#             )
#     return slow_and_stupid_mask_array
def get_fracture_regions(mask, diskradius=6, anatomical_ids=False):
    morphological_element = ball(diskradius // 2)

    # unique_labels = int(unique_labels)
    if anatomical_ids:
        groups = [value for key, value in anatomical_ids.items()]
    else:
        gt_groups = [[1, 10], [11, 20], [21, 30]]
        unique_labels = np.unique(mask).astype(int)
        groups = []
        for group in gt_groups:
            groups.append(
                [x for x in unique_labels if x in list(range(group[0], group[1]))]
            )
    slow_and_stupid_mask_array = np.zeros(mask.shape, dtype=np.uint8)
    # fracture_indices = np.empty(
    #     [0, foreground_indices.shape[1]], dtype=foreground_indices.dtype
    # )
    for group in groups:
        if len(group) > 1:
            # Apply closing to each nucleus to avoid artifacts
            bbox = get_bbox_from_mask(np.isin(mask, group))
            bbox_padded = pad_bbox(bbox, diskradius, array_shape=mask.shape)
            slicer = bounding_box_to_slice(bbox_padded)
            submask = mask[slicer]
            local_slow_and_stupid = np.zeros(submask.shape, dtype=np.uint8)

            for instance_id in group:
                instance = submask == instance_id
                instance_big = binary_dilation(
                    instance, morphological_element
                )
                instance_small = binary_erosion(
                    instance, morphological_element
                )
                boundary = instance_small ^ instance_big
                local_slow_and_stupid[boundary] += 1

            indices = np.argwhere(
                local_slow_and_stupid > 1
            )  # regions where two instance touch
            for dim in range(indices.shape[1]):
                indices[:, dim] += bbox_padded[dim][0]
            # slow_and_stupid_mask_array[local_slow_and_stupid > 1] += 1
            np.put(
                slow_and_stupid_mask_array,
                np.ravel_multi_index(indices.transpose(), dims=mask.shape),
                1,
            )
    return slow_and_stupid_mask_array


def quick_remap_instance_labels(anatomical, instances):
    # anatomical = np.ceil(original_instances.astype(np.single) / 10)
    resorted_instances = np.zeros_like(instances, dtype=np.uint16)
    start_id = {1: 1, 2: 11, 3: 21}
    unique_ids = np.unique(instances)
    unique_ids = unique_ids[unique_ids != 0]
    for unique_id in unique_ids:
        # ids = np.where(instances == unique_id)
        ids = np.argmax((instances == unique_id))
        # print(instances.ravel()[ids])
        anatomical_id = anatomical.ravel()[ids]
        # anatomical_id = anatomical[ids[0][0], ids[1][0], ids[2][0]]
        resorted_instances[instances == unique_id] = start_id[anatomical_id]
        start_id[anatomical_id] += 1
    return resorted_instances


def get_anatomical_mappings(anatomical, instances):
    anatomical_ids = {1: [], 2: [], 3: []}
    unique_ids = np.unique(instances)
    unique_ids = unique_ids[unique_ids != 0]
    for unique_id in unique_ids:
        id = np.argmax(instances == unique_id)
        anatomical_id = anatomical.ravel()[id]
        anatomical_ids[anatomical_id].append(int(unique_id))
    anatomical_ids_inverted_list = [{int(v): key} for key, value in anatomical_ids.items() for v in value]
    anatomical_ids_inverted = {}
    for aid in anatomical_ids_inverted_list:
        anatomical_ids_inverted.update(aid)

    return anatomical_ids, anatomical_ids_inverted


def _remap_instance_labels(pred_semantic, pred_instance, min_size=4000):
    pred_remapped_instance = pred_semantic * 100 * (pred_instance > 0) + pred_instance * (
                pred_semantic > 0)  # do all the splitting in one step. if intance and anatomical segmentatinos do not line up, the get split and merged again according to the anatomical setting!
    pred_remapped_instance = cc3d.connected_components(pred_remapped_instance)
    anatomical_ids, anatomical_ids_inverted = get_anatomical_mappings(pred_semantic, pred_remapped_instance)

    # now lets merge all the fuckers that are too small!
    new_instance_labels, new_instance_counts = np.unique(pred_remapped_instance, return_counts=True)
    new_instance_counts = new_instance_counts[new_instance_labels > 0]
    new_instance_labels = new_instance_labels[new_instance_labels > 0]

    mergers = new_instance_labels[new_instance_counts <= min_size]

    morphological_element = ball(3)
    morphological_element_big = ball(10)

    for merger in mergers:
        if np.sum(pred_remapped_instance == merger) > min_size:  # might be larger due to previous merger
            continue

        bbox = get_bbox_from_mask(pred_remapped_instance == merger)
        bbox_padded = pad_bbox(bbox, 15, array_shape=pred_remapped_instance.shape)
        slicer = bounding_box_to_slice(bbox_padded)
        sub_pred_remapped_instance = pred_remapped_instance[slicer]

        dilated_small_region = binary_dilation(sub_pred_remapped_instance == merger, morphological_element)

        # Find touching labels
        same_anatomical_region = anatomical_ids[anatomical_ids_inverted[merger]]
        touching_labels, touching_counts = np.unique(
            sub_pred_remapped_instance[np.logical_and(dilated_small_region, (sub_pred_remapped_instance != merger))],
            return_counts=True)
        same_anatomical_region_mask = [x in same_anatomical_region for x in touching_labels]
        touching_counts = touching_counts[same_anatomical_region_mask]
        touching_labels = touching_labels[same_anatomical_region_mask]

        if len(touching_labels) > 0:
            pred_remapped_instance[slicer][sub_pred_remapped_instance == merger] = touching_labels[
                np.argmax(touching_counts)]
        else:
            dilated_small_region = binary_dilation(sub_pred_remapped_instance == merger, morphological_element_big)

            # Find touching labels
            same_anatomical_region = anatomical_ids[anatomical_ids_inverted[merger]]
            touching_labels, touching_counts = np.unique(
                sub_pred_remapped_instance[
                    np.logical_and(dilated_small_region, (sub_pred_remapped_instance != merger))],
                return_counts=True)
            same_anatomical_region_mask = [x in same_anatomical_region for x in touching_labels]
            touching_counts = touching_counts[same_anatomical_region_mask]
            touching_labels = touching_labels[same_anatomical_region_mask]

            if len(touching_labels) > 0:
                pred_remapped_instance[slicer][sub_pred_remapped_instance == merger] = touching_labels[
                    np.argmax(touching_counts)]

    anatomical_ids, anatomical_ids_inverted = get_anatomical_mappings(pred_semantic, pred_remapped_instance)
    pred_remapped_instance_final = np.zeros_like(pred_remapped_instance)
    semantic_label_mapping = {0: 0, 1: 1, 2: 11, 3: 21}
    for id, anatomical_id in anatomical_ids_inverted.items():
        pred_remapped_instance_final[pred_remapped_instance == id] = semantic_label_mapping[anatomical_id]
        semantic_label_mapping[anatomical_id] += 1

    return pred_remapped_instance_final


def _abbc_compute_labels(instances, divergence_threshold, distance_threshold, fracture_diskradius):
    instance_labels = np.unique(instances)
    instance_labels = instance_labels[instance_labels > 0]

    abbc_labels = np.zeros_like(instances, dtype=np.float32)
    for instance_label in instance_labels:
        # object_mask = instances == instance_label
        bbox = get_bbox_from_mask(instances == instance_label)
        bbox_padded = pad_bbox(bbox, 20, array_shape=instances.shape)
        slicer = bounding_box_to_slice(bbox_padded)
        object_mask = instances[slicer] == instance_label

        distance = distance_transform_edt(object_mask)
        distance = gaussian_filter(distance, sigma=2)
        gradients = []
        for dim in range(1, instances.ndim):
            gradients.append(np.gradient(np.gradient(distance, axis=dim), axis=dim))
        div = np.sum(np.stack(gradients, axis=0), axis=0)

        inside_mask = -div > divergence_threshold
        inside_mask[distance > distance_threshold] = True

        abbc_labels[slicer][object_mask] = 1
        abbc_labels[slicer][inside_mask] = 2

    fractures = get_fracture_regions(instances, fracture_diskradius + 2)
    abbc_labels[(fractures > 0) & (abbc_labels == 2)] = 1
    fractures = get_fracture_regions(instances, fracture_diskradius)
    abbc_labels[fractures > 0] = 3
    return abbc_labels


def _heal_by_splitting(instances, network_embedding, divergence_threshold, distance_threshold,
                       fracture_diskradius):
    next_label = np.max(instances) + 1

    embedding_from_instance = _abbc_compute_labels(instances, divergence_threshold, distance_threshold,
                                                   fracture_diskradius)

    # let's first split everything that should be broken!
    regions = regionprops(instances)
    for region in regions:
        i_start, j_start, k_start, i_end, j_end, k_end = region["bbox"]
        instance_label = region["label"]
        test = np.logical_and(network_embedding[i_start:i_end, j_start:j_end, k_start:k_end] == 3,
                              embedding_from_instance[i_start:i_end, j_start:j_end, k_start:k_end] != 3)
        # negative values: a fracture is missing
        # positive values: there should be no fracture!

        test_binary = binary_erosion(test != 0, ball(1))  # an error of 1vx is just fine ;)
        test[~test_binary] = 0

        error = np.sum(
            np.abs(test[instances[i_start:i_end, j_start:j_end, k_start:k_end] == instance_label])).astype(
            float)
        error_region = np.sum(network_embedding[i_start:i_end, j_start:j_end, k_start:k_end] == 3).astype(float)
        # print(error_region)
        relative_error = error / (error_region + 1e-8)
        # print(relative_error)
        if relative_error > 0.1 or error > 100:  # 20% of the fracture is prediced wrong. should we check if there actually are alraedy two regions?
            # print(name, relative_error)
            # now lets grow the fractures in this region and see when they split
            cut_fracture = network_embedding[i_start:i_end, j_start:j_end, k_start:k_end] == 3
            cut_network_embedding = network_embedding[i_start:i_end, j_start:j_end, k_start:k_end]
            cut_labels = instances[i_start:i_end, j_start:j_end, k_start:k_end] == instance_label
            not_jet_split = True
            secure_while = 0
            # np.savez('/home/o340n/projects/2024_pengwin_challenge/data/abbc_div_0.11_dist6_frac_6/healing/broken.npz',
            #          cut_fracture)
            # bla = np.zeros_like(original_instances)
            while not_jet_split and secure_while < 10:  # lets go crazy!
                cut_fracture = binary_dilation(cut_fracture, ball(3))
                cut_labels[cut_fracture] = 0
                cut_labels_seg = cc3d.connected_components(cut_labels > 0)
                cut_labels_seg = cut_labels_seg.astype(int)
                cut_regions = regionprops(cut_labels_seg)

                # bla[i_start:i_end, j_start:j_end, k_start:k_end] = cut_labels_seg
                # np.savez(
                #     '/home/o340n/projects/2024_pengwin_challenge/data/abbc_div_0.11_dist6_frac_6/healing/bla_grown' + str(
                #         secure_while) + '.npz',
                #     bla)
                if len(cut_regions) > 1:
                    # print(np.sort([x.area for x in cut_regions])[-2])
                    # 100 over here is much worse than 500. maybe do some additional tuning! tomorrow
                    if np.sort([x.area for x in cut_regions])[-2] > 400:
                        not_jet_split = False  # we split it!
                secure_while += 1
                # print(secure_while)
            # ok, now we need to repair the damage we did
            if not not_jet_split:  # we split it!
                # print('split that fucker!')
                cut_instances = copy.deepcopy(cut_labels_seg)

                instancesids, instance_counts = np.unique(cut_instances, return_counts=True)
                instance_counts = instance_counts[instancesids != 0]
                instancesids = instancesids[instancesids != 0]
                instancesids = instancesids[instance_counts > 100]

                mask = instances[i_start:i_end, j_start:j_end, k_start:k_end] != instance_label
                minimum_distance = np.zeros_like(cut_fracture, dtype=np.single) + np.inf

                for instanceid in instancesids:
                    border = np.ones_like(cut_instances, dtype=bool)
                    border[cut_labels_seg == instanceid] = False

                    phi = np.ma.MaskedArray(border, mask)
                    dist = skfmm.distance(phi)
                    boringdist = dist.data
                    boringdist[dist.mask] = np.inf
                    cut_instances[np.logical_and(boringdist < minimum_distance, cut_labels_seg == 0)] = instanceid
                    minimum_distance = np.minimum(dist, minimum_distance)
                # np.savez(
                #     '/home/o340n/projects/2024_pengwin_challenge/data/abbc_div_0.11_dist6_frac_6/healing/cuthealed' + str(region["label"]) + '.npz',
                #     cut_instances)
                instances[i_start:i_end, j_start:j_end, k_start:k_end][cut_instances > 0] = next_label + \
                                                                                            cut_instances[
                                                                                                cut_instances > 0]  # should allow for 20 individual labels per initial label
                next_label = np.max(instances[i_start:i_end, j_start:j_end, k_start:k_end][cut_instances > 0])
            # else:
            # print(str(name) + ' ' + str(instance_label) + ' could not be fixed...')
    return instances


def _heal_by_splitting_quick(instances, network_embedding, divergence_threshold, distance_threshold,
                             fracture_diskradius):
    next_label = np.max(instances) + 1

    # embedding_from_instance = _abbc_compute_labels(instances, divergence_threshold, distance_threshold,
    #                                                fracture_diskradius)
    # embedding_from_instance = get_fracture_regions(instances, fracture_diskradius).astype(float) * 3

    # let's first split everything that should be broken!
    start = timeit.default_timer()
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(instances)["bounding_boxes"])}
    del props[0]

    network_embedding_patches = []
    embedding_from_instance_patches = []
    instances_patches = []
    labels_patches = []
    true_bbox = []
    for index, (label, bbox) in enumerate(props.items()):
        if bbox[0].start == 65535:  # fuckin empty slices
            continue
        # print(label, bbox)
        true_bbox.append(bbox)

        filter_mask = instances[bbox] == label
        instance_patch = copy.deepcopy(instances[bbox])
        # instance_patch[filter_mask != 1] = 0
        instances_patches.append(instance_patch)

        network_embedding_patch = copy.deepcopy(network_embedding[bbox])
        network_embedding_patch[filter_mask != 1] = 0
        network_embedding_patches.append(network_embedding_patch)

        # embedding_from_instance_patch = copy.deepcopy(embedding_from_instance[bbox])
        # embedding_from_instance_patch[filter_mask != 1] = 0
        # embedding_from_instance_patches.append(embedding_from_instance_patch)

        labels_patches.append([label])

    instances_patches_new = tqdmp(_heal_by_splitting_quick_sub,
                                  (instances_patches, network_embedding_patches, labels_patches), 6,
                                  desc="_heal_by_splitting_quick_sub", fracture_diskradius=fracture_diskradius,
                                  disable=False, mult_iter=True)
    instances_new = np.zeros_like(instances)
    next_label = 1
    real_id = 0
    # for index, bbox in enumerate(true_bbox):
    for index, (label, bbox) in enumerate(props.items()):
        if bbox[0].start == 65535:  # fuckin empty slices
            continue
        instances_patch = instances_patches_new[real_id].astype(np.uint16)
        filter_mask = instances[bbox] == label
        instance_patch_maxid = np.max(instances_patch)
        # for instance_patch_id in range(1, instance_patch_maxid + 1):
        instances_new[bbox][filter_mask] = instances_patch[filter_mask] + next_label
        next_label += instance_patch_maxid
        real_id += 1
    unique_values = np.unique(instances_new)
    unique_values = unique_values[1:]  # no background
    for new, old in enumerate(unique_values):
        instances_new[instances_new == old] = new + 1

    stop = timeit.default_timer()
    print('stupiduniqueloop: ', stop - start)
    return instances_new


def _heal_by_splitting_quick_sub(instances_patch, network_embedding_patch, label,
                                 fracture_diskradius=6):
    embedding_from_instance_patch = get_fracture_regions(instances_patch, fracture_diskradius).astype(float) * 3
    instance_label = label[0]

    next_label = np.max(instances_patch) + 1

    test = np.logical_and(network_embedding_patch == 3,
                          embedding_from_instance_patch != 3)
    # negative values: a fracture is missing
    # positive values: there should be no fracture!

    test_binary = binary_erosion(test != 0, ball(1))  # an error of 1vx is just fine ;)
    test[~test_binary] = 0

    error = np.sum(
        np.abs(test[instances_patch == instance_label])).astype(
        float)
    error_region = np.sum(network_embedding_patch == 3).astype(float)
    # print(error_region)
    relative_error = error / (error_region + 1e-8)
    # print(relative_error)
    if relative_error > 0.1 or error > 100:  # 20% of the fracture is prediced wrong. should we check if there actually are alraedy two regions?
        # print(name, relative_error)
        # now lets grow the fractures in this region and see when they split
        cut_fracture = network_embedding_patch == 3
        # cut_network_embedding = network_embedding_patch
        cut_labels = instances_patch == instance_label
        not_jet_split = True
        secure_while = 0
        # np.savez('/home/o340n/projects/2024_pengwin_challenge/data/abbc_div_0.11_dist6_frac_6/healing/broken.npz',
        #          cut_fracture)
        # bla = np.zeros_like(original_instances)
        while not_jet_split and secure_while < 10:  # lets go crazy!
            cut_fracture = binary_dilation(cut_fracture, ball(3))
            cut_labels[cut_fracture] = 0
            cut_labels_seg = cc3d.connected_components(cut_labels > 0)
            cut_labels_seg = cut_labels_seg.astype(int)
            cut_regions = regionprops(cut_labels_seg)

            # bla[i_start:i_end, j_start:j_end, k_start:k_end] = cut_labels_seg
            # np.savez(
            #     '/home/o340n/projects/2024_pengwin_challenge/data/abbc_div_0.11_dist6_frac_6/healing/bla_grown' + str(
            #         secure_while) + '.npz',
            #     bla)
            if len(cut_regions) > 1:
                # print(np.sort([x.area for x in cut_regions])[-2])
                # 100 over here is much worse than 500. maybe do some additional tuning! tomorrow
                if np.sort([x.area for x in cut_regions])[-2] > 400:  # 400
                    not_jet_split = False  # we split it!
            secure_while += 1
            # print(secure_while)
        # ok, now we need to repair the damage we did
        if not not_jet_split:  # we split it!
            print('split that fucker!')
            print(instance_label)
            cut_instances = copy.deepcopy(cut_labels_seg)

            instancesids, instance_counts = np.unique(cut_instances, return_counts=True)
            instance_counts = instance_counts[instancesids != 0]
            instancesids = instancesids[instancesids != 0]
            instancesids = instancesids[instance_counts > 100]

            mask = instances_patch != instance_label
            minimum_distance = np.zeros_like(cut_fracture, dtype=np.single) + np.inf

            for instanceid in instancesids:
                border = np.ones_like(cut_instances, dtype=bool)
                border[cut_labels_seg == instanceid] = False

                phi = np.ma.MaskedArray(border, mask)
                dist = skfmm.distance(phi)
                boringdist = dist.data
                boringdist[dist.mask] = np.inf
                cut_instances[np.logical_and(boringdist < minimum_distance, cut_labels_seg == 0)] = instanceid
                minimum_distance = np.minimum(dist, minimum_distance)
                instances_patch[cut_instances > 0] = next_label + cut_instances[
                    cut_instances > 0]  # should allow for 20 individual labels per initial label
                next_label = np.max(instances_patch[cut_instances > 0])
        # else:
        # print(str(name) + ' ' + str(instance_label) + ' could not be fixed...')
    return instances_patch


def _heal_by_merging2(instances, network_embedding, anatomical_ids, divergence_threshold, distance_threshold,
                      fracture_diskradius, filter=None, depth=0):
    print("depth: " + str(depth))
    # I added a max depth to make sure it does not run for too long. was not needed in the end
    # if depth > 2:
    #     return instances
    instance_ids, instance_counts = np.unique(instances, return_counts=True)
    instance_counts = instance_counts[instance_ids > 0]
    instance_ids = instance_ids[instance_ids > 0]

    instance_ids = instance_ids[instance_counts > 100]
    if filter is not None:
        instance_ids = [i for i in instance_ids if i in filter]

    combinations = list(itertools.combinations(instance_ids, 2))
    combined_instances = []

    anatomical_ids_inverted_list = [{int(v): key} for key, value in anatomical_ids.items() for v in value]
    anatomical_ids_inverted = {}
    for aid in anatomical_ids_inverted_list:
        anatomical_ids_inverted.update(aid)

    for instance_id1, instance_id2 in combinations:
        if anatomical_ids_inverted[instance_id1] != anatomical_ids_inverted[instance_id2]:
            continue  # if they do not belong to the same anatomical instance no need to merge them (also, there should not be any fracture line between them.
        bbox1 = get_bbox_from_mask(instances == instance_id1)
        bbox1_padded = pad_bbox(bbox1, 20, array_shape=instances.shape)

        bbox2 = get_bbox_from_mask(instances == instance_id2)
        bbox2_padded = pad_bbox(bbox2, 20, array_shape=instances.shape)

        bbox_padded = [[max(b1[0], b2[0]), min(b1[1], b2[1])] for b1, b2 in
                       zip(bbox1_padded, bbox2_padded)]  # intersect the bounding box

        if np.any([b[1] < b[0] for b in bbox_padded]):  # is there any overlap?
            continue
        slicer = bounding_box_to_slice(bbox_padded)
        instance_sliced = instances[slicer].copy()
        instance_sliced[~np.logical_or(instance_sliced == instance_id1, instance_sliced == instance_id2)] = 0
        fractures_region = get_fracture_regions(instance_sliced, fracture_diskradius,
                                                anatomical_ids={1: [instance_id1, instance_id2]})
        if np.sum(fractures_region) == 0:
            continue

        bbox_fracture = get_bbox_from_mask(fractures_region == 1)
        bboxfracture_padded = pad_bbox(bbox_fracture, 20, array_shape=instance_sliced.shape)
        slicer_fracture = bounding_box_to_slice(bboxfracture_padded)

        instance_fracture_sliced = instances[slicer][slicer_fracture].copy()

        # lets make a final check if they touch
        dil1 = binary_dilation(instance_fracture_sliced == instance_id1, ball(2))
        dil2 = binary_dilation(instance_fracture_sliced == instance_id2, ball(2))
        if np.sum(np.logical_and(dil1, dil2)) == 0:
            continue

        instance_fracture_sliced[
            ~np.logical_or(instance_fracture_sliced == instance_id1, instance_fracture_sliced == instance_id2)] = 0

        split_abbc = get_fracture_regions(instance_fracture_sliced, fracture_diskradius,
                                          anatomical_ids=anatomical_ids).astype(int) * 3

        merged_instances = instance_fracture_sliced.copy()
        merged_instances[merged_instances == instance_id2] = instance_id1
        merged_abbc = get_fracture_regions(merged_instances, fracture_diskradius, anatomical_ids=anatomical_ids).astype(
            int) * 3

        split_score = 0
        merged_score = 0
        for classlabel in [3]:
            tmp_split = distance.dice(split_abbc.ravel() == classlabel,
                                      network_embedding[slicer][slicer_fracture].ravel() == classlabel) * (
                            3 if classlabel == 3 else 1)
            tmp_merged = distance.dice(merged_abbc.ravel() == classlabel,
                                       network_embedding[slicer][slicer_fracture].ravel() == classlabel) * (
                             3 if classlabel == 3 else 1)
            split_score += tmp_split if ~np.isnan(tmp_split) else 0
            merged_score += tmp_merged if ~np.isnan(tmp_merged) else 0
        # print(instance_id1, instance_id2, merged_score, split_score)
        if (merged_score < split_score - 0.05) or (  # should be kind of decisive
                merged_score == 3.0 and split_score == 3.0):  # if both are 3.0 than there is not split but also a wrong split close by
            combined_instances.append(
                [int(instance_id1), int(instance_id2), split_score, merged_score, merged_score - split_score])

    if len(combined_instances) > 0:
        groups = []
        for combined_instance in combined_instances:
            tmp = [i for i, x in enumerate(groups) if combined_instance[0] in x[:2]]
            if len(tmp) > 0:
                groups[tmp[0]].append(combined_instance[1])
                continue
            tmp = [i for i, x in enumerate(groups) if combined_instance[1] in x[:2]]
            if len(tmp) > 0:
                groups[tmp[0]].append(combined_instance[0])
                continue
            groups.append(combined_instance[:2])
        for group in groups:
            group = np.unique(group)
            if len(group) == 2:
                instances[instances == group[1]] = group[0]
            if len(group) > 2:
                possible_combinations = list(itertools.combinations(group, 2))
                existing_combinations = []
                for combination in possible_combinations:
                    existing_combinations.extend([i for i, x in enumerate(combined_instances) if
                                                  x[0] == combination[0] and x[1] == combination[1]])
                score_improvements = [combined_instances[i][4] for i in existing_combinations]
                best_merger = np.argmin(score_improvements)
                instances[instances == combined_instances[best_merger][1]] = combined_instances[best_merger][0]
                instances = _heal_by_merging2(instances, network_embedding, anatomical_ids,
                                              divergence_threshold,
                                              distance_threshold,
                                              fracture_diskradius,
                                              filter=group,
                                              depth=depth + 1)  # check if there is more to merge...

    return instances


def _heal_by_merging(instances, network_embedding, divergence_threshold, distance_threshold,
                     fracture_diskradius, filter=None):
    # next_label = np.max(instances) + 1

    # embedding_from_instance = _abbc_compute_labels(instances, divergence_threshold, distance_threshold,
    #                                                fracture_diskradius)
    instance_ids, instance_counts = np.unique(instances, return_counts=True)
    instance_counts = instance_counts[instance_ids > 0]
    instance_ids = instance_ids[instance_ids > 0]

    instance_ids = instance_ids[instance_counts > 500]
    if filter is not None:
        instance_ids = [i for i in instance_ids if i in filter]
    # instance_counts = instance_counts[instance_counts > 500]

    combinations = list(itertools.combinations(instance_ids, 2))
    combined_instances = []
    for instance_id1, instance_id2 in combinations:
        if np.ceil(instance_id1 / 10) != np.ceil(instance_id2 / 10):
            continue  # if they do not belong to the same anatomical instance no need to merge them (also, there should not be any fracture line between them.
        bbox1 = get_bbox_from_mask(instances == instance_id1)
        bbox1_padded = pad_bbox(bbox1, 20, array_shape=instances.shape)

        bbox2 = get_bbox_from_mask(instances == instance_id2)
        bbox2_padded = pad_bbox(bbox2, 20, array_shape=instances.shape)

        bbox_padded = [[max(b1[0], b2[0]), min(b1[1], b2[1])] for b1, b2 in
                       zip(bbox1_padded, bbox2_padded)]  # intersect the bounding box

        if np.any([b[1] < b[0] for b in bbox_padded]):  # is there any overlap?
            continue
        slicer = bounding_box_to_slice(bbox_padded)
        instance_sliced = instances[slicer].copy()
        # instance_sliced[~np.logical_or(instance_sliced == instance_id1, instance_sliced == instance_id2)] = 0
        fractures_region = get_fracture_regions(instance_sliced, fracture_diskradius)
        if np.sum(fractures_region) == 0:
            continue

        bbox_fracture = get_bbox_from_mask(fractures_region == 1)
        bboxfracture_padded = pad_bbox(bbox_fracture, 20, array_shape=instance_sliced.shape)
        slicer_fracture = bounding_box_to_slice(bboxfracture_padded)
        # TODO: find a small region around the interface. no need to compute everything.

        instance_fracture_sliced = instances[slicer].copy()[slicer_fracture]
        # instance_fracture_sliced[
        #     ~np.logical_or(instance_fracture_sliced == instance_id1, instance_fracture_sliced == instance_id2)] = 0

        split_abbc = _abbc_compute_labels(instance_fracture_sliced, divergence_threshold, distance_threshold,
                                          fracture_diskradius)
        merged_instances = instance_fracture_sliced.copy()
        merged_instances[merged_instances == instance_id2] = instance_id1
        merged_abbc = _abbc_compute_labels(merged_instances, divergence_threshold, distance_threshold,
                                           fracture_diskradius)

        split_score = 0
        merged_score = 0
        for classlabel in range(1, 4):
            split_score += distance.dice(split_abbc.ravel() == classlabel,
                                         network_embedding[slicer][slicer_fracture].ravel() == classlabel) * (
                               3 if classlabel == 3 else 1)
            merged_score += distance.dice(merged_abbc.ravel() == classlabel,
                                          network_embedding[slicer][slicer_fracture].ravel() == classlabel) * (
                                3 if classlabel == 3 else 1)
        # print(merged_score, split_score)
        if merged_score < split_score - 0.1:  # there should be some reasonable improvement
            combined_instances.append(
                [int(instance_id1), int(instance_id2), split_score, merged_score, merged_score - split_score])

    if len(combined_instances) > 0:
        groups = []
        for combined_instance in combined_instances:
            tmp = [i for i, x in enumerate(groups) if combined_instance[0] in x[:2]]
            if len(tmp) > 0:
                groups[tmp[0]].append(combined_instance[1])
                continue
            tmp = [i for i, x in enumerate(groups) if combined_instance[1] in x[:2]]
            if len(tmp) > 0:
                groups[tmp[0]].append(combined_instance[0])
                continue
            groups.append(combined_instance[:2])
        for group in groups:
            group = np.unique(group)
            if len(group) == 2:
                instances[instances == group[1]] = group[0]
            if len(group) > 2:
                possible_combinations = list(itertools.combinations(group, 2))
                existing_combinations = []
                for combination in possible_combinations:
                    existing_combinations.extend([i for i, x in enumerate(combined_instances) if
                                                  x[0] == combination[0] and x[1] == combination[1]])
                score_improvements = [combined_instances[i][4] for i in existing_combinations]
                best_merger = np.argmin(score_improvements)

                instances[instances == combined_instances[best_merger][1]] = combined_instances[best_merger][0]
                instances = _heal_by_merging(instances, network_embedding, divergence_threshold,
                                             distance_threshold,
                                             fracture_diskradius, filter=group)  # check if there is more to merge...

    return instances
