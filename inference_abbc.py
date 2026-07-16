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
import os
import shutil
import timeit
from datetime import datetime
from os.path import join
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import SimpleITK
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy.spatial import distance

from tools.PENGWIN.abbc_conversion.abbc2instance import abbc2instance, abbc_refine_border, add_mask_to_closest_instance
from tools.PENGWIN.postprocess.postprocess_abbc import (
    _heal_by_merging2,
    _heal_by_splitting,
    _heal_by_splitting_quick,
    _remap_instance_labels,
    get_anatomical_mappings,
    quick_remap_instance_labels,
)
from tools.PENGWIN.utils.utils import MedVol, load_filenames


def _dice_for_label(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """
    返回指定标签的 Dice；若 pred、gt 均无该标签，则返回 1.0
    """
    pred_mask = pred == label
    gt_mask = gt == label
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0
    # 也可以用 scipy.spatial.distance.dice
    return 1 - distance.dice(pred_mask.ravel(), gt_mask.ravel())


def _mean_dice(pred: np.ndarray, gt: np.ndarray, ignore_label: int = 0) -> float:
    """
    计算 pred vs gt 的 mean Dice（macro-average，去掉 background）
    """
    labels = np.union1d(np.unique(pred), np.unique(gt))
    labels = labels[labels != ignore_label]
    dices = [_dice_for_label(pred, gt, lb) for lb in labels]
    return float(np.mean(dices))


def _mean_iou(pred: np.ndarray, gt: np.ndarray, ignore_label: int = 0) -> float:
    """
    mean IoU (= mIoU, macro)；若想用请把下面 main loop 里的 dice 换成 iou
    """
    labels = np.union1d(np.unique(pred), np.unique(gt))
    labels = labels[labels != ignore_label]
    ious = []
    for lb in labels:
        inter = np.logical_and(pred == lb, gt == lb).sum()
        union = np.logical_or(pred == lb, gt == lb).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / union)
    return float(np.mean(ious))


def inference_abbc_instance(load_dir, save_dir, instance_model_dir, fold_instance, gt_dir=None):
    """
    推理入口（仅实例分割版本）。

    Args:
        load_dir           (str): 待预测 *.mha 影像所在目录。
        save_dir           (str): 结果保存根目录，会在其中创建
                                  images/。
        instance_model_dir (str): 训练好的 abbc 实例模型目录。
        fold_instance      (tuple): 需要使用的 fold，例如 (0,) 或 ("all",)。
    """
    load_dir = os.path.expandvars(load_dir)
    save_dir = os.path.expandvars(save_dir)
    instance_model_dir = os.path.expandvars(instance_model_dir)
    processes = 8

    # 输出目录
    Path(join(save_dir)).mkdir(parents=True, exist_ok=True)

    # -------- 统计用 --------
    dice_all, iou_all = [], []      # 用来汇总整体平均

    print("LOAD IMAGE")
    names = load_filenames(load_dir)
    for name in names:
        print(f"########################   {name}    ##############################")
        img, props = custom_reader(join(load_dir, f"{name}.nii.gz"))
        print("IMAGE SHAPE:", img.shape)

        # ------------------------------------------------------------------ #
        # 0. 复制原始 CT 图像到输出目录，方便与预测结果一并查看
        # ------------------------------------------------------------------ #
        ct_out_path = join(
            save_dir,
            "images",
            f"{name}_input.nii.gz",
        )
        Path(os.path.dirname(ct_out_path)).mkdir(parents=True, exist_ok=True)
        if not Path(ct_out_path).is_file():           # 避免重复复制
            shutil.copyfile(join(load_dir, f"{name}.nii.gz"), ct_out_path)

        # ------------------------------------------------------------------ #
        # 1. 预测 (或读取缓存) adaptive-boundary-border-core segmentation
        # ------------------------------------------------------------------ #
        print("PREDICT ABB-C (instance model)")
        predfile = Path(join(instance_model_dir, f"fold_{fold_instance[0]}", "predictions", f"{name}.nii.gz"))
        if predfile.is_file():
            pred_border_core, _ = custom_reader(predfile)
            pred_border_core = pred_border_core[0, ...]  # 去掉通道维
        else:
            predictor = nnUNetPredictor(use_mirroring=False)
            predictor.initialize_from_trained_model_folder(instance_model_dir, use_folds=fold_instance, checkpoint_name="checkpoint_final.pth")
            pred_border_core = predictor.predict_from_list_of_npy_arrays(
                [img], segs_from_prev_stage_or_list_of_segs_from_prev_stage=None, properties_or_list_of_properties=[props], truncated_ofname=None
            )[0]
            # 缓存预测，方便后续调参
            if len(names) > 1:
                Path(predfile.parent).mkdir(parents=True, exist_ok=True)
                custom_writer(pred_border_core, predfile, props, dtype=np.int8)

        # ------------------------------------------------------------------ #
        # 2. 直接保存原始 ABB-C 标签（0: 背景, 1: boundary, 2: core, 3: border）
        # ------------------------------------------------------------------ #
        print("WRITE OUTPUT (raw ABB-C labels)")
        out_path = join(save_dir, "images", f"{name}_pred.nii.gz")
        # 保存为 int8 即可
        custom_writer(pred_border_core.astype(np.int8), out_path, props, dtype=np.int8)

        # ---------------- 评  测 -----------------
        if gt_dir is not None:
            # ------------- 处理 `_0000` 后缀 -----------------
            gt_base = name.replace("_0000", "")            # PENGWIN GT 文件不带通道后缀
            gt_file = Path(gt_dir) / f"{gt_base}.nii.gz"   # 如果 GT 是 .mha 请自行修改
            if not gt_file.exists():
                print(f"[WARN] GT not found: {gt_file}")
            else:
                gt, _ = custom_reader(gt_file)
                gt = gt[0]                                 # 去掉通道

                # 计算各标签 Dice / IoU（忽略 0）
                labels = np.union1d(np.unique(pred_border_core), np.unique(gt))
                labels = labels[labels != 0]
                dices, ious = [], []
                for lb in labels:
                    pred_m, gt_m = pred_border_core == lb, gt == lb
                    inter = np.logical_and(pred_m, gt_m).sum()
                    union = np.logical_or(pred_m,  gt_m).sum()
                    dices.append(1.0 if pred_m.sum()+gt_m.sum()==0 else 2*inter/(pred_m.sum()+gt_m.sum()))
                    ious .append(1.0 if union==0 else inter/union)

                mean_dice, mean_iou = float(np.mean(dices)), float(np.mean(ious))
                dice_all.append(mean_dice)
                iou_all .append(mean_iou)
                print(f"[EVAL] {name}: mean Dice={mean_dice:.4f}  mIoU={mean_iou:.4f}")

    # -------- 输出汇总 --------
    if gt_dir is not None and dice_all:
        print(f"\n====== Overall mean Dice: {np.mean(dice_all):.4f}  |  Overall mIoU: {np.mean(iou_all):.4f} "
              f"on {len(dice_all)} cases ======\n")

    return 0


def inference_abbc(load_dir, save_dir, instance_model_dir, semantic_model_dir, fold_instance, fold_semantic):
    load_dir = os.path.expandvars(load_dir)
    save_dir = os.path.expandvars(save_dir)
    instance_model_dir = os.path.expandvars(instance_model_dir)
    semantic_model_dir = os.path.expandvars(semantic_model_dir)
    processes = 8

    Path(join(save_dir, "images/pelvic-fracture-ct-segmentation")).mkdir(parents=True, exist_ok=True)

    print("LOAD IMAGE")
    names = load_filenames(load_dir)
    for name in names:
        print("########################   " + name + "    ##############################")
        img, props = custom_reader(join(load_dir, f"{name}.mha"))
        print(img.shape)

        # the nnunet predictions will be written to the "predictions" folder inside the respecitve fold -> saves time if you want to tune parameters
        print("PREDICT SEMANTIC")
        predfile = Path(join(semantic_model_dir, "fold_" + str(fold_semantic[0]), "predictions", name + ".mha"))
        if predfile.is_file():
            pred_semantic, _ = custom_reader(predfile)
            pred_semantic = pred_semantic[0, ...]
        else:
            predictor = nnUNetPredictor(use_mirroring=False)
            predictor.initialize_from_trained_model_folder(join(semantic_model_dir), use_folds=fold_semantic, checkpoint_name="checkpoint_final.pth")
            # this inference function includes resampling
            pred_semantic = predictor.predict_from_list_of_npy_arrays(
                [img], segs_from_prev_stage_or_list_of_segs_from_prev_stage=None, properties_or_list_of_properties=[props], truncated_ofname=None
            )[0]
            print(pred_semantic.shape)
            if len(names) > 1:
                Path(predfile.parent).mkdir(parents=True, exist_ok=True)
                custom_writer(pred_semantic, predfile, props, dtype=np.int8)

        print("PREDICT INSTANCES")
        predfile = Path(join(instance_model_dir, "fold_" + str(fold_instance[0]), "predictions", name + ".mha"))
        if predfile.is_file():
            pred_border_core, _ = custom_reader(predfile)
            pred_border_core = pred_border_core[0, ...]
        else:
            predictor = nnUNetPredictor(use_mirroring=False)
            predictor.initialize_from_trained_model_folder(join(instance_model_dir), use_folds=fold_instance, checkpoint_name="checkpoint_final.pth")
            # this inference function includes resampling
            pred_border_core = predictor.predict_from_list_of_npy_arrays(
                [img], segs_from_prev_stage_or_list_of_segs_from_prev_stage=None, properties_or_list_of_properties=[props], truncated_ofname=None
            )[0]
            print(pred_border_core.shape)
            if len(names) > 1:
                Path(predfile.parent).mkdir(parents=True, exist_ok=True)
                custom_writer(pred_border_core, predfile, props, dtype=np.int8)

        print("REMAP LABELS")
        startall = timeit.default_timer()

        min_size = 1000  # minimum size in the end

        # parameters for the abbc embedding, these should match the training data
        # for runtime reasons we actually only use the fracture_diskradius while performing splitting and merging
        divergence_threshold = 0.11
        distance_threshold = 6
        fracture_diskradius = 6

        labels = [2, 1, 3]
        core_label = labels[0]
        boundary_label = labels[1]
        border_label = labels[2]

        start = timeit.default_timer()

        # initial conversion from abbc via cores to instances
        pred_instance, _ = abbc2instance(pred_border_core, core_label=core_label, boundary_label=boundary_label, border_label=border_label)
        stop = timeit.default_timer()
        print("abbc2instance: ", stop - start)
        start = timeit.default_timer()

        instances = _remap_instance_labels(pred_semantic, pred_instance, min_size)
        stop = timeit.default_timer()

        # original_instances = instances.copy()
        pred_border_core[pred_semantic == 0] = 0
        print("_remap_instance_labels: ", stop - start)
        start = timeit.default_timer()
        # why do we split multiple times? sometimes a fracture should be split in three. in most cases this does not happen at the same time while expanding the fractures.
        # it could be improved...

        for _ in range(3):
            instances = _heal_by_splitting_quick(instances, pred_border_core, divergence_threshold, distance_threshold, fracture_diskradius)
        stop = timeit.default_timer()
        print("_heal_by_splitting: ", stop - start)

        start = timeit.default_timer()

        anatomical_ids, anatomical_ids_inverse = get_anatomical_mappings(pred_semantic, instances)

        instances = _heal_by_merging2(instances, pred_border_core, anatomical_ids, divergence_threshold, distance_threshold, fracture_diskradius, depth=0)
        stop = timeit.default_timer()
        print("_heal_by_merging: ", stop - start)
        start = timeit.default_timer()

        anatomical_ids, anatomical_ids_inverted = get_anatomical_mappings(pred_semantic, instances)
        pred_remapped_instance = np.zeros_like(instances)
        semantic_label_mapping = {0: 0, 1: 1, 2: 11, 3: 21}
        for id, anatomical_id in anatomical_ids_inverted.items():
            pred_remapped_instance[instances == id] = semantic_label_mapping[anatomical_id]
            semantic_label_mapping[anatomical_id] += 1
        stop = timeit.default_timer()
        print("_remap_instance_labels: ", stop - start)

        start = timeit.default_timer()

        fill_mask = pred_semantic != 0

        pred_remapped_instance_instances, pred_remapped_instance_counts = np.unique(pred_remapped_instance, return_counts=True)
        remove = pred_remapped_instance_instances[pred_remapped_instance_counts < 1000]  # really good. hardcoded min_size ;)
        pred_remapped_instance[np.isin(pred_remapped_instance, remove)] = 0

        pred_remapped_instance = abbc_refine_border(fill_mask, pred_remapped_instance, processes)

        stop = timeit.default_timer()
        print("add_mask_to_closest_instance: ", stop - start)

        print("WRITE OUTPUT")
        if len(names) > 1:
            custom_writer(pred_remapped_instance, join(save_dir, "images/pelvic-fracture-ct-segmentation", name + ".nii.gz"), props, dtype=np.int8)
        else:
            custom_writer(pred_remapped_instance, join(save_dir, "images/pelvic-fracture-ct-segmentation", "output.nii.gz"), props, dtype=np.int8)
        stop = timeit.default_timer()
        print("all: ", stop - startall)

    # custom_writer(pred_semantic, join(save_dir, "images/pelvic-fracture-ct-segmentation", "output_semantic.mha"), props)
    return 0


def custom_reader(file: str):
    img_sitk = SimpleITK.ReadImage(str(file))
    # Build affine matrix from SimpleITK image metadata
    affine = np.array(img_sitk.GetDirection()).reshape(3, 3)
    affine = np.dot(affine, np.diag(img_sitk.GetSpacing()))
    affine = np.concatenate([affine, np.array(img_sitk.GetOrigin())[:, np.newaxis]], axis=1)
    affine = np.concatenate([affine, np.array([[0, 0, 0, 1]])], axis=0)
    # sitk assumes LPS, so add a transform to RAS
    affine = np.dot(np.diag([-1, -1, 1, 1]), affine)
    # Make a nibabel image. Attention: simpleitk reorders axes!
    img_arr = SimpleITK.GetArrayFromImage(img_sitk).transpose(2, 1, 0)
    img_nib = nib.Nifti1Image(img_arr, affine=affine)
    # Rest is copied from nnunet's nibabelwithreorientio
    reoriented_image = img_nib.as_reoriented(nib.io_orientation(affine))
    reoriented_affine = reoriented_image.affine

    # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
    spacing = [float(i) for i in reoriented_image.header.get_zooms()[::-1]]

    # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
    image = reoriented_image.get_fdata().transpose((2, 1, 0))[None]
    props = {
        "spacing": spacing,
        "nibabel_stuff": {
            "original_affine": affine,
            "reoriented_affine": reoriented_affine,
        },
    }
    return image, props


def custom_writer(seg: np.ndarray, output_fname: str, properties: dict, dtype=np.uint8):
    # similar to nnunet's nibabelwithreorientio
    seg = seg.transpose((2, 1, 0))

    seg_nib = nib.Nifti1Image(seg, affine=properties["nibabel_stuff"]["reoriented_affine"])
    seg_nib_reoriented = seg_nib.as_reoriented(nib.io_orientation(properties["nibabel_stuff"]["original_affine"]))
    if not np.allclose(properties["nibabel_stuff"]["original_affine"], seg_nib_reoriented.affine):
        print(f"WARNING: Restored affine does not match original affine. File: {output_fname}")
        print(f"Original affine\n", properties["nibabel_stuff"]["original_affine"])
        print(f"Restored affine\n", seg_nib_reoriented.affine)

    # Make an sitk image from nibabel image
    img_sitk = SimpleITK.GetImageFromArray(seg_nib_reoriented.get_fdata().transpose((2, 1, 0)).astype(dtype))
    # sitk wants LPS as output orientation, but we have RAS => add transform
    affine_lps = np.dot(np.diag([-1, -1, 1, 1]), seg_nib_reoriented.affine)
    spacing = [float(i) for i in seg_nib_reoriented.header.get_zooms()]
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(affine_lps[:3, 3])
    img_sitk.SetDirection(affine_lps[:3, :3].dot(np.diag([1 / s for s in spacing])).flatten())
    # Write the image
    SimpleITK.WriteImage(img_sitk, output_fname)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--dataset', required=True, type=str, help='')
    # parser.add_argument('-im', '--instance_model', required=True, type=str, help='')
    # parser.add_argument('-sm', '--semantic_model', default='nnUNetResEncUNetLPlans', required=False, type=str, help='')
    # args = parser.parse_args()

    # os.environ["PROJEKTE"] = "/mnt/E132-Projekte"
    # load_dir = "/home/o340n/projects/2024_pengwin_challenge/data/PENGWIN_CT_train_images"
    # gt_load_dir = "$PROJEKTE/Projects/2024_Pengwin_Challenge/border_core/labelsTr_instances"
    instance_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/instance_model/"
    # instance_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/instance_model_D17_base/"
    # instance_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/instance_model_D17_base2k/"
    # instance_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/instance_model_D17_resenc/"
    # instance_model_dir = "$PROJEKTE/Projects/2024_Pengwin_Challenge/border_core/checkpoints/Dataset2001_border_core_v1/nnUNetTrainer__nnUNetPlans__3d_fullres"
    semantic_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/semantic_model/"
    # semantic_model_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/semantic_model_final/"
    # splits = "$PROJEKTE/Projects/2024_Pengwin_Challenge/splits_final.json"
    # save_dir = f"$PROJEKTE/Projects/2024_Pengwin_Challenge/border_core/predsTr_fold0/instance/{args.dataset}/{args.instance_model}"
    fold_instance = ("all",)
    fold_instance = (0,)
    fold_semantic = ("all",)
    fold_semantic = (0,)
    processes = 6

    # gt_load_dir = os.path.expandvars(gt_load_dir)
    # inference_v1(load_dir, save_dir, instance_model_dir, semantic_model_dir, splits, fold, processes)
    load_dir = "/home/o340n/projects/2024_pengwin_challenge/docker/input/images/pelvic-fracture-ct"

    save_dir = os.path.join("/home/o340n/projects/2024_pengwin_challenge/docker/output_sm_0_im_0_oldmin_5_3_min1000_semfgv2dil_bc2i_fmm_mergetwo_min100", Path(instance_model_dir).name)
    # load_dir = "/home/m167k/Desktop/pengwin_testing/input/images/pelvic-fracture-ct"
    # save_dir = "/home/m167k/Desktop/pengwin_testing/output"
    # gt_load_dir = "/home/m167k/Desktop/pengwin_testing/segmentations"
    inference_abbc(load_dir, save_dir, instance_model_dir, semantic_model_dir, fold_instance, fold_semantic)
    print(save_dir)
    # evaluate(join(save_dir, "remapped_instance"), gt_load_dir, save_dir, processes)
