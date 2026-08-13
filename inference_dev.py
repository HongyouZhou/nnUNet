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


from datetime import datetime
from pathlib import Path
import os
import argparse

from inference_abbc import inference_abbc_instance
from segmentation_io import (
    default_output_directory,
    read_laterality,
    resolve_bone_masks_directory,
)

if __name__ == "__main__":
    print("[%s] start inference_docker.py in main" % str(datetime.now()))
    parser = argparse.ArgumentParser(description="Run REPAIR segmentation inference")
    parser.add_argument(
        "-i", "--input", "--input-dir", "--input_dir",
        dest="input_path",
        type=str,
        default=None,
        help="Path to ct.nii.gz or a directory of NIfTI inputs",
    )
    parser.add_argument(
        "-o", "--output-dir", "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Path to output directory",
    )
    parser.add_argument(
        "--bone-masks-dir",
        type=str,
        default=None,
        help="Deployment path containing post-processing bone masks",
    )
    parser.add_argument(
        "--side",
        choices=("L", "R", "left", "right"),
        default=None,
        help="Laterality override; normally read from config.json",
    )
    args = parser.parse_args()

    input_path = Path(os.environ['PROJECT_HOME']) / "dev/data/nnUNet_raw/Dataset989_charite/imagesTr/"
    output_path = Path(os.environ['PROJECT_HOME']) / "dev/data/charite_results/Dataset989_charite/"
    resource_path = Path(
        os.environ.get(
            "RESOURCE_PATH",
            "/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/nnUNet_results/Dataset777_Merged/nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150__nnUNetResEncUNetMPlans__3d_fullres/",
        )
    )

    load_dir = Path(args.input_path) if args.input_path is not None else input_path
    if args.output_dir is not None:
        save_dir = Path(args.output_dir)
    elif args.input_path is not None:
        # Service mode: landmarks/reposition search the original FILE_UPLOAD
        # tree recursively for this exact filename.
        save_dir = default_output_directory(load_dir)
    else:
        save_dir = output_path
    # instance_model_dir = RESOURCE_PATH / "instance_model"
    # semantic_model_dir = RESOURCE_PATH / "semantic_model"
    instance_model_dir = resource_path
    semantic_model_dir = resource_path
    fold_instance = ("all",)
    fold_semantic = ("all",)
    bone_masks_dir = resolve_bone_masks_directory(load_dir, args.bone_masks_dir)
    laterality = read_laterality(load_dir, args.side)

    inference_abbc_instance(
        load_dir,
        save_dir,
        instance_model_dir,
        fold_instance,
        bone_masks_dir,
        laterality,
        gt_dir=None,
        contract_output_dir=(
            default_output_directory(load_dir)
            if args.input_path is not None
            else None
        ),
    )
