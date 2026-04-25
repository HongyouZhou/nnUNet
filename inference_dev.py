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

from inference_abbc import inference_abbc_instance

if __name__ == "__main__":
    print("[%s] start inference_docker.py in main" % str(datetime.now()))
    INPUT_PATH = Path(os.environ['PROJECT_HOME']) / "dev/data/nnUNet_raw/Dataset989_charite/imagesTr/"
    OUTPUT_PATH = Path(os.environ['PROJECT_HOME']) / "dev/data/charite_results/Dataset989_charite/"
    RESOURCE_PATH = Path(os.environ['PROJECT_HOME']) / "dev/data/nnUNet_results/Dataset989_charite/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/"

    load_dir = INPUT_PATH
    save_dir = OUTPUT_PATH
    # instance_model_dir = RESOURCE_PATH / "instance_model"
    # semantic_model_dir = RESOURCE_PATH / "semantic_model"
    instance_model_dir = RESOURCE_PATH
    semantic_model_dir = RESOURCE_PATH
    fold_instance = ("all",)
    fold_semantic = ("all",)

    inference_abbc_instance(
        load_dir,
        save_dir,
        instance_model_dir,
        fold_instance,
        gt_dir=None,
    )
