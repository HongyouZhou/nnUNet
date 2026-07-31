#!/bin/bash
#SBATCH --job-name=cortical_prep
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
source_dataset="${CORTICAL_SOURCE_DATASET:-$PROJECT_HOME/dev/data/segmentation/derived/cortical_dataset_68_final_20260728}"
dataset_dir="$nnUNet_raw/Dataset778_ChariteCorticalContinuity"
separator_plans="nnUNetResEncUNetMPlansSeparator"
continuity_plans="nnUNetResEncUNetMPlansContinuity"

cd "$repo_dir"
mkdir -p logs "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
export PYTHONPATH="$repo_dir"

if [[ -d "$dataset_dir" ]]; then
    python -m tools.charite_cortical.continuity_data.audit_dataset "$dataset_dir"
else
    python -m tools.charite_cortical.continuity_cli build \
        --source "$source_dataset" \
        --output "$dataset_dir" \
        --execute \
        --distance-backend scipy \
        --image-mode copy
fi

nnUNetv2_extract_fingerprint \
    -d 778 \
    -np "$SLURM_CPUS_PER_TASK" \
    --verify_dataset_integrity \
    --no_pbar

nnUNetv2_plan_experiment \
    -d 778 \
    -pl nnUNetPlannerResEncM \
    -preprocessor_name DefaultPreprocessor \
    -overwrite_target_spacing 0.5 0.5 0.5 \
    -overwrite_plans_name "$separator_plans"

nnUNetv2_plan_experiment \
    -d 778 \
    -pl nnUNetPlannerResEncM \
    -preprocessor_name ChariteCorticalPreprocessor \
    -overwrite_target_spacing 0.5 0.5 0.5 \
    -overwrite_plans_name "$continuity_plans"

python -m nnunetv2.training.cortical_continuity.configure_plans \
    --plans "$nnUNet_preprocessed/Dataset778_ChariteCorticalContinuity/${continuity_plans}.json" \
    --configuration 3d_fullres \
    --direction-set axial19

nnUNetv2_preprocess \
    -d 778 \
    -plans_name "$separator_plans" \
    -c 3d_fullres \
    -np "$SLURM_CPUS_PER_TASK" \
    --no_pbar

nnUNetv2_preprocess \
    -d 778 \
    -plans_name "$continuity_plans" \
    -c 3d_fullres \
    -np "$SLURM_CPUS_PER_TASK" \
    --no_pbar

python -m tools.charite_cortical.continuity_data.audit_dataset "$dataset_dir"
