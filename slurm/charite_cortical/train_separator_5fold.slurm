#!/bin/bash
#SBATCH --job-name=cortical_sep
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-fold%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
plans_name="nnUNetResEncUNetMPlansSeparator"
cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

plans_file="$nnUNet_preprocessed/Dataset778_ChariteCorticalContinuity/${plans_name}.json"
test -f "$plans_file"
test -f "$nnUNet_preprocessed/Dataset778_ChariteCorticalContinuity/splits_final.json"

python nnunetv2/run/run_training.py \
    778 \
    3d_fullres \
    "$SLURM_ARRAY_TASK_ID" \
    -tr nnUNetTrainer \
    -p "$plans_name" \
    -num_gpus 1 \
    --c
