#!/bin/bash
#SBATCH --job-name=cortical_cont_post
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x-%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}/Dataset778_ChariteCorticalContinuity"
cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -n "${CORTICAL_ARM:-}" ]]; then
    arm="$CORTICAL_ARM"
    fold="$SLURM_ARRAY_TASK_ID"
else
    arm="$(python -m tools.charite_cortical.continuity_workflow map-task --task-id "$SLURM_ARRAY_TASK_ID" --field arm)"
    fold="$(python -m tools.charite_cortical.continuity_workflow map-task --task-id "$SLURM_ARRAY_TASK_ID" --field fold)"
fi

artifact_dir="$CORTICAL_DAG_RUN_DIR/artifacts/$arm/fold_$fold"
echo "[SLURM] minimax postprocess/evaluation arm=$arm fold=$fold"
python -m tools.charite_cortical.continuity_workflow postprocess-evaluate \
    --dataset "$dataset" \
    --prediction-dir "$artifact_dir/predictions" \
    --output-dir "$artifact_dir/instances" \
    --arm "$arm" \
    --fold "$fold" \
    --output-json "$artifact_dir/metrics.json"
