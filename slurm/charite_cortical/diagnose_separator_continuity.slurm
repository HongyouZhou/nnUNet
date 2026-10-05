#!/bin/bash
#SBATCH --job-name=cortical_cont_diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/%x-%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
: "${CORTICAL_DIAGNOSTIC_DIR:?Set CORTICAL_DIAGNOSTIC_DIR}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m tools.charite_cortical.diagnose_continuity \
    --dataset "$PROJECT_HOME/dev/data/nnUNet_raw/Dataset778_ChariteCorticalContinuity" \
    --run-dir "$CORTICAL_DAG_RUN_DIR" --output "$CORTICAL_DIAGNOSTIC_DIR" \
    --task-id "$SLURM_ARRAY_TASK_ID"
