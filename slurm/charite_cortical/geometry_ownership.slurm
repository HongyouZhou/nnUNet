#!/bin/bash
#SBATCH --job-name=cortical_geometry
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
: "${CORTICAL_DIAGNOSTIC_DIR:?Set CORTICAL_DIAGNOSTIC_DIR}"
: "${CORTICAL_GEOMETRY_DIR:?Set CORTICAL_GEOMETRY_DIR}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m tools.charite_cortical.run_geometry_ownership \
    --dataset "$PROJECT_HOME/dev/data/nnUNet_raw/Dataset778_ChariteCorticalContinuity" \
    --run-dir "$CORTICAL_DAG_RUN_DIR" --diagnosis "$CORTICAL_DIAGNOSTIC_DIR" \
    --output "$CORTICAL_GEOMETRY_DIR" --task-id "$SLURM_ARRAY_TASK_ID"
