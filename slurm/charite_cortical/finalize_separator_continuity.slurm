#!/bin/bash
#SBATCH --job-name=cortical_cont_final
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
: "${CORTICAL_SELECTED_ARMS:?Set CORTICAL_SELECTED_ARMS}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

inputs=()
IFS=',' read -r -a arms <<< "$CORTICAL_SELECTED_ARMS"
for arm in "${arms[@]}"; do
    inputs+=(--input "$arm=$CORTICAL_DAG_RUN_DIR/metrics/$arm-final.json")
done
python -m tools.charite_cortical.continuity_workflow final-summary \
    "${inputs[@]}" \
    --gates-dir "$CORTICAL_DAG_RUN_DIR/gates" \
    --output "$CORTICAL_DAG_RUN_DIR/final-summary.json"
echo "[SLURM] Complete DAG summary: $CORTICAL_DAG_RUN_DIR/final-summary.json"
