#!/bin/bash
#SBATCH --job-name=cortical_cont_merge
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
: "${CORTICAL_ARM:?Set CORTICAL_ARM}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
mkdir -p logs "$CORTICAL_DAG_RUN_DIR/metrics"
export PYTHONPATH="$repo_dir"

inputs=()
for fold in 0 1 2 3 4; do
    inputs+=("$CORTICAL_DAG_RUN_DIR/artifacts/$CORTICAL_ARM/fold_$fold/metrics.json")
done
python -m tools.charite_cortical.continuity_workflow merge-metrics \
    --inputs "${inputs[@]}" \
    --arm "$CORTICAL_ARM" \
    --output "$CORTICAL_DAG_RUN_DIR/metrics/$CORTICAL_ARM-final.json"
