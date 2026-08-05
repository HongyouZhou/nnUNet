#!/bin/bash
#SBATCH --job-name=cortical_cont_smoke_gate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_SMOKE_RUN_DIR:?Set CORTICAL_SMOKE_RUN_DIR}"
: "${CORTICAL_CODE_REVISION:?Set CORTICAL_CODE_REVISION}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
results_root="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
max_epoch_seconds="${CORTICAL_SMOKE_MAX_EPOCH_SECONDS:-120}"

cd "$repo_dir"
mkdir -p logs "$CORTICAL_SMOKE_RUN_DIR"
export PYTHONPATH="$repo_dir"
python -m tools.charite_cortical.continuity_workflow smoke-status \
    --results-root "$results_root" \
    --max-epoch-seconds "$max_epoch_seconds" \
    --code-revision "$CORTICAL_CODE_REVISION" \
    --output "$CORTICAL_SMOKE_RUN_DIR/smoke-status.json"
echo "[SMOKE] all three full epochs passed the ${max_epoch_seconds}s throughput gate"
