#!/bin/bash
#SBATCH --job-name=cortical_cont_gate
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
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
mkdir -p logs "$CORTICAL_DAG_RUN_DIR/metrics" "$CORTICAL_DAG_RUN_DIR/gates"
export PYTHONPATH="$repo_dir"

for arm in matched_base continuity continuity_density; do
    python -m tools.charite_cortical.continuity_workflow merge-metrics \
        --inputs \
            "$CORTICAL_DAG_RUN_DIR/artifacts/$arm/fold_0/metrics.json" \
            "$CORTICAL_DAG_RUN_DIR/artifacts/$arm/fold_1/metrics.json" \
        --arm "$arm" \
        --output "$CORTICAL_DAG_RUN_DIR/metrics/$arm-pilot.json"
done

set +e
python -m tools.charite_cortical.density_prior increment-gate \
    --reference "$CORTICAL_DAG_RUN_DIR/metrics/matched_base-pilot.json" \
    --candidate "$CORTICAL_DAG_RUN_DIR/metrics/continuity-pilot.json" \
    --comparison-name continuity_vs_matched_base \
    --output "$CORTICAL_DAG_RUN_DIR/gates/continuity-gate.json"
continuity_status="$?"
python -m tools.charite_cortical.density_prior increment-gate \
    --reference "$CORTICAL_DAG_RUN_DIR/metrics/continuity-pilot.json" \
    --candidate "$CORTICAL_DAG_RUN_DIR/metrics/continuity_density-pilot.json" \
    --comparison-name density_vs_continuity \
    --output "$CORTICAL_DAG_RUN_DIR/gates/density-gate.json"
density_status="$?"
set -e

if { [[ "$continuity_status" -ne 0 ]] && [[ "$continuity_status" -ne 2 ]]; } || \
   { [[ "$density_status" -ne 0 ]] && [[ "$density_status" -ne 2 ]]; }; then
    echo "Gate command failed: continuity=$continuity_status density=$density_status" >&2
    exit 1
fi
echo "[SLURM] Gate JSONs written; metric rejection is diagnostic control flow, not a job failure"
