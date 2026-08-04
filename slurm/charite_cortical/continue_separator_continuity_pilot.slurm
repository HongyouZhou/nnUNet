#!/bin/bash
#SBATCH --job-name=cortical_cont_watch
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
results_root="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
attempt="${CORTICAL_PILOT_ATTEMPT:-0}"
max_attempts="${CORTICAL_PILOT_MAX_ATTEMPTS:-3}"

cd "$repo_dir"
mkdir -p logs "$CORTICAL_DAG_RUN_DIR/status"
export PYTHONPATH="$repo_dir"

status_json="$CORTICAL_DAG_RUN_DIR/status/pilot-attempt-${attempt}.json"
missing="$(python -m tools.charite_cortical.continuity_workflow training-status \
    --results-root "$results_root" \
    --stage pilot \
    --output "$status_json" \
    --print-missing)"

if [[ -z "$missing" ]]; then
    echo "[SLURM] All six pilot checkpoints are complete"
    bash slurm/charite_cortical/submit_separator_continuity_pilot_downstream.sh
    exit 0
fi
if (( attempt >= max_attempts )); then
    echo "Pilot checkpoints still missing after $attempt retries: $missing" >&2
    exit 1
fi

next_attempt="$((attempt + 1))"
retry_job="$(sbatch --parsable \
    --array="${missing}%4" \
    --mem=128G \
    --export=ALL,PROJECT_HOME="$PROJECT_HOME",CORTICAL_REPO_DIR="$repo_dir" \
    slurm/charite_cortical/train_separator_continuity_pilot.slurm)"
watch_job="$(sbatch --parsable \
    --dependency="afterany:$retry_job" \
    --export=ALL,PROJECT_HOME="$PROJECT_HOME",CORTICAL_REPO_DIR="$repo_dir",CORTICAL_DAG_RUN_DIR="$CORTICAL_DAG_RUN_DIR",CORTICAL_PILOT_ATTEMPT="$next_attempt",CORTICAL_PILOT_MAX_ATTEMPTS="$max_attempts" \
    slurm/charite_cortical/continue_separator_continuity_pilot.slurm)"
python -m tools.charite_cortical.continuity_workflow record-job \
    --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "pilot_retry_$next_attempt" --job-id "$retry_job"
python -m tools.charite_cortical.continuity_workflow record-job \
    --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "pilot_watch_$next_attempt" --job-id "$watch_job"
echo "[SLURM] Missing pilot tasks $missing; retry=$retry_job watch=$watch_job memory=128G"
