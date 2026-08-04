#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 [--pilot-job JOBID] [--run-dir PATH] [--max-attempts N] [--dry-run]" >&2
}

pilot_job=""
run_dir=""
max_attempts=3
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot-job) pilot_job="$2"; shift 2 ;;
        --run-dir) run_dir="$2"; shift 2 ;;
        --max-attempts) max_attempts="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) usage; exit 2 ;;
    esac
done

: "${PROJECT_HOME:?Set PROJECT_HOME}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}/Dataset778_ChariteCorticalContinuity"
results_root="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
preprocessed_root="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
if [[ -z "$run_dir" ]]; then
    run_dir="$PROJECT_HOME/dev/data/cortical_separator_continuity_dag/run_$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-attempts must be a positive integer" >&2
    exit 2
fi

cd "$repo_dir"
if (( dry_run )); then
    cat <<EOF
separator-continuity DAG dry-run
  run_dir: $run_dir
  upstream pilot: ${pilot_job:-submit 0-5%4 with 128G}
  completion: checkpoint_final guard, at most $max_attempts bounded 128G retries
  pilot inference: 0-5%4, one OOF arm/fold per task
  pilot postprocess/evaluation: 0-5%3, frozen minimax-watershed contract
  gates: continuity_vs_matched_base, density_vs_continuity
  remaining: matched_base always; regularized arms conditional; folds 2-4
  final: per-arm fivefold merge and final-summary.json
EOF
    exit 0
fi

mkdir -p "$run_dir"
export PYTHONPATH="$repo_dir"
python -m tools.charite_cortical.continuity_workflow init-run \
    --run-dir "$run_dir" \
    --dataset "$dataset" \
    --preprocessed-root "$preprocessed_root" \
    --results-root "$results_root" >/dev/null

if [[ -z "$pilot_job" ]]; then
    pilot_job="$(sbatch --parsable \
        --mem=128G \
        --export=ALL,PROJECT_HOME="$PROJECT_HOME",CORTICAL_REPO_DIR="$repo_dir" \
        slurm/charite_cortical/train_separator_continuity_pilot.slurm)"
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$run_dir" --stage pilot_training --job-id "$pilot_job"
else
    scontrol show job "$pilot_job" >/dev/null
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$run_dir" --stage attached_pilot_training --job-id "$pilot_job"
fi

watch_job="$(sbatch --parsable \
    --dependency="afterany:$pilot_job" \
    --export=ALL,PROJECT_HOME="$PROJECT_HOME",CORTICAL_REPO_DIR="$repo_dir",CORTICAL_DAG_RUN_DIR="$run_dir",CORTICAL_PILOT_ATTEMPT=0,CORTICAL_PILOT_MAX_ATTEMPTS="$max_attempts" \
    slurm/charite_cortical/continue_separator_continuity_pilot.slurm)"
python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$run_dir" --stage pilot_watch_0 --job-id "$watch_job"
echo "DAG_RUN_DIR=$run_dir"
echo "PILOT_JOB=$pilot_job"
echo "WATCH_JOB=$watch_job"
