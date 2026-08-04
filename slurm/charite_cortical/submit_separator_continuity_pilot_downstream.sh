#!/bin/bash

set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
common="ALL,PROJECT_HOME=$PROJECT_HOME,CORTICAL_REPO_DIR=$repo_dir,CORTICAL_DAG_RUN_DIR=$CORTICAL_DAG_RUN_DIR"

cd "$repo_dir"
infer_job="$(sbatch --parsable --export="$common" slurm/charite_cortical/infer_separator_continuity.slurm)"
post_job="$(sbatch --parsable --dependency="afterok:$infer_job" --export="$common" slurm/charite_cortical/postprocess_evaluate_separator_continuity.slurm)"
gate_job="$(sbatch --parsable --dependency="afterok:$post_job" --export="$common" slurm/charite_cortical/gate_separator_continuity_pilot.slurm)"
dispatch_job="$(sbatch --parsable --dependency="afterok:$gate_job" --export="$common" slurm/charite_cortical/dispatch_separator_continuity_remaining.slurm)"

python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage pilot_inference --job-id "$infer_job"
python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage pilot_postprocess_evaluate --job-id "$post_job"
python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage pilot_gates --job-id "$gate_job"
python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage remaining_dispatch --job-id "$dispatch_job"
echo "[DAG] pilot inference=$infer_job postprocess=$post_job gate=$gate_job dispatch=$dispatch_job"
