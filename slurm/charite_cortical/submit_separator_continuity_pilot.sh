#!/bin/bash

# Submit only after preparation succeeds. Retire the known pending legacy
# array first; running/completed jobs are not selected by the PENDING filter.

set -euo pipefail

repo_dir="${CORTICAL_REPO_DIR:-${PROJECT_HOME:?Set PROJECT_HOME}/dev/nnUNet}"
legacy_job_id="10246333"

if [[ -n "$(squeue -h -j "$legacy_job_id" -t PENDING 2>/dev/null)" ]]; then
    scancel '10246333_[0-3]'
    echo "Cancelled pending legacy array 10246333_[0-3]"
fi

cd "$repo_dir"
git pull --ff-only
sbatch slurm/charite_cortical/train_separator_continuity_pilot.slurm
