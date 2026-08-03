#!/bin/bash

# Baseline folds always continue. Each auxiliary increment is submitted only
# if its own folds-0/1 downstream gate passed.

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting}"
: "${CORTICAL_CONTINUITY_GATE:?Path to continuity-vs-base gate JSON}"
: "${CORTICAL_DENSITY_GATE:?Path to density-vs-continuity gate JSON}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
job_file="slurm/charite_cortical/train_separator_continuity_remaining.slurm"

gate_passed() {
    python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
raise SystemExit(0 if value.get("full_fivefold_allowed") is True else 1)
PY
}

cd "$repo_dir"
git pull --ff-only
sbatch --export=ALL,CORTICAL_ARM=matched_base "$job_file"

if gate_passed "$CORTICAL_CONTINUITY_GATE"; then
    sbatch --export=ALL,CORTICAL_ARM=continuity "$job_file"
    if gate_passed "$CORTICAL_DENSITY_GATE"; then
        sbatch --export=ALL,CORTICAL_ARM=continuity_density "$job_file"
    else
        echo "Density increment failed its pilot gate; folds 2-4 not submitted"
    fi
else
    echo "Continuity increment failed its pilot gate; no regularized folds 2-4 submitted"
fi
