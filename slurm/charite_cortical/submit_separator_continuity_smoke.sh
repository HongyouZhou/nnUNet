#!/bin/bash

set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
run_dir="${CORTICAL_SMOKE_RUN_DIR:-$PROJECT_HOME/dev/data/cortical_separator_continuity_smoke/run_$(date +%Y%m%d_%H%M%S)}"
dependency="${CORTICAL_SMOKE_DEPENDENCY:-}"

cd "$repo_dir"
revision="$(git rev-parse HEAD)"
common="ALL,PROJECT_HOME=$PROJECT_HOME,CORTICAL_REPO_DIR=$repo_dir,CORTICAL_SMOKE_RUN_DIR=$run_dir,CORTICAL_CODE_REVISION=$revision"
mkdir -p "$run_dir" logs
test_submit_args=(--parsable --export="$common")
if [[ -n "$dependency" ]]; then
    test_submit_args+=(--dependency="$dependency")
fi
test_job="$(sbatch "${test_submit_args[@]}" slurm/charite_cortical/test_separator_continuity.slurm)"
smoke_job="$(sbatch --parsable \
    --dependency="afterany:$test_job" \
    --export="$common" \
    slurm/charite_cortical/train_separator_continuity_smoke.slurm)"
gate_job="$(sbatch --parsable \
    --dependency="afterany:$smoke_job" \
    --export="$common" \
    slurm/charite_cortical/gate_separator_continuity_smoke.slurm)"

python - "$run_dir/jobs.json" "$test_job" "$smoke_job" "$gate_job" "$dependency" "$revision" <<'PY'
import json
import sys
from pathlib import Path

path, test_job, smoke_job, gate_job, dependency, revision = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "test_job": test_job,
            "smoke_job": smoke_job,
            "gate_job": gate_job,
            "upstream_dependency": dependency,
            "code_revision": revision,
            "formal_pilot_submitted": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "SMOKE_RUN_DIR=$run_dir"
echo "SMOKE_TEST_JOB=$test_job"
echo "SMOKE_JOB=$smoke_job"
echo "SMOKE_GATE_JOB=$gate_job"
