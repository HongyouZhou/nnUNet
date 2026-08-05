#!/bin/bash

# Submit only after preparation succeeds. Retire the known pending legacy
# array first; running/completed jobs are not selected by the PENDING filter.

set -euo pipefail

repo_dir="${CORTICAL_REPO_DIR:-${PROJECT_HOME:?Set PROJECT_HOME}/dev/nnUNet}"
legacy_job_id="10246333"
: "${CORTICAL_SMOKE_RUN_DIR:?Set CORTICAL_SMOKE_RUN_DIR to a passed smoke run}"

cd "$repo_dir"
git pull --ff-only
revision="$(git rev-parse HEAD)"
python - "$CORTICAL_SMOKE_RUN_DIR/smoke-status.json" "$revision" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(
    Path(sys.argv[1]).expanduser().resolve(strict=True).read_text(encoding="utf-8")
)
if status.get("passed") is not True or status.get("code_revision") != sys.argv[2]:
    raise RuntimeError(
        "A passed performance smoke from the current revision is required"
    )
PY

if [[ -n "$(squeue -h -j "$legacy_job_id" -t PENDING 2>/dev/null)" ]]; then
    scancel '10246333_[0-3]'
    echo "Cancelled pending legacy array 10246333_[0-3]"
fi

sbatch slurm/charite_cortical/train_separator_continuity_pilot.slurm
