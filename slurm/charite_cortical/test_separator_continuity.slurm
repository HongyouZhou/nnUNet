#!/bin/bash
#SBATCH --job-name=cortical_cont_tests
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_SMOKE_RUN_DIR:?Set CORTICAL_SMOKE_RUN_DIR}"
: "${CORTICAL_CODE_REVISION:?Set CORTICAL_CODE_REVISION}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"

cd "$repo_dir"
mkdir -p logs "$CORTICAL_SMOKE_RUN_DIR"
export PYTHONPATH="$repo_dir"
pytest -q tests/cortical_separator_continuity
python - "$CORTICAL_SMOKE_RUN_DIR/tests-passed.json" "$CORTICAL_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

path, revision = sys.argv[1:]
Path(path).write_text(
    json.dumps({"passed": True, "code_revision": revision}, indent=2) + "\n",
    encoding="utf-8",
)
PY
