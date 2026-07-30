#!/bin/bash
#SBATCH --job-name=cortical_ca
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=s-sc-pgpu04
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-fold%A_%a.out

set -euo pipefail

# One array task trains one fold from scratch (or resumes it with --c).
# Submit from the repository root after creating logs/:
#
#   mkdir -p logs
#   sbatch --export=ALL,CORTICAL_GATE_FILE=/absolute/path/oracle-gate.json \
#       slurm/charite_cortical/train_continuity_5fold.slurm

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array}"
: "${CORTICAL_GATE_FILE:?Set CORTICAL_GATE_FILE to a frozen 68-case gate JSON}"

export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
dataset_preprocessed="$nnUNet_preprocessed/$dataset_name"
gate_file="$(realpath "$CORTICAL_GATE_FILE")"

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

if [[ ! -f "$gate_file" ]]; then
    echo "[SLURM] Missing frozen oracle gate: $gate_file" >&2
    exit 2
fi

IFS=$'\t' read -r direction_set plans_name < <(
    python - "$gate_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    gate = json.load(handle)

if gate.get("neural_training_allowed") is not True:
    reasons = gate.get("reasons", ["gate did not allow neural training"])
    raise SystemExit("Cortical oracle gate failed: " + "; ".join(map(str, reasons)))

direction_set = gate.get("selected_o2_direction_set")
plans_by_direction = {
    "axial19": "nnUNetResEncUNetMPlansContinuity",
    "dense39": "nnUNetResEncUNetMPlansContinuityDense39",
}
if direction_set not in plans_by_direction:
    raise SystemExit(
        f"Unsupported selected_o2_direction_set in gate: {direction_set!r}"
    )

print(direction_set, plans_by_direction[direction_set], sep="\t")
PY
)

plans_file="$dataset_preprocessed/${plans_name}.json"
splits_file="$dataset_preprocessed/splits_final.json"

for required_file in "$plans_file" "$splits_file"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[SLURM] Missing required file: $required_file" >&2
        echo "[SLURM] Run slurm/charite_cortical/build_plan_preprocess.slurm first." >&2
        exit 2
    fi
done

data_identifier="$(
    python - "$plans_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    plans = json.load(handle)

try:
    print(plans["configurations"]["3d_fullres"]["data_identifier"])
except KeyError as error:
    raise SystemExit(
        f"Plans do not define 3d_fullres.data_identifier: missing {error}"
    ) from error
PY
)"
preprocessed_data="$dataset_preprocessed/$data_identifier"

if [[ ! -d "$preprocessed_data" ]]; then
    echo "[SLURM] Missing preprocessed configuration: $preprocessed_data" >&2
    echo "[SLURM] Run the matching axial19/dense39 preprocessing first." >&2
    exit 2
fi

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Array task/fold: $SLURM_ARRAY_TASK_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] Repository: $repo_dir"
echo "[SLURM] Dataset: $dataset_name"
echo "[SLURM] Direction set: $direction_set"
echo "[SLURM] Plans: $plans_name"
echo "[SLURM] Gate: $gate_file"
echo "[SLURM] Results: $nnUNet_results"
echo "[SLURM] Starting scratch C+A training (resume enabled)."

python nnunetv2/run/run_training.py \
    778 \
    3d_fullres \
    "$SLURM_ARRAY_TASK_ID" \
    -tr nnUNetTrainerCorticalContinuity \
    -p "$plans_name" \
    -num_gpus 1 \
    --c

echo "[SLURM] Fold $SLURM_ARRAY_TASK_ID finished."
