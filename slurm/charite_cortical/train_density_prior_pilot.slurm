#!/bin/bash
#SBATCH --job-name=cortical_dens_pilot
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out

# Four independent one-GPU jobs: control folds 0/1 and prior folds 0/1.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
plans_name="nnUNetResEncUNetMPlansSeparatorDensityPrior"
plans_file="$preprocessed_dir/${plans_name}.json"
gate_file="${CORTICAL_DENSITY_AUDIT_GATE:-$preprocessed_dir/density_prior_audit_gate.json}"
export CORTICAL_DENSITY_CALIBRATION_DIR="${CORTICAL_DENSITY_CALIBRATION_DIR:-$preprocessed_dir/density_prior_calibration}"

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

python - "$gate_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    gate = json.load(stream)
if gate.get("density_prior_training_allowed") is not True:
    raise SystemExit("Density-prior audit gate did not allow training: " + "; ".join(gate.get("reasons", [])))
PY

test -f "$plans_file"
test -f "$preprocessed_dir/splits_final.json"
data_identifier="$(
    python - "$plans_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    plans = json.load(stream)
print(plans["configurations"]["3d_fullres"]["data_identifier"])
PY
)"
test -d "$preprocessed_dir/$data_identifier"
for fold in 0 1 2 3 4; do
    test -f "$CORTICAL_DENSITY_CALIBRATION_DIR/fold_${fold}.json"
done

if (( SLURM_ARRAY_TASK_ID < 2 )); then
    trainer="nnUNetTrainerCorticalSeparatorControl"
    fold="$SLURM_ARRAY_TASK_ID"
else
    trainer="nnUNetTrainerCorticalSeparatorDensityPrior"
    fold="$((SLURM_ARRAY_TASK_ID - 2))"
fi

echo "[SLURM] trainer=$trainer fold=$fold plans=$plans_name data=$data_identifier"
python nnunetv2/run/run_training.py \
    778 \
    3d_fullres \
    "$fold" \
    -tr "$trainer" \
    -p "$plans_name" \
    -num_gpus 1 \
    --c
