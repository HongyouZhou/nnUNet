#!/bin/bash
#SBATCH --job-name=cortical_dens_full
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out

# Complete paired five-fold evidence after the folds 0/1 pilot gate:
# control folds 2/3/4 and density-prior folds 2/3/4.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array}"
: "${CORTICAL_DENSITY_PILOT_GATE:?Set the paired folds 0/1 pilot gate JSON}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
plans_name="nnUNetResEncUNetMPlansSeparatorDensityPrior"
plans_file="$preprocessed_dir/${plans_name}.json"
export CORTICAL_DENSITY_CALIBRATION_DIR="${CORTICAL_DENSITY_CALIBRATION_DIR:-$preprocessed_dir/density_prior_calibration}"

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

python - "$CORTICAL_DENSITY_PILOT_GATE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    gate = json.load(stream)
if gate.get("full_fivefold_allowed") is not True:
    raise SystemExit("Density-prior pilot gate did not allow five-fold completion: " + "; ".join(gate.get("reasons", [])))
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
for fold_index in 0 1 2 3 4; do
    test -f "$CORTICAL_DENSITY_CALIBRATION_DIR/fold_${fold_index}.json"
done

if (( SLURM_ARRAY_TASK_ID < 3 )); then
    trainer="nnUNetTrainerCorticalSeparatorControl"
    fold="$((SLURM_ARRAY_TASK_ID + 2))"
else
    trainer="nnUNetTrainerCorticalSeparatorDensityPrior"
    fold="$((SLURM_ARRAY_TASK_ID - 1))"
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
