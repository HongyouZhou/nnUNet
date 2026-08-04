#!/bin/bash
#SBATCH --job-name=cortical_cont_pilot
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-5%4
#SBATCH --output=logs/%x-%A_%a.out

# tasks 0-1 matched base; 2-3 continuity; 4-5 continuity+density.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The six-channel 0.5-mm target is substantially larger than a standard
# semantic target. Cluster hostnames fall back to 12 training augmentation
# workers in get_allowed_n_proc_DA(), plus 6 validation workers, which exceeded
# the former implicit 32-GiB allocation before the first batch. Keep all three
# pilot arms on the same bounded worker topology and prevent nested BLAS/OpenMP
# pools inside the augmentation processes.
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
plans_name="nnUNetResEncUNetMPlansSeparatorContinuityPrior"
plans_file="$preprocessed_dir/${plans_name}.json"
export CORTICAL_CONTINUITY_CALIBRATION_DIR="${CORTICAL_CONTINUITY_CALIBRATION_DIR:-$preprocessed_dir/separator_continuity_calibration}"

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

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

if (( SLURM_ARRAY_TASK_ID < 2 )); then
    trainer="nnUNetTrainerCorticalSeparatorMatchedBase"
    fold="$SLURM_ARRAY_TASK_ID"
elif (( SLURM_ARRAY_TASK_ID < 4 )); then
    trainer="nnUNetTrainerCorticalSeparatorContinuity"
    fold="$((SLURM_ARRAY_TASK_ID - 2))"
else
    trainer="nnUNetTrainerCorticalSeparatorContinuityDensityPrior"
    fold="$((SLURM_ARRAY_TASK_ID - 4))"
    test -f "$CORTICAL_CONTINUITY_CALIBRATION_DIR/fold_${fold}.json"
fi

echo "[SLURM] trainer=$trainer fold=$fold plans=$plans_name data=$data_identifier"
echo "[SLURM] cpus=${SLURM_CPUS_PER_TASK:-unknown} mem_per_node_mb=${SLURM_MEM_PER_NODE:-unknown} augmentation_workers=$nnUNet_n_proc_DA"

training_pid=""
requeue_for_resume() {
    echo "[SLURM] Time-limit signal received; terminating trainer and requeueing $SLURM_JOB_ID"
    if [[ -n "$training_pid" ]]; then
        kill -TERM "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_for_resume USR1

python nnunetv2/run/run_training.py \
    778 \
    3d_fullres \
    "$fold" \
    -tr "$trainer" \
    -p "$plans_name" \
    -num_gpus 1 \
    --c &
training_pid="$!"
wait "$training_pid"
