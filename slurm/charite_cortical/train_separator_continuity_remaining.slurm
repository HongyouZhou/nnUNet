#!/bin/bash
#SBATCH --job-name=cortical_cont_rem
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out

# Submit through submit_separator_continuity_remaining.sh. Each array covers
# folds 2-4 for one explicitly selected arm.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}"
: "${CORTICAL_ARM:?Set CORTICAL_ARM through the submission helper}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
plans_name="nnUNetResEncUNetMPlansSeparatorContinuityPrior"
export CORTICAL_CONTINUITY_CALIBRATION_DIR="${CORTICAL_CONTINUITY_CALIBRATION_DIR:-$preprocessed_dir/separator_continuity_calibration}"

case "$CORTICAL_ARM" in
    matched_base)
        trainer="nnUNetTrainerCorticalSeparatorMatchedBase"
        ;;
    continuity)
        trainer="nnUNetTrainerCorticalSeparatorContinuity"
        ;;
    continuity_density)
        trainer="nnUNetTrainerCorticalSeparatorContinuityDensityPrior"
        ;;
    *)
        echo "Unknown CORTICAL_ARM=$CORTICAL_ARM" >&2
        exit 2
        ;;
esac

fold="$((SLURM_ARRAY_TASK_ID + 2))"
if [[ "$CORTICAL_ARM" == "continuity_density" ]]; then
    test -f "$CORTICAL_CONTINUITY_CALIBRATION_DIR/fold_${fold}.json"
fi

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"
echo "[SLURM] trainer=$trainer fold=$fold cpus=${SLURM_CPUS_PER_TASK:-unknown} mem_per_node_mb=${SLURM_MEM_PER_NODE:-unknown} augmentation_workers=$nnUNet_n_proc_DA"

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
    778 3d_fullres "$fold" \
    -tr "$trainer" \
    -p "$plans_name" \
    -num_gpus 1 \
    --c &
training_pid="$!"
wait "$training_pid"
