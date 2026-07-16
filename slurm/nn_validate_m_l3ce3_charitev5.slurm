#!/bin/bash
#SBATCH --job-name=nnunet_m_l3v5val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -e

mkdir -p logs

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] GPUs requested: 1"

source ~/.bashrc
mamba activate repair

export nnUNet_raw="$PROJECT_HOME/dev/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_HOME/dev/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_HOME/dev/data/nnUNet_results"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_HOME/dev/nnUNet"
echo "[SLURM] Working dir: $(pwd)"

export PYTHONPATH="${PWD}"
unset NNUNET_FULL_INIT_CHECKPOINT

OUTPUT="$nnUNet_results/Dataset777_Merged/nnUNetTrainer_L3SamplingCE3_ChariteV5HardPatchFineTune80__nnUNetResEncUNetMPlans__3d_fullres/fold_all"
VALIDATION="$OUTPUT/validation"
FINAL_VALIDATION="$OUTPUT/validation_checkpoint_final"
BEST_VALIDATION="$OUTPUT/validation_checkpoint_best"

if [[ ! -f "$OUTPUT/checkpoint_final.pth" ]]; then
    echo "[SLURM] Missing checkpoint_final.pth in $OUTPUT"
    exit 2
fi

if [[ ! -f "$OUTPUT/checkpoint_best.pth" ]]; then
    echo "[SLURM] Missing checkpoint_best.pth in $OUTPUT"
    exit 2
fi

echo "[SLURM] Running validation with checkpoint_final."
rm -rf "$VALIDATION" "$FINAL_VALIDATION"
python nnunetv2/run/run_training.py 777 3d_fullres all \
    -tr nnUNetTrainer_L3SamplingCE3_ChariteV5HardPatchFineTune80 \
    -p nnUNetResEncUNetMPlans \
    --val \
    -num_gpus 1

echo "[SLURM] Saving checkpoint_final validation in $FINAL_VALIDATION"
mv "$VALIDATION" "$FINAL_VALIDATION"

echo "[SLURM] Running validation with checkpoint_best."
rm -rf "$VALIDATION" "$BEST_VALIDATION"
python nnunetv2/run/run_training.py 777 3d_fullres all \
    -tr nnUNetTrainer_L3SamplingCE3_ChariteV5HardPatchFineTune80 \
    -p nnUNetResEncUNetMPlans \
    --val \
    --val_best \
    -num_gpus 1

echo "[SLURM] Saving checkpoint_best validation in $BEST_VALIDATION"
mv "$VALIDATION" "$BEST_VALIDATION"

echo "[SLURM] Restoring checkpoint_final validation folder."
cp -al "$FINAL_VALIDATION" "$VALIDATION"

echo "[SLURM] V5 validation finished."
