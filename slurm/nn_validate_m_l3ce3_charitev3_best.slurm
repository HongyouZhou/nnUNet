#!/bin/bash
#SBATCH --job-name=nnunet_m_l3v3best
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

OUTPUT="$nnUNet_results/Dataset777_Merged/nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150__nnUNetResEncUNetMPlans__3d_fullres/fold_all"
FINAL_VALIDATION="$OUTPUT/validation"
FINAL_BACKUP="$OUTPUT/validation_checkpoint_final"
BEST_VALIDATION="$OUTPUT/validation_checkpoint_best"

if [[ ! -f "$OUTPUT/checkpoint_best.pth" ]]; then
    echo "[SLURM] Missing checkpoint_best.pth in $OUTPUT"
    exit 2
fi

if [[ ! -d "$FINAL_VALIDATION" || ! -f "$FINAL_VALIDATION/summary.json" ]]; then
    echo "[SLURM] Missing existing final validation summary in $FINAL_VALIDATION"
    exit 2
fi

echo "[SLURM] Preserving current final validation in $FINAL_BACKUP"
rm -rf "$FINAL_BACKUP"
cp -al "$FINAL_VALIDATION" "$FINAL_BACKUP"

echo "[SLURM] Running validation with checkpoint_best."
rm -rf "$FINAL_VALIDATION"
python nnunetv2/run/run_training.py 777 3d_fullres all \
    -tr nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150 \
    -p nnUNetResEncUNetMPlans \
    --val \
    --val_best \
    -num_gpus 1

echo "[SLURM] Saving checkpoint_best validation in $BEST_VALIDATION"
rm -rf "$BEST_VALIDATION"
mv "$FINAL_VALIDATION" "$BEST_VALIDATION"

echo "[SLURM] Restoring final validation folder."
cp -al "$FINAL_BACKUP" "$FINAL_VALIDATION"

echo "[SLURM] checkpoint_best validation finished."
