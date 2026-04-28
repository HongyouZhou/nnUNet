#!/bin/bash
#SBATCH --job-name=nnunet_val
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=s-sc-pgpu04
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] GPUs requested: 1"

##############################
# Activate mamba environment
##############################
source ~/.bashrc
mamba activate repair

##############################
# nnU-Net data paths
##############################
export nnUNet_raw="$PROJECT_HOME/dev/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_HOME/dev/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_HOME/dev/data/nnUNet_results"

##############################
# CUDA memory hygiene
##############################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

##############################
# Go to project dir
##############################
cd "$PROJECT_HOME/dev/nnUNet"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Python path
##############################
export PYTHONPATH="${PWD}"

##############################
# Validation only.
# --val loads checkpoint_final.pth and is mutually exclusive with --c.
# Training is still running, so we stage checkpoint_latest.pth as the
# "final" checkpoint just for this validation run, then restore.
##############################
CKPT_DIR="$nnUNet_results/Dataset777_Merged/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all"
LATEST="$CKPT_DIR/checkpoint_latest.pth"
FINAL="$CKPT_DIR/checkpoint_final.pth"
BACKUP=""

if [ ! -f "$LATEST" ]; then
    echo "[SLURM] ERROR: $LATEST not found"; exit 1
fi
if [ -f "$FINAL" ]; then
    BACKUP="$FINAL.bak.$SLURM_JOB_ID"
    echo "[SLURM] Backing up existing checkpoint_final.pth -> $BACKUP"
    mv "$FINAL" "$BACKUP"
fi
echo "[SLURM] Staging checkpoint_latest.pth as checkpoint_final.pth"
cp "$LATEST" "$FINAL"

restore() {
    rm -f "$FINAL"
    if [ -n "$BACKUP" ] && [ -f "$BACKUP" ]; then
        echo "[SLURM] Restoring original checkpoint_final.pth"
        mv "$BACKUP" "$FINAL"
    fi
}
trap restore EXIT

echo "[SLURM] Running validation..."
python nnunetv2/run/run_training.py 777 3d_fullres all -num_gpus 1 -p nnUNetResEncUNetLPlans --val

echo "[SLURM] Validation finished."
