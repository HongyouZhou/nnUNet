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
# Validation only on latest checkpoint
# (--c loads checkpoint_latest.pth; --val skips training and runs validation)
##############################
echo "[SLURM] Running validation on checkpoint_latest.pth..."
python nnunetv2/run/run_training.py 777 3d_fullres all -num_gpus 1 -p nnUNetResEncUNetLPlans --c --val

echo "[SLURM] Validation finished."
