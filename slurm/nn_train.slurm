#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclude=s-sc-pgpu04
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] GPUs requested: 4"

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
# Go to project dir
##############################
cd "$PROJECT_HOME/dev/nnUNet"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Python path
##############################
export PYTHONPATH="${PWD}"

##############################
# Run nnUNet training
##############################
echo "[SLURM] Starting training on 4 GPUs..."
python nnunetv2/run/run_training.py 777 3d_fullres all -num_gpus 4 -p nnUNetResEncUNetLPlans

echo "[SLURM] Training finished."
