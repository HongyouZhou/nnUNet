#!/bin/bash
#SBATCH --job-name=inference_dev
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=24:00:00
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
# Go to project dir
##############################
cd "$PROJECT_HOME/dev/nnUNet"
echo "[SLURM] Working dir: $(pwd)"

##############################
# Python path
##############################
export PYTHONPATH="${PWD}"

##############################
# Run inference
##############################
echo "[SLURM] Starting inference..."
python inference_dev.py

echo "[SLURM] Inference finished."
