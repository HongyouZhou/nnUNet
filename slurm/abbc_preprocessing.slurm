#!/bin/bash
#SBATCH --job-name=abbc_preprocessing
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] CPUs requested: 8"

##############################
# Activate mamba environment
##############################
source /home/hozh10/.bashrc
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
cd "$PROJECT_HOME/dev/nnUNet"   # ← 建议用绝对路径
echo "[SLURM] Working dir: $(pwd)"

##############################
# Python path
##############################
export PYTHONPATH="${PWD}"

##############################
# Run abbc preprocessing
##############################
echo "[SLURM] Starting abbc preprocessing..."
python tools/PENGWIN/preprocessing/preprocessing.py

echo "[SLURM] Preprocessing finished."
