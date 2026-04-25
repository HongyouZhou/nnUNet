#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"

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
cd "$PROJECT_HOME/dev/nnUNet"   # ← 建议用绝对路径
echo "[SLURM] Working dir: $(pwd)"

##############################
# Python path
##############################
export PYTHONPATH="${PWD}"

##############################
# Run nnUNet preprocessing
##############################
python nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py \
    -d 777 \
    --verify_dataset_integrity \
    -pl nnUNetPlannerResEncL

echo "[SLURM] Preprocessing finished."
