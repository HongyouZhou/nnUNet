#!/bin/bash
#SBATCH --job-name=nnunet_m_l3charv4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

PLANS_FILE="$nnUNet_preprocessed/Dataset777_Merged/nnUNetResEncUNetMPlans.json"
FULLRES_DATA="$nnUNet_preprocessed/Dataset777_Merged/nnUNetPlans_3d_fullres"
export NNUNET_FULL_INIT_CHECKPOINT="$nnUNet_results/Dataset777_Merged/nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth"

if [[ ! -f "$PLANS_FILE" ]]; then
    echo "[SLURM] Missing $PLANS_FILE"
    echo "[SLURM] Run: sbatch slurm/nn_plan_m.slurm"
    exit 2
fi

if [[ ! -d "$FULLRES_DATA" ]]; then
    echo "[SLURM] Missing $FULLRES_DATA"
    echo "[SLURM] Run fullres preprocessing before training."
    exit 2
fi

if [[ ! -f "$NNUNET_FULL_INIT_CHECKPOINT" ]]; then
    echo "[SLURM] Missing full-init checkpoint: $NNUNET_FULL_INIT_CHECKPOINT"
    exit 2
fi

echo "[SLURM] Starting Dataset777 M Charite v4 low-LR fine-tune from ChariteV3 checkpoint_final."
python nnunetv2/run/run_training.py 777 3d_fullres all \
    -tr nnUNetTrainer_L3SamplingCE3_ChariteV4FineTune100 \
    -p nnUNetResEncUNetMPlans \
    -num_gpus 1

echo "[SLURM] Training finished."
