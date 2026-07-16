#!/bin/bash
#SBATCH --job-name=nnunet_m_l3v3calf
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

OUTPUT_BASE="$nnUNet_results/Dataset777_Merged/nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150__nnUNetResEncUNetMPlans__3d_fullres/fold_all"
CHECKPOINT="$OUTPUT_BASE/checkpoint_final.pth"
CALIBRATION_OUTPUT="$OUTPUT_BASE/calibration_label3_bias_checkpoint_final"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[SLURM] Missing checkpoint: $CHECKPOINT"
    exit 2
fi

echo "[SLURM] Starting label3 bias calibration sweep from V3 checkpoint_final."
python tools/label3_bias_sweep.py 777 3d_fullres all \
    -tr nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150 \
    -p nnUNetResEncUNetMPlans \
    --checkpoint "$CHECKPOINT" \
    --output "$CALIBRATION_OUTPUT" \
    --label 3 \
    --biases=-0.40,-0.30,-0.20,-0.10,-0.05,0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.70 \
    --device cuda

echo "[SLURM] Label3 bias calibration sweep finished."
