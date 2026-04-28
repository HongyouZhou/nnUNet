#!/bin/bash
#SBATCH --job-name=nnunet_pred
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=s-sc-pgpu04
#SBATCH --time=02:00:00
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_HOME/dev/nnUNet"
export PYTHONPATH="${PWD}"
echo "[SLURM] Working dir: $(pwd)"

##############################
# I/O layout (matches local rsync target)
##############################
RUN_DIR="$PROJECT_HOME/dev/data/predict_runs/tmp_12_02"
IN_DIR="$RUN_DIR/imagesTs"
ABBC_DIR="$RUN_DIR/abbc"
INST_ROOT="$RUN_DIR/instances"
mkdir -p "$ABBC_DIR" "$INST_ROOT"

if [ -z "$(ls -A "$IN_DIR" 2>/dev/null)" ]; then
    echo "[SLURM] ERROR: no input images in $IN_DIR"; exit 1
fi
echo "[SLURM] Input images:"; ls "$IN_DIR"

##############################
# nnUNet inference (semantic ABBC, 4 classes)
# Use checkpoint_latest.pth so we don't need to stage anything.
##############################
echo "[SLURM] Running nnUNetv2_predict (chk=latest)..."
nnUNetv2_predict \
    -i "$IN_DIR" \
    -o "$ABBC_DIR" \
    -d 777 \
    -c 3d_fullres \
    -p nnUNetResEncUNetLPlans \
    -f all \
    -chk checkpoint_latest.pth

echo "[SLURM] ABBC predictions:"; ls "$ABBC_DIR"

##############################
# Stage *_input + *_pred pairs for inference_abbc2instance.py
# (script expects "<case>_input.nii.gz" + "<case>_pred.nii.gz" siblings)
##############################
PAIR_DIR="$RUN_DIR/pairs"
rm -rf "$PAIR_DIR"; mkdir -p "$PAIR_DIR"
for img in "$IN_DIR"/*_0000.nii.gz; do
    base=$(basename "$img" _0000.nii.gz)
    if [ ! -f "$ABBC_DIR/$base.nii.gz" ]; then
        echo "[SLURM] WARN: missing prediction for $base"; continue
    fi
    cp "$img"                  "$PAIR_DIR/${base}_input.nii.gz"
    cp "$ABBC_DIR/$base.nii.gz" "$PAIR_DIR/${base}_pred.nii.gz"
done
echo "[SLURM] Pairs staged:"; ls "$PAIR_DIR"

##############################
# Run all 3 abbc2instance methods, each into its own subdir
##############################
for METHOD in original watershed flow; do
    echo "[SLURM] === abbc2instance method=$METHOD ==="
    OUT="$INST_ROOT/$METHOD"
    mkdir -p "$OUT"
    python tools/PENGWIN/inference/inference_abbc2instance.py \
        --input_dir  "$PAIR_DIR" \
        --output_dir "$OUT" \
        --method "$METHOD" \
        --processes 4
done

echo "[SLURM] Predict + postprocess finished."
echo "[SLURM] Results:"
find "$INST_ROOT" -maxdepth 3 -name "*_instance.nii.gz"
