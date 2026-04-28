#!/bin/bash
#SBATCH --job-name=nnunet_pred
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
PROB_DIR="$RUN_DIR/prob_l3"
INST_ROOT="$RUN_DIR/instances"
SPLIT_ROOT="$RUN_DIR/cortical_split"
mkdir -p "$ABBC_DIR" "$PROB_DIR" "$INST_ROOT" "$SPLIT_ROOT"

if [ -z "$(ls -A "$IN_DIR" 2>/dev/null)" ]; then
    echo "[SLURM] ERROR: no input images in $IN_DIR"; exit 1
fi
echo "[SLURM] Input images:"; ls "$IN_DIR"

##############################
# nnUNet inference (semantic ABBC, 4 classes) WITH softmax probabilities.
# Use checkpoint_latest.pth so we don't need to stage anything.
##############################
echo "[SLURM] Running nnUNetv2_predict (chk=latest, save_probabilities)..."
nnUNetv2_predict \
    -i "$IN_DIR" \
    -o "$ABBC_DIR" \
    -d 777 \
    -c 3d_fullres \
    -p nnUNetResEncUNetLPlans \
    -f all \
    -chk checkpoint_latest.pth \
    --save_probabilities

echo "[SLURM] ABBC predictions:"; ls "$ABBC_DIR"

##############################
# Extract softmax channel 3 (fracture surface) to per-case nifti
##############################
echo "[SLURM] Extracting prob_l3 channel..."
python tools/PENGWIN/inference/extract_prob_channel.py \
    --abbc_dir "$ABBC_DIR" \
    --out_dir  "$PROB_DIR" \
    --channel  3
echo "[SLURM] prob_l3 outputs:"; ls "$PROB_DIR"

##############################
# Stage *_input + *_pred pairs for inference_abbc2instance.py
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
# abbc2instance: only watershed (original/watershed/flow were equivalent in the
# previous run; watershed uses CT intensity and is the most informative one).
##############################
echo "[SLURM] === abbc2instance method=watershed ==="
INST_DIR="$INST_ROOT/watershed"
mkdir -p "$INST_DIR"
python tools/PENGWIN/inference/inference_abbc2instance.py \
    --input_dir  "$PAIR_DIR" \
    --output_dir "$INST_DIR" \
    --method watershed \
    --processes 4

##############################
# Cortical-anchored split, using prob_l3 for soft seeds + soft cost field
##############################
echo "[SLURM] === cortical_anchored_split (with prob_l3) ==="
for img in "$IN_DIR"/*_0000.nii.gz; do
    case=$(basename "$img" _0000.nii.gz)
    abbc_pred="$ABBC_DIR/$case.nii.gz"
    inst="$INST_DIR/$case/${case}_instance.nii.gz"
    prob_l3="$PROB_DIR/${case}_prob3.nii.gz"

    if [ ! -f "$inst" ] || [ ! -f "$abbc_pred" ]; then
        echo "[SLURM] WARN: missing inputs for $case (inst=$inst, abbc=$abbc_pred)"
        continue
    fi
    out="$SPLIT_ROOT/$case"
    mkdir -p "$out"

    cmd=(python tools/PENGWIN/postprocess/run_cortical_split.py
         --abbc_pred "$abbc_pred"
         --instances "$inst"
         --outdir    "$out"
         --also_use_pred_label3_as_seeds)
    if [ -f "$prob_l3" ]; then
        cmd+=(--prob_label_3 "$prob_l3")
    else
        echo "[SLURM] WARN: $prob_l3 not found, running cortical split without prob"
    fi
    echo "[SLURM] Running cortical split on $case ..."
    "${cmd[@]}"
done

echo "[SLURM] Done. Outputs:"
find "$ABBC_DIR" "$PROB_DIR" "$INST_ROOT" "$SPLIT_ROOT" -maxdepth 4 -name "*.nii.gz" | sort
