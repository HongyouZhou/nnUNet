#!/bin/bash
#SBATCH --job-name=cortical_cont_infer
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/%x-%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}/Dataset778_ChariteCorticalContinuity"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_npp=1
export nnUNet_nps=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

if [[ -n "${CORTICAL_ARM:-}" ]]; then
    arm="$CORTICAL_ARM"
    fold="$SLURM_ARRAY_TASK_ID"
else
    arm="$(python -m tools.charite_cortical.continuity_workflow map-task --task-id "$SLURM_ARRAY_TASK_ID" --field arm)"
    fold="$(python -m tools.charite_cortical.continuity_workflow map-task --task-id "$SLURM_ARRAY_TASK_ID" --field fold)"
fi
trainer="$(python - <<PY
from tools.charite_cortical.continuity_workflow import trainer_for_arm
print(trainer_for_arm("$arm"))
PY
)"

artifact_dir="$CORTICAL_DAG_RUN_DIR/artifacts/$arm/fold_$fold"
input_dir="$artifact_dir/inputs"
prediction_dir="$artifact_dir/predictions"
mkdir -p "$input_dir" "$prediction_dir"
expected="$(python -m tools.charite_cortical.continuity_workflow stage-fold-inputs --dataset "$dataset" --fold "$fold" --output "$input_dir")"

checkpoint="$nnUNet_results/Dataset778_ChariteCorticalContinuity/${trainer}__nnUNetResEncUNetMPlansSeparatorContinuityPrior__3d_fullres/fold_${fold}/checkpoint_final.pth"
test -f "$checkpoint"

echo "[SLURM] OOF inference arm=$arm fold=$fold expected_cases=$expected checkpoint=$checkpoint"
nnUNetv2_predict \
    -i "$input_dir" \
    -o "$prediction_dir" \
    -d 778 \
    -c 3d_fullres \
    -p nnUNetResEncUNetMPlansSeparatorContinuityPrior \
    -tr "$trainer" \
    -f "$fold" \
    -chk checkpoint_final.pth \
    -npp 1 \
    -nps 1 \
    --save_probabilities \
    --continue_prediction \
    --disable_progress_bar

actual="$(find "$prediction_dir" -maxdepth 1 -type f -name '*.npz' | wc -l)"
if [[ "$actual" -ne "$expected" ]]; then
    echo "Expected $expected probability outputs, found $actual" >&2
    exit 1
fi
