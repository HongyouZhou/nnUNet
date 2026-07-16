#!/bin/bash
#SBATCH --job-name=nn_split_candidates
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate repair

set -euo pipefail

PROJECT_ROOT="/sc-projects/sc-proj-cc09-repair/hongyou"
DATA_ROOT="$PROJECT_ROOT/dev/data"
REPO_ROOT="$PROJECT_ROOT/dev/nnUNet"
ABBC_DIR="$DATA_ROOT/nnUNet_results/Dataset777_Merged/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/validation"
CT_DIR="$DATA_ROOT/nnUNet_raw/Dataset777_Merged/imagesTr"
GT_DIR="$DATA_ROOT/nnUNet_raw/Dataset777_Merged/labelsTr_raw"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/interactive_candidate_scan/abbc_lplan_validation}"

case_args=()
if [[ -n "${CASES:-}" ]]; then
    normalized_cases="${CASES//:/ }"
    read -r -a requested_cases <<< "$normalized_cases"
    case_args=(--cases "${requested_cases[@]}")
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

python tools/PENGWIN/postprocess/find_interactive_split_candidates.py \
    --abbc-dir "$ABBC_DIR" \
    --ct-dir "$CT_DIR" \
    --gt-dir "$GT_DIR" \
    --outdir "$OUT_DIR" \
    --min-overlap-voxels 500 \
    --min-gt-coverage 0.2 \
    --min-gt-size 1000 \
    --core-erosion-mm 2 \
    --core-component-min-voxels 20 \
    "${case_args[@]}"
