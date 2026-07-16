#!/bin/bash
#SBATCH --job-name=nn_split_viz
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
OUT_DIR="${OUT_DIR:-$DATA_ROOT/interactive_candidate_visualizations}"
CORE_DISTINCT_MAX_DISTANCE_MM="${CORE_DISTINCT_MAX_DISTANCE_MM:-15}"

case_args=(
    --case charite_38:1:1:7
    --case charite_3058:1:1:2
    --case charite_2394:1:1:3
)
if [[ -n "${CASE_SPECS:-}" ]]; then
    read -r -a requested_cases <<< "$CASE_SPECS"
    case_args=()
    for case_spec in "${requested_cases[@]}"; do
        case_args+=(--case "$case_spec")
    done
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

python tools/PENGWIN/postprocess/visualize_interactive_split_candidates.py \
    --abbc-dir "$ABBC_DIR" \
    --ct-dir "$CT_DIR" \
    --gt-dir "$GT_DIR" \
    --outdir "$OUT_DIR" \
    --prompt-radius-mm 2 \
    --core-erosion-mm 2 \
    --core-distinct-max-distance-mm "$CORE_DISTINCT_MAX_DISTANCE_MM" \
    "${case_args[@]}"
