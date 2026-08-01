#!/bin/bash
#SBATCH --job-name=cortical_dense39
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${CORTICAL_GATE_FILE:?Set CORTICAL_GATE_FILE to the dense39 gate JSON}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_root="$nnUNet_preprocessed/Dataset778_ChariteCorticalContinuity"
source_plans="$dataset_root/nnUNetResEncUNetMPlansContinuity.before_cortical_continuity.json"
dense_plans="$dataset_root/nnUNetResEncUNetMPlansContinuityDense39.json"

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

python -c "import json,sys; value=json.load(open(sys.argv[1])); assert value.get('neural_training_allowed') is True, value.get('reasons'); assert value.get('selected_o2_direction_set') == 'dense39'" "$CORTICAL_GATE_FILE"
test -f "$source_plans"
test ! -e "$dense_plans"

python -m nnunetv2.training.cortical_continuity.configure_plans \
    --plans "$source_plans" \
    --output "$dense_plans" \
    --configuration 3d_fullres \
    --direction-set dense39

nnUNetv2_preprocess \
    -d 778 \
    -plans_name nnUNetResEncUNetMPlansContinuityDense39 \
    -c 3d_fullres \
    -np "${CORTICAL_CONTINUITY_PREPROCESS_PROCESSES:-1}" \
    --no_pbar
