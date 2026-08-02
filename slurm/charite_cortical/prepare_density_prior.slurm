#!/bin/bash
#SBATCH --job-name=cortical_dens_prep
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Build the independent paired-control/prior preprocessed target and run the
# OOF density audit. A failed audit exits before preprocessing or GPU training.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_name="Dataset778_ChariteCorticalContinuity"
dataset_dir="$nnUNet_raw/$dataset_name"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
plans_name="nnUNetResEncUNetMPlansSeparatorDensityPrior"
plans_file="$preprocessed_dir/${plans_name}.json"
calibration_dir="$preprocessed_dir/density_prior_calibration"
audit_gate="$preprocessed_dir/density_prior_audit_gate.json"
preprocess_processes="${CORTICAL_DENSITY_PREPROCESS_PROCESSES:-1}"

if [[ ! "$preprocess_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "CORTICAL_DENSITY_PREPROCESS_PROCESSES must be a positive integer" >&2
    exit 2
fi

cd "$repo_dir"
mkdir -p logs "$preprocessed_dir" "$calibration_dir"
export PYTHONPATH="$repo_dir"

python -m tools.charite_cortical.continuity_data.audit_dataset "$dataset_dir"

nnUNetv2_plan_experiment \
    -d 778 \
    -pl nnUNetPlannerResEncM \
    -preprocessor_name ChariteDensityPriorSeparatorPreprocessor \
    -overwrite_target_spacing 0.5 0.5 0.5 \
    -overwrite_plans_name "$plans_name"

python -m nnunetv2.training.cortical_continuity.configure_plans \
    --plans "$plans_file" \
    --configuration 3d_fullres \
    --separator-density-prior

python -m tools.charite_cortical.density_prior calibrate \
    --dataset "$dataset_dir" \
    --plans "$plans_file" \
    --output-dir "$calibration_dir"

python -m tools.charite_cortical.density_prior audit \
    --dataset "$dataset_dir" \
    --calibration-dir "$calibration_dir" \
    --output "$audit_gate"

nnUNetv2_preprocess \
    -d 778 \
    -plans_name "$plans_name" \
    -c 3d_fullres \
    -np "$preprocess_processes" \
    --no_pbar

echo "[SLURM] Density-prior preparation and audit passed."
echo "[SLURM] Calibration: $calibration_dir"
echo "[SLURM] Audit gate: $audit_gate"
