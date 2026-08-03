#!/bin/bash
#SBATCH --job-name=cortical_cont_prep
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Build the independent six-channel target, fold calibrations, and diagnostic
# density audit. The audit result never gates preprocessing or GPU training.

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
plans_name="nnUNetResEncUNetMPlansSeparatorContinuityPrior"
plans_file="$preprocessed_dir/${plans_name}.json"
calibration_dir="$preprocessed_dir/separator_continuity_calibration"
audit_json="$preprocessed_dir/separator_continuity_density_diagnostic.json"
preprocess_processes="${CORTICAL_CONTINUITY_PREPROCESS_PROCESSES:-1}"

if [[ ! "$preprocess_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "CORTICAL_CONTINUITY_PREPROCESS_PROCESSES must be a positive integer" >&2
    exit 2
fi

cd "$repo_dir"
git pull --ff-only
mkdir -p logs "$preprocessed_dir" "$calibration_dir"
export PYTHONPATH="$repo_dir"

for required in labelsTr supportTr validMasksTr corticalInstancesTr imagesTr; do
    test -d "$dataset_dir/$required"
done
test -f "$dataset_dir/splits_final.json"
test -f "$dataset_dir/continuity_manifest.json"

nnUNetv2_plan_experiment \
    -d 778 \
    -pl nnUNetPlannerResEncM \
    -preprocessor_name ChariteCorticalSeparatorContinuityPreprocessor \
    -overwrite_target_spacing 0.5 0.5 0.5 \
    -overwrite_plans_name "$plans_name"

python -m nnunetv2.training.cortical_separator_continuity.configure_plans \
    --plans "$plans_file" \
    --configuration 3d_fullres

# Training and calibration must hash the same frozen split file.
install -m 0644 "$dataset_dir/splits_final.json" "$preprocessed_dir/splits_final.json"

python -m tools.charite_cortical.density_prior calibrate \
    --dataset "$dataset_dir" \
    --plans "$plans_file" \
    --output-dir "$calibration_dir"

# Diagnostic-only: this command always exits successfully after writing JSON,
# even when training_allowed is false.
python -m tools.charite_cortical.density_prior audit \
    --dataset "$dataset_dir" \
    --calibration-dir "$calibration_dir" \
    --output "$audit_json"

nnUNetv2_preprocess \
    -d 778 \
    -plans_name "$plans_name" \
    -c 3d_fullres \
    -np "$preprocess_processes" \
    --no_pbar

data_identifier="$(
    python - "$plans_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    plans = json.load(stream)
print(plans["configurations"]["3d_fullres"]["data_identifier"])
PY
)"
case_count="$(find "$preprocessed_dir/$data_identifier" -maxdepth 1 -type f -name '*.b2nd' ! -name '*_seg.b2nd' | wc -l)"
if [[ "$case_count" -ne 68 ]]; then
    echo "Expected 68 preprocessed cases, found $case_count" >&2
    exit 1
fi

echo "[SLURM] Prepared $case_count cases in $data_identifier"
echo "[SLURM] Diagnostic audit: $audit_json"
