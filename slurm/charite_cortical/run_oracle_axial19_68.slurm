#!/bin/bash
#SBATCH --job-name=cortical_oracle
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-67
#SBATCH --output=logs/%x-case%A_%a.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"

set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME before submitting this job}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a SLURM array}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
dataset_dir="$nnUNet_raw/Dataset778_ChariteCorticalContinuity"
oracle_root="${CORTICAL_ORACLE_ROOT:-$PROJECT_HOME/dev/data/cortical_continuity_oracles}"

cd "$repo_dir"
mkdir -p logs "$oracle_root"
export PYTHONPATH="$repo_dir"

case_id=$(python -c "import json,sys; value=json.load(open(sys.argv[1])); cases=sorted(value['cases'], key=lambda item: item['case_id']); print(cases[int(sys.argv[2])]['case_id'])" "$dataset_dir/continuity_manifest.json" "$SLURM_ARRAY_TASK_ID")
case_root="$oracle_root/$case_id"
mkdir -p "$case_root"

if [[ ! -f "$case_root/o1.json" ]]; then
    python -m tools.charite_cortical.continuity_cli oracle \
        --dataset "$dataset_dir" \
        --case "$case_id" \
        --mode O1 \
        --direction-set axial19 \
        --execute \
        --output "$case_root/o1.json"
fi

if [[ ! -f "$case_root/o2-axial19.json" ]]; then
    python -m tools.charite_cortical.continuity_cli oracle \
        --dataset "$dataset_dir" \
        --case "$case_id" \
        --mode O2 \
        --direction-set axial19 \
        --execute \
        --output "$case_root/o2-axial19.json"
fi
