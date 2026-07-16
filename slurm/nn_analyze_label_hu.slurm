#!/bin/bash
#SBATCH --job-name=nn_label_hu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
REPO="$PROJECT_HOME/dev/nnUNet"
DATASET="$PROJECT_HOME/dev/data/nnUNet_raw/Dataset777_Merged"
OUTPUT="$PROJECT_HOME/dev/data/analysis/Dataset777_label_hu_20260711"
PYTHON="$PROJECT_HOME/miniforge3/envs/repair/bin/python"

mkdir -p "$REPO/logs" "$OUTPUT"
cd "$REPO"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" tools/analyze_label_hu_distributions.py \
  "$DATASET" \
  "$OUTPUT" \
  --workers 8 \
  --sample-limit 100000 \
  --local-radius-mm 3

echo "Results: $OUTPUT"
