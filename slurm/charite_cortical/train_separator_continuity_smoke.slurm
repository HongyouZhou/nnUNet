#!/bin/bash
#SBATCH --job-name=cortical_cont_smoke
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x-%A_%a.out

# Two fold-0 epochs per arm: epoch 0 absorbs Torch compile and epoch 1 measures
# steady-state throughput. Output is separate and never resumes a prior smoke.

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail

: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}"
: "${CORTICAL_SMOKE_RUN_DIR:?Set CORTICAL_SMOKE_RUN_DIR}"
: "${CORTICAL_CODE_REVISION:?Set CORTICAL_CODE_REVISION}"
python - "$CORTICAL_SMOKE_RUN_DIR/tests-passed.json" "$CORTICAL_CODE_REVISION" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status != {"passed": True, "code_revision": sys.argv[2]}:
    raise RuntimeError("HPC continuity tests did not pass for this code revision")
PY
export nnUNet_raw="${nnUNet_raw:-$PROJECT_HOME/dev/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_HOME/dev/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_HOME/dev/data/nnUNet_results}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_n_proc_DA=4
export nnUNet_def_n_proc=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
plans_name="nnUNetResEncUNetMPlansSeparatorContinuityPrior"
dataset_name="Dataset778_ChariteCorticalContinuity"
preprocessed_dir="$nnUNet_preprocessed/$dataset_name"
export CORTICAL_CONTINUITY_CALIBRATION_DIR="${CORTICAL_CONTINUITY_CALIBRATION_DIR:-$preprocessed_dir/separator_continuity_calibration}"

case "$SLURM_ARRAY_TASK_ID" in
    0) trainer="nnUNetTrainerCorticalSeparatorMatchedBaseSmoke" ;;
    1) trainer="nnUNetTrainerCorticalSeparatorContinuitySmoke" ;;
    2) trainer="nnUNetTrainerCorticalSeparatorContinuityDensityPriorSmoke" ;;
    *) echo "Smoke task must be 0..2" >&2; exit 2 ;;
esac

output_folder="$nnUNet_results/$dataset_name/${trainer}__${plans_name}__3d_fullres"
if [[ -e "$output_folder" ]]; then
    echo "Refusing to mix a new smoke with existing output $output_folder" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" == 2 ]]; then
    test -f "$CORTICAL_CONTINUITY_CALIBRATION_DIR/fold_0.json"
fi

cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"
echo "[SMOKE] trainer=$trainer fold=0 workers=$nnUNet_n_proc_DA mem_mb=${SLURM_MEM_PER_NODE:-unknown}"
python nnunetv2/run/run_training.py \
    778 3d_fullres 0 \
    -tr "$trainer" \
    -p "$plans_name" \
    -num_gpus 1
