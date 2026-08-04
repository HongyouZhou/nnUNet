#!/bin/bash
#SBATCH --job-name=cortical_cont_dispatch
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
mamba activate "${CORTICAL_ENV_NAME:-repair}"
set -euo pipefail
: "${PROJECT_HOME:?Set PROJECT_HOME}"
: "${CORTICAL_DAG_RUN_DIR:?Set CORTICAL_DAG_RUN_DIR}"
repo_dir="${CORTICAL_REPO_DIR:-$PROJECT_HOME/dev/nnUNet}"
cd "$repo_dir"
mkdir -p logs
export PYTHONPATH="$repo_dir"

selected_arms="$(python - "$CORTICAL_DAG_RUN_DIR/gates/continuity-gate.json" "$CORTICAL_DAG_RUN_DIR/gates/density-gate.json" <<'PY'
import json, sys
continuity = json.load(open(sys.argv[1], encoding="utf-8"))["full_fivefold_allowed"]
density = json.load(open(sys.argv[2], encoding="utf-8"))["full_fivefold_allowed"]
arms = ["matched_base"]
if continuity:
    arms.append("continuity")
    if density:
        arms.append("continuity_density")
print(",".join(arms))
PY
)"
IFS=',' read -r -a arms <<< "$selected_arms"
merge_jobs=()
for arm in "${arms[@]}"; do
    common="ALL,PROJECT_HOME=$PROJECT_HOME,CORTICAL_REPO_DIR=$repo_dir,CORTICAL_DAG_RUN_DIR=$CORTICAL_DAG_RUN_DIR,CORTICAL_ARM=$arm"
    train_job="$(sbatch --parsable \
        --mem=128G \
        --export="$common" \
        slurm/charite_cortical/train_separator_continuity_remaining.slurm)"
    infer_job="$(sbatch --parsable \
        --dependency="afterok:$train_job" \
        --array=2-4%3 \
        --export="$common" \
        slurm/charite_cortical/infer_separator_continuity.slurm)"
    post_job="$(sbatch --parsable \
        --dependency="afterok:$infer_job" \
        --array=2-4%3 \
        --export="$common" \
        slurm/charite_cortical/postprocess_evaluate_separator_continuity.slurm)"
    merge_job="$(sbatch --parsable \
        --dependency="afterok:$post_job" \
        --export="$common" \
        slurm/charite_cortical/merge_separator_continuity_final.slurm)"
    merge_jobs+=("$merge_job")
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "${arm}_remaining_training" --job-id "$train_job"
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "${arm}_remaining_inference" --job-id "$infer_job"
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "${arm}_remaining_postprocess" --job-id "$post_job"
    python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage "${arm}_fivefold_merge" --job-id "$merge_job"
done

dependency="afterok:$(IFS=:; echo "${merge_jobs[*]}")"
final_job="$(sbatch --parsable \
    --dependency="$dependency" \
    --export=ALL,PROJECT_HOME="$PROJECT_HOME",CORTICAL_REPO_DIR="$repo_dir",CORTICAL_DAG_RUN_DIR="$CORTICAL_DAG_RUN_DIR",CORTICAL_SELECTED_ARMS="$selected_arms" \
    slurm/charite_cortical/finalize_separator_continuity.slurm)"
python -m tools.charite_cortical.continuity_workflow record-job --run-dir "$CORTICAL_DAG_RUN_DIR" --stage final_summary --job-id "$final_job"
echo "[DAG] selected_arms=$selected_arms final_job=$final_job"
