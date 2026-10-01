#!/bin/bash
# Resources and task mapping are supplied by tools.charite_cortical.submit_a100.
set -euo pipefail
: "${CORTICAL_GROUP_SCRIPT:?Missing original one-GPU script}"
: "${CORTICAL_GROUP_TASK_IDS:?Missing original task IDs}"
: "${CORTICAL_GROUP_SIZE:?Missing allocation size}"
: "${CORTICAL_GROUP_CPUS:?Missing per-task CPU count}"
: "${CORTICAL_GROUP_MEM_MB:?Missing per-task memory}"
IFS=: read -r -a task_ids <<< "$CORTICAL_GROUP_TASK_IDS"
group_index="${SLURM_ARRAY_TASK_ID:-0}"
first="$((group_index * CORTICAL_GROUP_SIZE))"
pids=()

stop_steps() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
resume_group() {
    echo "[A100] Time limit approaching; stopping independent steps before requeue"
    stop_steps
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap resume_group USR1
trap 'stop_steps; exit 143' TERM INT

echo "[A100] host=$(hostname) group=$group_index tasks=$CORTICAL_GROUP_TASK_IDS size=$CORTICAL_GROUP_SIZE"
for ((offset=0; offset<CORTICAL_GROUP_SIZE; offset++)); do
    task_id="${task_ids[first + offset]}"
    srun --exclusive --exact --nodes=1 --ntasks=1 \
        --cpus-per-task="$CORTICAL_GROUP_CPUS" \
        --mem="${CORTICAL_GROUP_MEM_MB}M" --gres=gpu:1 \
        --output="logs/%x-%j-task${task_id}.out" \
        env SLURM_ARRAY_TASK_ID="$task_id" bash "$CORTICAL_GROUP_SCRIPT" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
exit "$status"
