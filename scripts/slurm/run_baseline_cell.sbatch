#!/usr/bin/env bash
#SBATCH --job-name=p2b-bl
#SBATCH --output=/dcs07/chen/data/alignment-agent/baselines_campaign/logs/%x-%A_%a.out
#SBATCH --error=/dcs07/chen/data/alignment-agent/baselines_campaign/logs/%x-%A_%a.out
# One (task, method variant) baseline cell per array index. Rows file (tab-separated, no header):
#   variant  runner  task_toml  extra_args
# runner is a scripts/run_<runner>_baseline.py name, or "task_baseline" for the scripted methods
# (scripts/run_task_baseline.py + rescoring with the agents' scorer).
# Partition, resources, --exclude and the array throttle are given on the sbatch command line
# (see scripts/slurm/submit_baseline_campaign.sh). A per-cell status JSON is written to
# $CAMPAIGN/status/<task>__<variant>.json (start, end, exit code); OOM / wall-time kills are read from sacct.
set -uo pipefail
CAMPAIGN=${BL_CAMPAIGN:-/dcs07/chen/data/alignment-agent/baselines_campaign}   # a separate campaign sets BL_CAMPAIGN and BL_OUT
OUT_ROOT=${BL_OUT:-results/baselines}
REPO=${SAA_REPO:-$HOME/spatial-alignment-agent}
ROWS=${ROWS:?set ROWS}
cd "$REPO"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROWS")
IFS=$'\t' read -r VARIANT RUNNER TASK EXTRA <<<"$LINE"
[ -n "$TASK" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
TID=$(basename "$(dirname "$TASK")")
STATUS=$CAMPAIGN/status/${TID}__${VARIANT}.json
mkdir -p "$CAMPAIGN/status" /fastscratch/myscratch/ychen5/mpl /fastscratch/myscratch/ychen5/baseline_work
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE DGLBACKEND=pytorch PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/fastscratch/myscratch/ychen5/mpl
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NT"
MV=/dcs07/chen/data/alignment-agent/method-venvs
case $RUNNER in
  paste) PY=$MV/paste/bin/python ;;
  paste2) PY=.method-venv-ot/bin/python ;;
  spateo) PY=.method-venv-spateo/bin/python ;;
  spacel) PY=.method-venv-spacel/bin/python ;;
  geaso) PY=.method-venv-geaso/bin/python ;;
  cast) PY=$MV/cast/bin/python ;;
  staligner) PY=$MV/staligner/bin/python ;;
  slat) PY=$MV/scslat/bin/python ;;
  stalign) PY=$MV/stalign/bin/python ;;
  stair) PY=$MV/stair/bin/python ;;
  santo) PY=$MV/santo/bin/python; unset PYTHONPATH ;;
  gpsa) PY=$MV/gpsa/bin/python ;;
  gpsa_cuda) PY=$MV/gpsa-cuda/bin/python; SCRIPT=gpsa ;;   # same runner, torch cu121; pass --device cuda
  moscot) PY=$MV/moscot/bin/python ;;
  task_baseline) PY=.venv/bin/python ;;
  *) echo "unknown runner $RUNNER"; exit 2 ;;
esac
# Wall-clock budget for the method itself: the job's limit minus 45 min, so a slow run is recorded as a timeout here
# (and scoring of a finished run is not cut off).
LIMIT_S=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" -o %L | head -1 | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3)s+=3600*$(n-2); if(n>=4)s+=86400*$(n-3); print s}')
[[ "$LIMIT_S" =~ ^[0-9]+$ ]] || LIMIT_S=172800
BUDGET=$(( ${LIMIT_S:-172800} - 2700 )); [ "$BUDGET" -gt 600 ] || BUDGET=600
START=$(date +%s)
printf '{"task":"%s","variant":"%s","runner":"%s","job":"%s_%s","node":"%s","start":"%s","state":"running","extra_args":"%s"}\n' \
  "$TID" "$VARIANT" "$RUNNER" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(date -Is)" "$EXTRA" > "$STATUS"
echo "[$(date -Is)] $(hostname) $VARIANT $RUNNER $TASK $EXTRA (budget ${BUDGET}s)"
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null
if [ "$RUNNER" = task_baseline ]; then
  timeout "$BUDGET" "$PY" scripts/run_task_baseline.py "$TASK" --method "$VARIANT" --max-genes 100 --output-root "$OUT_ROOT"
  RC=$?
  OUT=$OUT_ROOT/$TID/$VARIANT
  if [ $RC -eq 0 ] && [ -d "$OUT/aligned_coordinates" ]; then
    PYTHONPATH=src timeout 3600 "$PY" scripts/score_agent_submission.py "$TASK" "$OUT" --output "$OUT/scores.json" >/dev/null || RC=99
  fi
else
  # shellcheck disable=SC2086
  timeout "$BUDGET" "$PY" "scripts/run_${SCRIPT:-$RUNNER}_baseline.py" "$TASK" --output-root "$OUT_ROOT" $EXTRA
  RC=$?
fi
END=$(date +%s)
printf '{"task":"%s","variant":"%s","runner":"%s","job":"%s_%s","node":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_s":%s,"exit_code":%s,"state":"%s","extra_args":"%s","log":"%s"}\n' \
  "$TID" "$VARIANT" "$RUNNER" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$START" "$END" "$((END-START))" "$RC" \
  "$([ $RC -eq 0 ] && echo done || ([ $RC -eq 124 ] && echo timeout || echo failed))" "$EXTRA" \
  "$CAMPAIGN/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" > "$STATUS"
echo "[$(date -Is)] exit $RC after $((END-START))s"
exit $RC
