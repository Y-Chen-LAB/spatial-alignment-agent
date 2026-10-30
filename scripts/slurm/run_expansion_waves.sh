#!/usr/bin/env bash
#SBATCH --job-name=p2b-driver
#SBATCH --output=logs/p2b-driver-%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
# Work through a task list in waves of one model at a time: P2B_WAVE bare rows and P2B_WAVE
# equipped rows for that model, one rep per task, then wait for the wave to finish before
# the next model's wave. Ten runs in the queue at a time, so a slow task delays only itself.
#
# The driver runs for six hours and, if rows remain, submits a fresh copy of itself before
# exiting, so a long campaign continues without holding a long-lived allocation.
#
#   P2B_TASK_FILE=benchmarks/spatial_alignment/campaigns/new_tasks_all.txt sbatch -p shared scripts/slurm/run_expansion_waves.sh
#   P2B_WAVE=5 P2B_MODELS="codex:gpt-5.5 gemini:gemini-3.5-flash" sbatch -p shared ...
#
# A wave is 2 x P2B_WAVE rows per model, so the queue holds a known amount and a slow task
# delays only its own wave. submit_alignment.sh keeps scored, in-flight and twice-failed rows
# out, so a wave that partly fails is simply retried by a later one. claude is included: when
# its usage window is exhausted its rows fail cheaply and come back in the next wave.
set -euo pipefail
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
: "${P2B_WAVE:=5}"
: "${P2B_MODELS:=claude:claude-sonnet-4-6 codex:gpt-5.5 codex:gpt-6-astra gemini:gemini-3.5-flash}"
: "${P2B_TASK_FILE:=benchmarks/spatial_alignment/campaigns/new_tasks_all.txt}"
: "${P2B_MAX_WAVES:=200}"          # 376 rows at 10 per wave, plus retries
mkdir -p logs
log() { echo "[$(date -Is)] $*"; }

remaining() {   # rows of this list still unscored, across the configured models
    "$P2B_PYTHON" - "$P2B_MATRIX" "$P2B_RUNS_ROOT/done" "$P2B_TASK_FILE" "$P2B_MODELS" <<'PY'
import sys, pandas as pd
from pathlib import Path
matrix, done_dir, task_file, models = sys.argv[1:5]
m = pd.read_csv(matrix)
m = m[m.runnable_today & m.task.isin(Path(task_file).read_text().split())]
m = m[m.model.isin([s.split(":")[1] for s in models.split()])]
done = {p.name for p in Path(done_dir).glob("*")} if Path(done_dir).is_dir() else set()
key = m.agent + "__" + m.model + "__" + m.task + "__" + m.condition + "__rep" + m.rep.astype(str)
print(int((~key.isin(done)).sum()))
PY
}

START=$(date +%s)
log "wave driver: $(remaining) rows left in $P2B_TASK_FILE; wave = ${P2B_WAVE} bare + ${P2B_WAVE} equipped per model"
for wave in $(seq 1 "$P2B_MAX_WAVES"); do
    left=$(remaining)
    [ "$left" -gt 0 ] || { log "nothing left; done"; break; }
    log "wave $wave: $left rows still unscored"
    # one model per wave, cycling
    set -- $P2B_MODELS
    n_models=$#
    idx=$(( (wave - 1) % n_models + 1 ))
    spec=$(eval echo \${$idx})
    MINE=""
    for spec in $spec; do
        agent=${spec%%:*}; model=${spec##*:}
        for cond in bare equipped; do
            out=$(P2B_MAX_CONCURRENT="$P2B_WAVE" scripts/slurm/submit_alignment.sh \
                    --task-file "$P2B_TASK_FILE" --agent "$agent" --model "$model" \
                    --condition "$cond" --one-rep-per-task --limit "$P2B_WAVE" 2>&1 \
                  | grep -E "selected|Submitted|nothing|short" || true)
            id=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$" || true)
            # every test here must not be the loop body's last failing command: set -e would end the driver
            if [ -n "$id" ]; then
                MINE="$MINE $id"
                log "  $model $cond -> $id ($(echo "$out" | grep -oE '[0-9]+ to submit' || echo 'submitted'))"
            elif echo "$out" | grep -q "short"; then
                log "  $model $cond: login too close to expiry, deferred to a later wave"
            else
                log "  $model $cond: nothing to submit"
            fi
        done
    done
    if [ -z "$MINE" ]; then
        log "wave $wave for $model submitted nothing (all scored, in flight, or login short); moving on"
        sleep 30
        continue
    fi
    sleep 45
    while [ "$(squeue -h -j "$(echo $MINE | tr ' ' ,)" 2>/dev/null | wc -l)" -gt 0 ]; do sleep 45; done
    log "wave $wave ($model) drained; scored so far: $(ls "$P2B_RUNS_ROOT/done" | wc -l)"
    # Hand over to a fresh driver before the wall clock ends, so nothing is cut mid-wave.
    ELAPSED=$(( $(date +%s) - START ))
    if [ "$ELAPSED" -gt "${P2B_HANDOVER_SECONDS:-19800}" ] && [ "$(remaining)" -gt 0 ]; then
        log "six-hour limit approaching; submitting a fresh driver to continue"
        sbatch -p "${P2B_PARTITION}" "$P2B_REPO/scripts/slurm/run_expansion_waves.sh" || log "handover submission failed"
        exit 0
    fi
done
log "finished: $(remaining) rows still unscored in $P2B_TASK_FILE"
