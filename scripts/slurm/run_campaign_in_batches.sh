#!/usr/bin/env bash
#SBATCH --job-name=p2b-driver
#SBATCH --output=logs/p2b-driver-%j.out
#SBATCH --time=5-00:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
# Run the campaign a few tasks at a time, each model capped, until every row has scored.
#
#   sbatch -p shared scripts/slurm/run_campaign_in_batches.sh
#   P2B_BATCH_TASKS=8 P2B_PER_MODEL=15 sbatch -p shared scripts/slurm/run_campaign_in_batches.sh
#   P2B_AGENTS="codex gemini" sbatch -p shared scripts/slurm/run_campaign_in_batches.sh   # hold one arm back
#
# Each batch takes the next N tasks that still have unscored rows, in the order of
# benchmarks/spatial_alignment/campaigns/task_order.txt, and submits their missing rows as one array
# per model with SLURM's %cap, so no model exceeds the cap. Rows already scored are never
# resubmitted (submit_alignment.sh skips them). The next batch starts when nothing of the
# current one is queued or running. A task is visited at most once per pass; a second pass
# revisits tasks with gaps, and that is all: an agent that produced no submission twice is
# a result, not a retry.
set -euo pipefail
# SLURM runs a copy of this script from its spool directory; find the repo from the submit dir.
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
: "${P2B_BATCH_TASKS:=8}"
: "${P2B_PER_MODEL:=15}"
: "${P2B_PASSES:=2}"
: "${P2B_AGENTS:=claude codex gemini}"   # arms to run; rows of other agents are ignored
ORDER=benchmarks/spatial_alignment/campaigns/task_order.txt
STATE="$P2B_RUNS_ROOT/driver"
mkdir -p "$STATE" logs
log() { echo "[$(date -Is)] $*"; }

# Tasks with at least one unscored row for the configured agents, in campaign order, minus
# those in the file given.
remaining_tasks() {
    "$P2B_PYTHON" - "$P2B_MATRIX" "$P2B_RUNS_ROOT/done" "$ORDER" "${1:-/dev/null}" "$P2B_AGENTS" <<'PY'
import sys, pandas as pd
from pathlib import Path
matrix, done_dir, order, seen, agents = sys.argv[1:6]
m = pd.read_csv(matrix)
m = m[m.agent.isin(agents.split())]
done = {p.name for p in Path(done_dir).glob("*")} if Path(done_dir).is_dir() else set()
key = m.agent + "__" + m.model + "__" + m.task + "__" + m.condition + "__rep" + m.rep.astype(str)
missing = set(m[~key.isin(done)].task) - set(Path(seen).read_text().split())
for t in Path(order).read_text().split():
    if t in missing: print(t)
PY
}

for pass_no in $(seq 1 "$P2B_PASSES"); do
    SEEN="$STATE/pass${pass_no}_visited.txt"; : > "$SEEN"
    log "pass $pass_no: $(remaining_tasks "$SEEN" | wc -l) tasks still have unscored rows"
    while :; do
        BATCH="$STATE/batch.txt"
        remaining_tasks "$SEEN" | head -n "$P2B_BATCH_TASKS" > "$BATCH"
        [ -s "$BATCH" ] || break
        log "batch: $(tr '\n' ' ' < "$BATCH")"
        MINE=""
        for agent in $P2B_AGENTS; do
            out=$(P2B_MAX_CONCURRENT="$P2B_PER_MODEL" scripts/slurm/submit_alignment.sh --task-file "$BATCH" --agent "$agent" 2>&1 | grep -E "selected|Submitted|nothing")
            echo "$out" | sed "s/^/  $agent: /"
            id=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"); [ -n "$id" ] && MINE="$MINE $id"
        done
        cat "$BATCH" >> "$SEEN"
        # Wait only on this batch's own arrays: arrays submitted by hand (a paced claude
        # batch, a one-off retry) must not hold the next batch back.
        sleep 90
        while [ -n "$MINE" ] && [ "$(squeue -h -j "$(echo $MINE | tr ' ' ,)" 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
        log "batch drained; scored so far: $(ls "$P2B_RUNS_ROOT/done" | wc -l) of $(($(wc -l < "$P2B_MATRIX") - 1))"
    done
done
log "finished: $(ls "$P2B_RUNS_ROOT/done" | wc -l) rows scored; $(remaining_tasks | wc -l) tasks still have gaps after $P2B_PASSES passes"
