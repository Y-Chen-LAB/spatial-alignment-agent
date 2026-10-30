#!/usr/bin/env bash
# Submit part of the run matrix as a SLURM array.
#   scripts/slurm/submit_alignment.sh --shard 1 --dry-run
#   scripts/slurm/submit_alignment.sh --shard 1
#   scripts/slurm/submit_alignment.sh --cell A_breadth --shard 2
#   scripts/slurm/submit_alignment.sh --agent azure-openai --task sa-001-dlpfc-serial-align
#   scripts/slurm/submit_alignment.sh --shard 1 --agent azure-openai,claude,gemini   # everything but codex
#   scripts/slurm/submit_alignment.sh --shard 1 --gpu       # one GPU per job, on the gpu partition
#   scripts/slurm/submit_alignment.sh --task-file started.txt --agent claude --limit 5   # pace one arm
#
# Jobs are CPU-only unless --gpu is given: the alignment code the agents write is numpy and
# scipy, and the historical runs were CPU. --gpu asks SLURM for one device (P2B_GRES picks a
# type, e.g. gpu:tesa100:1), moves the jobs to the gpu partition unless P2B_PARTITION was set
# explicitly, and exposes the device inside the container.
#
# Resubmitting is safe: a row whose run already scored (a marker under $P2B_RUNS_ROOT/done)
# is skipped, so after an interrupted array the same command runs only what is missing.
# Rows that failed are not marked done and run again; pass --rerun to ignore markers.
#   scripts/slurm/submit_alignment.sh --task sa-001-dlpfc-serial-align
set -euo pipefail
USER_PARTITION="${P2B_PARTITION:-}"      # remembered before site_config.sh fills in the default
source "$(dirname "${BASH_SOURCE[0]}")/site_config.sh"
cd "$P2B_REPO"

CELL=""; AGENT_FILTER=""; MODEL_FILTER=""; COND_FILTER=""; TASK_FILTER=""; TASK_FILE=""; SHARD=""; DRY=0; RERUN=0; GPU=0; LIMIT=0; MAX_ATTEMPTS=2; RETRY_ONLY=0; ONE_PER_TASK=0; MAX_REP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cell) CELL="$2"; shift 2 ;;
        --agent) AGENT_FILTER="$2"; shift 2 ;;
        --model) MODEL_FILTER="$2"; shift 2 ;;         # comma-separated model ids
        --condition) COND_FILTER="$2"; shift 2 ;;      # equipped or bare
        --task) TASK_FILTER="$2"; shift 2 ;;
        --task-file) TASK_FILE="$2"; shift 2 ;;     # one task id per line
        --limit) LIMIT="$2"; shift 2 ;;             # submit at most this many rows, in task-file order
        --one-rep-per-task) ONE_PER_TASK=1; shift ;;   # at most one row per task, spreading a small batch across tasks
        --max-rep) MAX_REP="$2"; shift 2 ;;             # skip rows above this rep (breadth first: --max-rep 1)
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;  # skip rows that already failed this many times (0 = never skip)
        --retry-only) RETRY_ONLY=1; shift ;;           # only rows that have been attempted and failed
        --shard) SHARD="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --rerun) RERUN=1; shift ;;
        --gpu) GPU=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "$GPU" = 1 ]; then
    P2B_GPU=1
    P2B_GRES="${P2B_GRES:-gpu:1}"
    [ -n "$USER_PARTITION" ] || P2B_PARTITION=gpu
    export P2B_GPU P2B_GRES P2B_PARTITION
else
    P2B_GPU=0; P2B_GRES=""; export P2B_GPU P2B_GRES     # the flag is the only switch
fi

# claude runs carry a copy of the login; it must outlive them (see the script).
case ",$AGENT_FILTER," in *,claude,*|,,) [ "$DRY" = 1 ] || scripts/refresh_claude_login.sh || exit 1 ;; esac

mkdir -p "$P2B_ROWS_ROOT"
ROWS=$(mktemp "$P2B_ROWS_ROOT/p2b_rows.XXXXXX")
# Rows already queued or running: the rows files of this user's live p2b-align arrays.
# %F is the array's master id, one per array however many tasks are running. A lookup that
# finds nothing must not kill the submission under set -e, so every step tolerates failure.
LIVE_ROWS=""
for A in $(squeue -u "$USER" -h -n p2b-align -o "%F" 2>/dev/null | sort -u); do
    f=$( { sacct -j "$A" -X -n -o SubmitLine%400 2>/dev/null || true; } | { grep -oE "P2B_ROWS=[^ ,]+" || true; } | head -1 | cut -d= -f2)
    [ -n "$f" ] && [ -f "$f" ] || continue
    # only the tasks SLURM still holds count as live, in squeue's own range syntax
    idx=$( { squeue -j "$A" -h -o "%i" 2>/dev/null || true; } | sed -E 's/^[0-9]+_//; s/%[0-9]+//; s/[][]//g' | paste -sd, -)
    LIVE_ROWS="$LIVE_ROWS $f:$idx"
done

"$P2B_PYTHON" - "$P2B_MATRIX" "$CELL" "$AGENT_FILTER" "$TASK_FILTER" "$SHARD" "$ROWS" "$P2B_RUNS_ROOT/done" "$RERUN" "$LIVE_ROWS" "$TASK_FILE" "$LIMIT" "$MODEL_FILTER" "$COND_FILTER" "$MAX_ATTEMPTS" "$P2B_RUNS_ROOT" "$RETRY_ONLY" "$ONE_PER_TASK" "$MAX_REP" <<'PY'
import sys, pandas as pd
from pathlib import Path
matrix, cell, agent_filter, task_filter, shard, out, done_dir, rerun, live_rows, task_file, limit, model_filter, cond_filter, max_attempts, runs_root, retry_only, one_per_task, max_rep = sys.argv[1:19]
m = pd.read_csv(matrix)
m = m[m.runnable_today]
if cell: m = m[m.cell == cell]
if agent_filter: m = m[m.agent.isin(agent_filter.split(','))]
if model_filter: m = m[m.model.isin(model_filter.split(','))]
if cond_filter: m = m[m.condition == cond_filter]
if int(max_rep) > 0: m = m[m.rep <= int(max_rep)]
if task_filter: m = m[m.task == task_filter]
if task_file:
    order = Path(task_file).read_text().split()
    m = m[m.task.isin(order)].copy()
    m["_ord"] = m.task.map({t: i for i, t in enumerate(order)})
    m = m.sort_values(["_ord", "model", "condition", "rep"]).drop(columns="_ord")
if shard: m = m[m.shard == int(shard)]
selected = len(m)
cols = ["agent","model","task","condition","rep","prompt_variant","prebootstrap"]
key = m.agent + "__" + m.model + "__" + m.task + "__" + m.condition + "__rep" + m.rep.astype(str)
n_done = n_live = 0
if rerun != "1":
    # The array script touches one marker per scored row, named by the row's identity.
    done = {p.name for p in Path(done_dir).glob("*")} if Path(done_dir).is_dir() else set()
    n_done = int(key.isin(done).sum()); m = m[~key.isin(done)]; key = key[~key.isin(done)]
    # Rows in a live array are queued or running; submitting them again would duplicate work.
    live = set()
    for entry in live_rows.split():
        f, _, idx = entry.partition(":")
        lines = Path(f).read_text().splitlines()
        held = set()
        for part in idx.split(","):
            if not part: continue
            a, _, b = part.partition("-"); held.update(range(int(a), int(b or a) + 1))
        for i in held:
            if i < len(lines):
                a, mo, t, c, r = lines[i].split(",")[:5]
                live.add(f"{a}__{mo}__{t}__{c}__rep{r}")
    n_live = int(key.isin(live).sum()); m = m[~key.isin(live)]; key = key[~key.isin(live)]
    # Rows whose *agent* failed max_attempts times are left for a human: an agent that ran to
    # completion and produced no valid submission twice is a result. Infrastructure failures
    # (rate limits, timeouts, out-of-memory, node faults, bootstrap) never count; they are
    # retried until they run.
    n_exhausted = 0
    if int(max_attempts) > 0:
        import collections, json, re
        def _tail(path, n=4000):
            # last n bytes only: agent traces run to many MB, and every failed run is checked
            with path.open("rb") as h:
                h.seek(max(0, h.seek(0, 2) - n))
                return h.read().decode("utf-8", "replace")
        pushback = re.compile(r"(?i)\b429\b|rate.?limit|too many requests|overloaded|quota|resource.?exhausted|usage limit|oauth|expired|ECONNRESET|timed out|does not support this model|claude update")
        attempts = collections.Counter()
        for row in Path(runs_root).glob("*/row.csv"):
            a, mo, t, c, r = row.read_text().strip().split(",")[:5]
            for man in row.parent.glob("*/manifest.json"):
                d = json.loads(man.read_text())
                if d.get("status") != "failed" or d.get("failure_stage") not in ("agent", "score"):
                    continue
                text = str(d.get("failure_message", ""))
                for f in ("raw_stderr.txt", "raw_stdout.jsonl"):   # claude reports its limit in the stdout result event
                    if man.with_name(f).exists(): text += _tail(man.with_name(f))
                if pushback.search(text):
                    continue
                attempts[f"{a}__{mo}__{t}__{c}__rep{r}"] += 1
        exhausted = key.map(lambda k: attempts.get(k, 0) >= int(max_attempts)).astype(bool)
        n_exhausted = int(exhausted.sum()) if len(m) else 0
        if len(m): m = m[~exhausted.values]
# A warm row starts from this model's scored bare run, so without one it exits at once
# (alignment_array.sbatch, exit 4); hold it back until the bare run has scored.
if len(m) and (m.condition == "warm").any():
    bare_done = {p.name.rsplit("__", 2)[0] for p in Path(done_dir).glob("*__bare__rep*")}
    has_bare = (m.agent + "__" + m.model + "__" + m.task).isin(bare_done)
    keep = (m.condition != "warm") | has_bare
    m = m[keep]; key = key[keep]
if retry_only == "1":
    attempted = set()
    for row in Path(runs_root).glob("*/row.csv"):
        a, mo, t, c, r = row.read_text().strip().split(",")[:5]; attempted.add(f"{a}__{mo}__{t}__{c}__rep{r}")
    m = m[key.isin(attempted)]
if one_per_task == "1":
    m = m.drop_duplicates(subset=["task"])
capped = ""
if int(limit) > 0 and len(m) > int(limit):
    capped = f" (capped at {limit} of {len(m)})"; m = m.head(int(limit))
m[cols].to_csv(out, index=False, header=False)
print(f"{selected} runs selected, {n_done} already scored, {n_live} queued or running, {n_exhausted} failed {max_attempts}x (left alone), {len(m)} to submit{capped}", file=sys.stderr)
PY

N=$(wc -l < "$ROWS")
[ "$N" -gt 0 ] || { echo "nothing to submit"; rm -f "$ROWS"; exit 0; }
# SLURM's %N caps how many array tasks run at once; 0 means no cap, the scheduler decides.
THROTTLE=""; [ "${P2B_MAX_CONCURRENT:-0}" -gt 0 ] && THROTTLE="%$P2B_MAX_CONCURRENT"
echo "rows file: $ROWS  ($N runs)"
echo "array: 0-$((N - 1))$THROTTLE"

if [ "$DRY" = 1 ]; then
    echo "--- first 3 rows ---"; head -3 "$ROWS"
    echo "--- would run ---"
    echo "sbatch --array=0-$((N-1))$THROTTLE --partition=$P2B_PARTITION \\"
    echo "  --cpus-per-task=$P2B_CPUS --mem=$P2B_MEM --time=$P2B_TIME${P2B_GRES:+ --gres=$P2B_GRES}${P2B_EXCLUDE_NODES:+ --exclude=$P2B_EXCLUDE_NODES} \\"
    echo "  --export=ALL,P2B_ROWS=$ROWS scripts/slurm/alignment_array.sbatch"
    rm -f "$ROWS"      # a real submission's rows file must outlive its last pending task; a dry run's need not exist at all
    exit 0
fi

mkdir -p logs
sbatch --array="0-$((N - 1))$THROTTLE" \
       --partition="$P2B_PARTITION" \
       --cpus-per-task="$P2B_CPUS" --mem="$P2B_MEM" --time="$P2B_TIME" \
       ${P2B_GRES:+--gres="$P2B_GRES"} \
       ${P2B_EXCLUDE_NODES:+--exclude="$P2B_EXCLUDE_NODES"} \
       --export=ALL,P2B_ROWS="$ROWS" \
       scripts/slurm/alignment_array.sbatch
