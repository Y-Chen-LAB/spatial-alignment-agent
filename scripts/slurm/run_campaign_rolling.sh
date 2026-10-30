#!/usr/bin/env bash
#SBATCH --job-name=p2b-driver
#SBATCH --output=logs/p2b-driver-%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
# Keep each model at P2B_PER_MODEL rows in flight until every row has scored.
#
#   P2B_AGENTS="codex gemini" sbatch -p shared scripts/slurm/run_campaign_rolling.sh
#
# Every minute, for each agent: count its rows queued or running, and if that is below the
# cap submit the next rows in campaign order (tasks already started first), as a small
# array capped at the same number. submit_alignment.sh does the bookkeeping: rows already
# scored, queued, running, or failed twice on the agent's own account are never resubmitted.
# Long runs therefore no longer hold the rest of the campaign back. Stops when nothing is
# left to submit and nothing is in flight.
set -euo pipefail
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
: "${P2B_PER_MODEL:=10}"
# agent:model pairs, because one agent can carry two models (codex runs gpt-5.5 and gpt-6-astra)
: "${P2B_MODELS:=claude:claude-sonnet-4-6 codex:gpt-5.5 codex:gpt-6-astra gemini:gemini-3.5-flash}"
: "${P2B_POLL_SECONDS:=60}"
: "${ORDER:=benchmarks/spatial_alignment/campaigns/task_order.txt}"
: "${P2B_CONDITIONS:=bare equipped}"   # add warm to sweep that condition too
# P2B_MAX_REP=1 keeps a model on rep 1 everywhere before any repeat (breadth first)
mkdir -p logs
log() { echo "[$(date -Is)] $*"; }

# Rows of one model that SLURM still holds, across every live p2b-align array.
inflight() {
    local model=$1 n=0
    for A in $(squeue -u "$USER" -h -n p2b-align -o "%F" 2>/dev/null | sort -u); do
        f=$( { sacct -j "$A" -X -n -o SubmitLine%400 2>/dev/null || true; } | { grep -oE "P2B_ROWS=[^ ,]+" || true; } | head -1 | cut -d= -f2)
        [ -n "$f" ] && [ -f "$f" ] || continue
        idx=$( { squeue -j "$A" -h -o "%i" 2>/dev/null || true; } | sed -E 's/^[0-9]+_//; s/%[0-9]+//; s/[][]//g' | paste -sd, -)
        n=$((n + $("$P2B_PYTHON" - "$f" "$idx" "$model" <<'PY'
import sys
rows = open(sys.argv[1]).read().splitlines(); held = set()
for part in sys.argv[2].split(","):
    if not part: continue
    a, _, b = part.partition("-"); held.update(range(int(a), int(b or a) + 1))
print(sum(1 for i in held if i < len(rows) and rows[i].split(",")[1] == sys.argv[3]))
PY
)))
    done
    echo "$n"
}

# Seconds until this model may be submitted again, read from its most recent failed run.
# A subscription or rate limit makes every row die in seconds, so submitting through it
# only burns rows; wait for the reset the agent reports, or 30 min when it names none.
backoff() {
    "$P2B_PYTHON" - "$P2B_RUNS_ROOT" "$1" <<'PY'
import sys, json, re, datetime as dt
from pathlib import Path
root, model = Path(sys.argv[1]), sys.argv[2]
pushback = re.compile(r"(?i)\b429\b|rate.?limit|too many requests|session limit|usage limit|quota|resource.?exhausted")
now = dt.datetime.now(dt.timezone.utc)
latest = None
# a fresh claude login (e.g. after an account upgrade) can lift the limit, so failures from
# before it are not evidence; if the limit still holds, the next run fails and restores the hold
since = now.timestamp() - 6 * 3600
creds = Path.home() / ".claude" / ".credentials.json"
if model.startswith("claude") and creds.exists():
    since = max(since, creds.stat().st_mtime)
for man in root.glob(f"*/*_{model}_*/manifest.json"):
    if man.stat().st_mtime < since: continue
    if latest is None or man.stat().st_mtime > latest.stat().st_mtime: latest = man
if latest is None: print(0); sys.exit()
d = json.loads(latest.read_text())
# only the agent's own output: failure_message echoes the whole command and prompt, and the
# equipped prompt mentions a "package-count quota", which would read as a rate limit
text = ""
for f in ("raw_stderr.txt", "raw_stdout.jsonl"):
    if latest.with_name(f).exists():
        with latest.with_name(f).open("rb") as h:
            h.seek(max(0, h.seek(0, 2) - 4000)); text += h.read().decode("utf-8", "replace")
# a run that used its whole time limit was working, not being refused; a stray retry message
# near the end of its log is not a reason to pause the model
if d.get("failure_stage") == "agent_timeout": print(0); sys.exit()
if d.get("status") != "failed" or not pushback.search(text): print(0); sys.exit()
age = now.timestamp() - latest.stat().st_mtime
wait = 1800 - age
m = re.search(r"resets (\d{1,2})(?::(\d{2}))?\s*([ap]m)\s*\(UTC\)", text)
if m:
    h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
    t = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if t < now - dt.timedelta(hours=1): t += dt.timedelta(days=1)
    wait = (t - now).total_seconds() + 120
print(max(0, int(wait)))
PY
}

log "rolling driver: models=[$P2B_MODELS] cap=$P2B_PER_MODEL each, task list $ORDER"
declare -A last_backoff_log=()
idle_rounds=0
while :; do
    submitted_any=0; inflight_total=0; held_any=0
    # claude rows carry a copy of the host login; renew it on the host shortly before it
    # expires so rows submitted after that start with a valid one (no-op otherwise)
    login_ok=1
    case " $P2B_MODELS " in *" claude:"*) scripts/refresh_claude_login.sh >/dev/null 2>&1 || { login_ok=0; log "  claude login refresh failed; log in interactively"; } ;; esac
    for spec in $P2B_MODELS; do
        agent=${spec%%:*}; model=${spec##*:}
        n=$(inflight "$model"); inflight_total=$((inflight_total + n))
        free=$((P2B_PER_MODEL - n))
        [ "$free" -gt 0 ] || continue
        # an expired login makes submit_alignment.sh refuse claude rows; that is a hold, not "done"
        if [ "$agent" = claude ] && [ "$login_ok" = 0 ]; then held_any=1; continue; fi
        wait_s=$(backoff "$model" 2>/dev/null || echo 0)
        if [ "${wait_s:-0}" -gt 0 ]; then
            [ "${last_backoff_log[$model]:-0}" -lt $(( $(date +%s) - 900 )) ] && {
                log "  $model: rate-limited, holding submissions ~$((wait_s / 60)) min"; last_backoff_log[$model]=$(date +%s); }
            held_any=1   # rows are still owed; the driver must not read this as done
            continue
        fi
        # half the free slots to each condition, so bare and equipped advance together
        n_cond=$(echo $P2B_CONDITIONS | wc -w)
        half=$(( (free + n_cond - 1) / n_cond ))
        for cond in $P2B_CONDITIONS; do
            out=$(P2B_MAX_CONCURRENT="$P2B_PER_MODEL" scripts/slurm/submit_alignment.sh --task-file "$ORDER" \
                    --agent "$agent" --model "$model" --condition "$cond" --one-rep-per-task ${P2B_MAX_REP:+--max-rep "$P2B_MAX_REP"} --limit "$half" 2>&1 \
                  | grep -E "selected|Submitted|nothing|short" || true)
            if echo "$out" | grep -q "Submitted"; then
                submitted_any=1
                log "  $model $cond: in flight $n, +$half -> $(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')"
            fi
        done
    done
    ELAPSED=$(( $(date +%s) - ${START:=$(date +%s)} ))
    if [ "$ELAPSED" -gt "${P2B_HANDOVER_SECONDS:-19800}" ]; then
        log "six-hour limit approaching; submitting a fresh driver to continue"
        ORDER="$ORDER" P2B_PER_MODEL="$P2B_PER_MODEL" sbatch -p "$P2B_PARTITION" --export=ALL "$P2B_REPO/scripts/slurm/run_campaign_rolling.sh" || log "handover failed"
        exit 0
    fi
    if [ "$submitted_any" = 0 ] && [ "$inflight_total" = 0 ] && [ "$held_any" = 0 ]; then
        idle_rounds=$((idle_rounds + 1))
        [ "$idle_rounds" -ge 3 ] && { log "nothing left to submit and nothing in flight; done. scored: $(ls "$P2B_RUNS_ROOT/done" | wc -l)"; break; }
    else
        idle_rounds=0
    fi
    sleep "$P2B_POLL_SECONDS"
done
