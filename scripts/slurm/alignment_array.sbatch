#!/usr/bin/env bash
#SBATCH --job-name=p2b-align
#SBATCH --output=logs/p2b-align-%A_%a.out
#SBATCH --error=logs/p2b-align-%A_%a.err
#SBATCH --signal=B:TERM@120
set -euo pipefail

# SLURM copies this script into its spool directory before running it, so its own location
# says nothing about where the repo is. The submit script exports P2B_REPO and submits from
# the repo root, so either of these points home.
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"

# One matrix row per array index. The row list is filtered at submit time and written to
# $P2B_ROWS, so array index N maps to the Nth line of that file rather than to the whole
# matrix; this keeps indices dense when only part of the matrix is being run.
ROW_FILE="${P2B_ROWS:?P2B_ROWS not set; submit through submit_alignment.sh}"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROW_FILE")
[ -n "$ROW" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }

IFS=',' read -r AGENT MODEL TASK CONDITION REP PROMPT_VARIANT PREBOOTSTRAP <<< "$ROW"
echo "[$(date -Is)] task=$TASK agent=$AGENT model=$MODEL condition=$CONDITION rep=$REP"

# Codex used to be refused here because its read-deny profile is macOS sandbox-exec. Container
# mode supplies the boundary instead, so codex runs on Linux like the others; the runner still
# exits rather than downgrading if no container runtime is available.

mkdir -p "$P2B_RUNS_ROOT" logs
# Refuse to start on a node whose local disk is nearly full, so the row is retried elsewhere
# rather than failing half-way. Checked before anything is written there. With the shared
# package cache an equipped workspace is ~60 MB, but the cache itself is ~21 GB per node and
# a killed job's leftovers can be large. SLURM here does not track TmpDisk.
FREE_GB=$(df -BG --output=avail "$(dirname "$P2B_WORKSPACE_ROOT")" | tail -1 | tr -dc '0-9')
# A node's first job builds the ~21 GB package cache; once it exists a job needs far less.
if [ -d "$P2B_UV_CACHE/archive-v0" ]; then NEED="${P2B_MIN_FREE_GB_CACHED:-15}"; else NEED="${P2B_MIN_FREE_GB:-40}"; fi
[ "${FREE_GB:-0}" -ge "$NEED" ] || {
    echo "only ${FREE_GB}G free under $(dirname "$P2B_WORKSPACE_ROOT") on $(hostname); need ${NEED}G" >&2; exit 3; }
mkdir -p "$P2B_WORKSPACE_ROOT" "$P2B_UV_CACHE"
chmod 700 "$P2B_UV_CACHE"
# Named by the array id so they match logs/p2b-align-<array>_<index>; SLURM_JOB_ID alone is
# the per-task raw id, which nothing else shows.
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JOB_WORKSPACE_ROOT="$P2B_WORKSPACE_ROOT/$JOB_TAG"
JOB_RUNS_ROOT="$P2B_RUNS_ROOT/$JOB_TAG"
mkdir -p "$JOB_WORKSPACE_ROOT" "$JOB_RUNS_ROOT" "$P2B_RUNS_ROOT/done"
# The node's shared uv cache can hold package copies that lost files (before agents saw it read-only, a run
# could delete through its environment's symlinks), and every environment linked to one is broken. Remove
# those copies first so uv extracts them again; one job per node at a time. The report stays with the run.
{ flock -w 900 9 && python3 "$P2B_REPO/src/genomics_agent_bench/env_health.py" cache "$P2B_UV_CACHE" --repair; } \
    9>"$P2B_UV_CACHE/.p2b-repair.lock" > "$JOB_RUNS_ROOT/uv_cache_check.txt" 2>&1 || true
tail -1 "$JOB_RUNS_ROOT/uv_cache_check.txt" 2>/dev/null | sed "s/^/[$(hostname)] uv cache: /" >&2
# SLURM sends TERM two minutes before the wall clock (--signal above), and scancel sends it
# too: remove the workspace then, so a killed job cannot leave gigabytes on the node. The
# runner's own state cleanup runs on its exit as well.
# The runner is started in the background and waited on: bash runs a trap only between
# foreground commands, so a foreground runner would keep the trap from firing until SLURM
# had already killed everything.
RUNNER_PID=""
cleanup_on_kill() {
    echo "[$(date -Is)] killed; stopping the runner and removing the workspace" >&2
    [ -n "$RUNNER_PID" ] && kill -TERM "$RUNNER_PID" 2>/dev/null && wait "$RUNNER_PID" 2>/dev/null
    rm -rf "$JOB_WORKSPACE_ROOT"
    exit 143
}
trap cleanup_on_kill TERM INT
# Keep the matrix row with the run, so a run directory can be matched back to its rep.
echo "$ROW" > "$JOB_RUNS_ROOT/row.csv"
DONE_KEY="${AGENT}__${MODEL}__${TASK}__${CONDITION}__rep${REP}"

# Containment runtime. The runner exits if this is missing rather than running non-blind.
# With P2B_CONTAINER_MODULE unset, try the two usual module names before giving up, so a
# fresh login that forgot to export it still runs.
have_runtime() { command -v "$P2B_SINGULARITY_BIN" >/dev/null || command -v apptainer >/dev/null; }
# `module` is a shell function; a batch shell that did not inherit it needs Lmod initialised.
if ! type module >/dev/null 2>&1; then
    for init in "${LMOD_PKG:-}/init/bash" /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -r "$init" ] && { source "$init"; break; }
    done
fi
if [ -n "$P2B_CONTAINER_MODULE" ]; then
    module load "$P2B_CONTAINER_MODULE" || { echo "could not load $P2B_CONTAINER_MODULE" >&2; exit 3; }
elif ! have_runtime; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi
have_runtime || {
    echo "no container runtime on PATH; set P2B_CONTAINER_MODULE or P2B_SINGULARITY_BIN" >&2; exit 3; }
# Resolve the symlink so the manifest names the build that actually ran.
case "$P2B_SINGULARITY_IMAGE" in docker://*) ;; *)
    [ -f "$P2B_SINGULARITY_IMAGE" ] || { echo "image not found: $P2B_SINGULARITY_IMAGE" >&2; exit 3; }
    P2B_SINGULARITY_IMAGE="$(readlink -f "$P2B_SINGULARITY_IMAGE")" ;; esac
echo "container: $(command -v "$P2B_SINGULARITY_BIN" || command -v apptainer)  image: $P2B_SINGULARITY_IMAGE"

GPU_FLAG=()
[ "$P2B_GPU" = "1" ] && GPU_FLAG=(--gpu)

# The warm-start condition hands the agent this model's own best scored bare submission for
# the task as the current answer; everything else matches the equipped condition.
WARM_FLAG=()
if [ "$CONDITION" = "warm" ]; then
    PRIOR=$("$P2B_PYTHON" scripts/find_bare_submission.py "$TASK" "$MODEL" 2>/dev/null) || {
        echo "no scored bare submission for $MODEL on $TASK; cannot warm start" >&2; exit 4; }
    WARM_FLAG=(--warm-start-from "$PRIOR")
    echo "warm start from: $PRIOR"
fi

# The generalization experiment's guided condition ("guided:<cell>"): a bare run that also gets the strategy
# synthesized in that cell (PLAYBOOK.md and policy/) as strategy_guide/. No prior alignment is put on disk.
GUIDE_FLAG=()
case "$CONDITION" in guided:*)
    GUIDE="${P2B_GEN_ROOT:-/dcs07/chen/data/alignment-agent/generalization}/synthesis/${CONDITION#guided:}/silo/output"
    [ -f "$GUIDE/PLAYBOOK.md" ] || { echo "no synthesized strategy at $GUIDE" >&2; exit 4; }
    GUIDE_FLAG=(--guidance-dir "$GUIDE")
    echo "strategy guide: $GUIDE" ;;
esac

run_agent() {
if [ "$AGENT" = "azure-openai" ]; then
    # Azure credentials are read only by the host-side request code. --cleanenv keeps them
    # unavailable to the generated solver, which receives only the sanitized workspace.
    "$P2B_PYTHON" scripts/run_openai_solver.py \
        "benchmarks/spatial_alignment/tasks/$TASK/task.toml" \
        --provider azure \
        --model "$MODEL" \
        --blind-solve \
        --blind-mode container \
        --container-bin "$P2B_SINGULARITY_BIN" \
        --container-image "$P2B_SINGULARITY_IMAGE" \
        --workspace-root "$JOB_WORKSPACE_ROOT" \
        --runs-root "$JOB_RUNS_ROOT" \
        --timeout-minutes "$P2B_TIMEOUT_MINUTES" \
        --max-output-tokens "$P2B_MAX_OUTPUT_TOKENS" \
        --api-timeout-seconds "$P2B_API_TIMEOUT" \
        --force
else
    "$P2B_PYTHON" scripts/run_local_coding_agent.py \
        "benchmarks/spatial_alignment/tasks/$TASK/task.toml" \
        --agent "$AGENT" \
        --model "$MODEL" \
        --prompt-variant "$PROMPT_VARIANT" \
        --prebootstrap-method-venv "$PREBOOTSTRAP" \
        --blind-solve \
        --blind-mode container \
        --container-bin "$P2B_SINGULARITY_BIN" \
        --container-image "$P2B_SINGULARITY_IMAGE" \
        "${GPU_FLAG[@]}" "${WARM_FLAG[@]}" "${GUIDE_FLAG[@]}" \
        --uv-cache "$P2B_UV_CACHE" \
        --workspace-root "$JOB_WORKSPACE_ROOT" \
        --runs-root "$JOB_RUNS_ROOT" \
        --timeout-minutes "$P2B_TIMEOUT_MINUTES" \
        --score-timeout-seconds "$P2B_SCORE_TIMEOUT" \
        ${P2B_CLAUDE_EFFORT:+--claude-effort "$P2B_CLAUDE_EFFORT"}
fi
}
set +e
run_agent &
RUNNER_PID=$!
wait "$RUNNER_PID"; RUNNER_RC=$?
set -e
echo "[$(date -Is)] runner exited $RUNNER_RC"

# Mark the row done only on a scored run, so a resubmission retries failures and skips the
# rest. The workspace is node-local and never outlives the job: everything worth keeping
# (scripts, outputs, report) is already snapshotted into the run directory, and the method
# environments it holds run to gigabytes, which filled /tmp on shared nodes.
STATUS=$("$P2B_PYTHON" - "$JOB_RUNS_ROOT" <<'PY'
import json, sys
from pathlib import Path
for m in Path(sys.argv[1]).glob("*/manifest.json"):
    print(json.loads(m.read_text()).get("status", "")); break
PY
)
echo "[$(date -Is)] status=${STATUS:-none}"
rm -rf "$JOB_WORKSPACE_ROOT"
if [ "$STATUS" = "scored" ]; then
    touch "$P2B_RUNS_ROOT/done/$DONE_KEY"
else
    echo "not marked done; see $JOB_RUNS_ROOT"
    exit 1
fi
