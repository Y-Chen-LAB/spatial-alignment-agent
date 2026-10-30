#!/usr/bin/env bash
#SBATCH --job-name=gen-synth
#SBATCH --output=/dcs07/chen/data/alignment-agent/generalization/logs/synth-%A_%a.out
#SBATCH --error=/dcs07/chen/data/alignment-agent/generalization/logs/synth-%A_%a.out
# Generalization experiment, step 1: one synthesis agent per array index. Rows file (GEN_ROWS): one cell name per
# line, e.g. family-train4-f0-scored (split-scheme-fold-visibility). Builds the cell's silo if needed, runs the
# agent in it (scripts/generalization/run_synthesis.py), and when a policy was delivered submits step 2, the
# direct application of that policy to every held-out task of the cell (generalization_apply.sbatch).
set -uo pipefail
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
GEN=/dcs07/chen/data/alignment-agent/generalization
CELL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GEN_ROWS:?set GEN_ROWS}")
[ -n "$CELL" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
IFS=- read -r SPLIT SCHEME F VIS SUFFIX <<<"$CELL"   # optional suffix "sol": solution-only silo (build_silo --records solution)
echo "[$(date -Is)] $(hostname) cell=$CELL"
command -v "$P2B_SINGULARITY_BIN" >/dev/null || export P2B_SINGULARITY_BIN=/jhpce/shared/jhpce/core/singularity/3.11.4/bin/singularity
export P2B_SINGULARITY_IMAGE="$(readlink -f "$P2B_SINGULARITY_IMAGE")"
export P2B_AGENT_RUNTIME="${P2B_AGENT_RUNTIME:-/users/ychen5/p2b-agent-runtime/claude-2.1.281}"
mkdir -p "$P2B_UV_CACHE"; chmod 700 "$P2B_UV_CACHE"

[ -f "$GEN/synthesis/$CELL/silo/SYNTHESIS.md" ] || \
    "$P2B_PYTHON" scripts/generalization/build_silo.py --split "$SPLIT" --scheme "$SCHEME" --fold "${F#f}" --visibility "$VIS" ${SUFFIX:+--records solution --suffix="-$SUFFIX"} || exit 2
"$P2B_PYTHON" scripts/generalization/run_synthesis.py "$GEN/synthesis/$CELL" --timeout-minutes "${GEN_SYNTH_MINUTES:-720}" --effort "${GEN_SYNTH_EFFORT:-high}"
RC=$?
# Large copies of task data the agent left in scratch/ are not deliverables.
find "$GEN/synthesis/$CELL/silo/scratch" -name '*.h5ad' -delete 2>/dev/null
echo "[$(date -Is)] synthesis exited $RC"

# Apply only a finished delivery: all three deliverables present and the agent's final result not an error
# (a run killed mid-way, e.g. by an expired login, can leave a draft run.sh behind).
OUT="$GEN/synthesis/$CELL/silo/output"
AGENT_ERR=$(tail -n 50 "$GEN/synthesis/$CELL/run/raw_stdout.jsonl" 2>/dev/null | grep '"type":"result"' | tail -1 | grep -c '"is_error":true')
if [ -f "$OUT/policy/run.sh" ] && [ -f "$OUT/PLAYBOOK.md" ] && [ -f "$OUT/NOTES.md" ] && [ "$AGENT_ERR" = 0 ] && [ "${GEN_AUTO_APPLY:-1}" = 1 ]; then
    "$P2B_PYTHON" scripts/generalization/submit_apply.py "$CELL"
else
    echo "no finished policy for $CELL (agent error=$AGENT_ERR); nothing applied"
fi
exit $RC
