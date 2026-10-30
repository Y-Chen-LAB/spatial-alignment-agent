#!/usr/bin/env bash
#SBATCH --job-name=gen-apply
#SBATCH --output=/dcs07/chen/data/alignment-agent/generalization/logs/apply-%A_%a.out
#SBATCH --error=/dcs07/chen/data/alignment-agent/generalization/logs/apply-%A_%a.out
# Generalization experiment, step 2: one (cell, held-out task) per array index. Rows file (GEN_ROWS):
# "<cell>\t<task>" per line. Runs the cell's synthesized policy on the task with no model in the loop and
# scores it (scripts/generalization/apply_policy.py). Submitted by scripts/generalization/submit_apply.py.
set -uo pipefail
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
GEN=/dcs07/chen/data/alignment-agent/generalization
IFS=$'\t' read -r CELL TASK <<<"$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GEN_ROWS:?set GEN_ROWS}")"
[ -n "$TASK" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "[$(date -Is)] $(hostname) cell=$CELL task=$TASK"
command -v "$P2B_SINGULARITY_BIN" >/dev/null || export P2B_SINGULARITY_BIN=/jhpce/shared/jhpce/core/singularity/3.11.4/bin/singularity
export P2B_SINGULARITY_IMAGE="$(readlink -f "$P2B_SINGULARITY_IMAGE")"
mkdir -p "$P2B_UV_CACHE"; chmod 700 "$P2B_UV_CACHE"
"$P2B_PYTHON" scripts/generalization/apply_policy.py "$GEN/synthesis/$CELL" "$TASK"
