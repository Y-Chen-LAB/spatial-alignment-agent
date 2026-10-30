#!/usr/bin/env bash
#SBATCH --job-name=p2b-baseline
#SBATCH --output=logs/p2b-baseline-%A_%a.out
#SBATCH --error=logs/p2b-baseline-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
# Scripted (non-agent) baselines for one task, so every agent score has references beside it.
# One array task per task.toml; each runs every method in P2B_BASELINE_METHODS.
#
#   ls benchmarks/spatial_alignment/tasks/*/task.toml > $ROWS
#   sbatch --array=0-$(($(wc -l < $ROWS)-1))%20 --export=ALL,P2B_TASKS=$ROWS scripts/slurm/run_baselines.sbatch
set -uo pipefail
source "${P2B_REPO:-$SLURM_SUBMIT_DIR}/scripts/slurm/site_config.sh"
cd "$P2B_REPO"
: "${P2B_BASELINE_METHODS:=identity centroid_similarity pca_similarity icp_similarity}"
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${P2B_TASKS:?set P2B_TASKS to a file of task.toml paths}")
[ -n "$TASK" ] || { echo "no task for index $SLURM_ARRAY_TASK_ID"; exit 1; }
export HDF5_USE_FILE_LOCKING=FALSE
echo "[$(date -Is)] $TASK"
for m in $P2B_BASELINE_METHODS; do
    echo "--- $m"
    timeout ${P2B_BASELINE_TIMEOUT:-3000} "$P2B_PYTHON" scripts/run_task_baseline.py "$TASK" --method "$m" --max-genes 100 \
        --output-root results/baselines || echo "    $m failed or timed out"
    # re-score with the agents' scorer, so baselines carry the same metrics (incl. paper_metrics) as agent runs
    tid=$("$P2B_PYTHON" -c "import tomllib,sys;print(tomllib.load(open(sys.argv[1],'rb'))['id'])" "$TASK")
    out="results/baselines/$tid/$m"
    [ -d "$out/aligned_coordinates" ] && timeout ${P2B_SCORE_TIMEOUT:-3600} "$P2B_PYTHON" scripts/score_agent_submission.py "$TASK" "$out" \
        --output "$out/scores.json" >/dev/null 2>"$out/rescore_stderr.txt" || echo "    $m rescoring failed"
done
echo "[$(date -Is)] done"
